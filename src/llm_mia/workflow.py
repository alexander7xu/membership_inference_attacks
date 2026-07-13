from __future__ import annotations

import gc
import logging
import math
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score

from src.experiment import (
    collect_environment,
    collect_git_state,
    config_fingerprint,
    seed_everything,
    write_experiment_markdown,
    write_yaml,
)
from src.llm_mia.analysis import (
    bootstrap_binary_metrics,
    candidate_relative_loglikelihood,
    online_rmia_score,
    population_relative_loglikelihood,
)
from src.llm_mia.data import (
    QARecord,
    candidate_id,
    deduplicate_candidate_rows,
    deterministic_subset,
    deterministic_unique_subset,
    file_sha256,
    public_candidate,
    read_json,
    read_jsonl,
    read_shadow_masks,
    record_from_public_candidate,
    record_to_json,
    split_records,
    split_squad_train,
    squad_row_to_record,
    text_sha256,
    trivia_row_to_record,
    write_json,
    write_jsonl,
    write_shadow_masks,
)
from src.llm_mia.hf import (
    attach_lora,
    ensure_token_embeddings,
    generate_completions,
    load_causal_lm,
    load_model_for_inference,
    load_tokenizer,
    make_trainer,
    materialize_records,
    score_records,
)
from src.llm_mia.metrics import exact_match, mean, perplexity, token_f1

LOGGER = logging.getLogger(__name__)
_ACTIVE_WANDB_RUN: ContextVar[Any | None] = ContextVar("active_wandb_run", default=None)


def run_stage(cfg: DictConfig, *, command: str) -> None:
    project_root = Path.cwd()
    seed_everything(
        int(cfg.runtime.seed), deterministic=bool(cfg.runtime.deterministic)
    )
    wandb_run = _start_wandb_run(cfg, project_root=project_root, command=command)
    run_token = _ACTIVE_WANDB_RUN.set(wandb_run)
    try:
        stage = str(cfg.workflow.stage)
        if stage == "smoke":
            run_pipeline(cfg, project_root=project_root, command=command, smoke=True)
        elif stage == "formal":
            run_pipeline(cfg, project_root=project_root, command=command, smoke=False)
        elif stage == "prepare":
            prepare_data(cfg, project_root)
        elif stage == "train_target":
            train_target(cfg, project_root=project_root, command=command, smoke=False)
        elif stage == "evaluate":
            evaluate_all(cfg, project_root=project_root, command=command, smoke=False)
        elif stage == "generate":
            generate_target_candidates(
                cfg, project_root=project_root, command=command, smoke=False
            )
        elif stage == "build_candidates":
            build_candidate_set(cfg, project_root=project_root, smoke=False)
        elif stage == "train_shadows":
            train_shadows(cfg, project_root=project_root, command=command, smoke=False)
        elif stage == "attack":
            run_attack(cfg, project_root=project_root, command=command, smoke=False)
        else:
            raise ValueError(f"Unknown workflow.stage: {stage}")
    finally:
        _ACTIVE_WANDB_RUN.reset(run_token)
        if wandb_run is not None:
            wandb_run.finish()


def run_pipeline(
    cfg: DictConfig,
    *,
    project_root: Path,
    command: str,
    smoke: bool,
) -> None:
    prepare_data(cfg, project_root)
    train_target(cfg, project_root=project_root, command=command, smoke=smoke)
    evaluate_all(cfg, project_root=project_root, command=command, smoke=smoke)
    generate_target_candidates(
        cfg, project_root=project_root, command=command, smoke=smoke
    )
    build_candidate_set(cfg, project_root=project_root, smoke=smoke)
    train_shadows(cfg, project_root=project_root, command=command, smoke=smoke)
    run_attack(cfg, project_root=project_root, command=command, smoke=smoke)


def profile_name(cfg: DictConfig, smoke: bool) -> str:
    return "smoke" if smoke else str(cfg.workflow.profile)


def model_root(cfg: DictConfig, project_root: Path, *, smoke: bool) -> Path:
    return (
        project_root
        / str(cfg.paths.output_root)
        / profile_name(cfg, smoke)
        / str(cfg.model.key)
    )


def _start_wandb_run(
    cfg: DictConfig, *, project_root: Path, command: str
) -> Any | None:
    if not bool(cfg.wandb.enabled):
        return None
    import wandb

    local_dir = (
        model_root(
            cfg,
            project_root,
            smoke=str(cfg.workflow.stage) == "smoke",
        )
        / "wandb"
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=str(cfg.wandb.project),
        entity=None if cfg.wandb.entity is None else str(cfg.wandb.entity),
        mode=str(cfg.wandb.mode),
        dir=str(local_dir),
        name=(f"{cfg.workflow.profile}-{cfg.model.key}-seed{int(cfg.runtime.seed)}"),
        tags=[str(tag) for tag in cfg.wandb.tags],
        config=OmegaConf.to_container(cfg, resolve=True),
        notes=command,
    )


def _active_wandb_run_id() -> str | None:
    run = _ACTIVE_WANDB_RUN.get()
    return None if run is None else str(run.id)


def _log_wandb_stage(
    *,
    stage: str,
    metrics: dict[str, float],
    paths: dict[str, Path],
) -> dict[str, str]:
    run = _ACTIVE_WANDB_RUN.get()
    if run is None:
        return {}
    import wandb

    run.log({f"{stage}/{key}": value for key, value in metrics.items()})
    artifact = wandb.Artifact(
        name=f"{run.id}-{stage}",
        type="experiment-output",
        metadata={"stage": stage, "wandb_run_id": str(run.id)},
    )
    for name, path in paths.items():
        if path.is_dir():
            artifact.add_dir(str(path), name=name)
        elif path.exists():
            artifact.add_file(str(path), name=name)
    run.log_artifact(artifact)
    return {f"wandb_{stage}_artifact": f"{artifact.name}:{artifact.digest}"}


def data_root(cfg: DictConfig, project_root: Path) -> Path:
    return project_root / str(cfg.paths.data_dir)


def _data_manifest_expected(cfg: DictConfig) -> dict[str, Any]:
    return {
        "squad_dataset": str(cfg.data.squad_dataset),
        "squad_config": str(cfg.data.squad_config),
        "squad_revision": str(cfg.data.squad_revision),
        "trivia_dataset": str(cfg.data.trivia_dataset),
        "trivia_config": str(cfg.data.trivia_config),
        "trivia_revision": str(cfg.data.trivia_revision),
        "seed": int(cfg.runtime.seed),
        "split_rule": "stable sha256(seed:namespace:record_id)",
        "counts": {
            "squad_target_train": int(cfg.data.target_train_size),
            "squad_attacker_auxiliary_pool": int(cfg.data.auxiliary_pool_size),
            "squad_validation_candidates": int(
                cfg.data.squad_validation_candidate_size
            ),
            "squad_validation_population": int(
                cfg.data.squad_validation_population_size
            ),
            "trivia_validation_candidates": int(
                cfg.data.trivia_validation_candidate_size
            ),
        },
    }


def _manifest_files_match(root: Path, manifest: dict[str, Any]) -> bool:
    file_hashes = manifest.get("file_sha256", {})
    if not isinstance(file_hashes, dict) or not file_hashes:
        return False
    return all(
        (root / name).exists() and file_sha256(root / name) == digest
        for name, digest in file_hashes.items()
    )


def prepare_data(cfg: DictConfig, project_root: Path) -> None:
    root = data_root(cfg, project_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "split_manifest.json"
    expected = _data_manifest_expected(cfg)
    if manifest_path.exists() and not bool(cfg.workflow.force):
        manifest = read_json(manifest_path)
        if all(manifest.get(key) == value for key, value in expected.items()) and (
            _manifest_files_match(root, manifest)
        ):
            return
        raise ValueError(
            "Existing data manifest does not match the formal configuration. "
            "Use workflow.force=true only to intentionally regenerate it."
        )

    train_split = split_squad_train(
        load_squad_records(cfg, "train"),
        seed=int(cfg.runtime.seed),
        target_train_size=int(cfg.data.target_train_size),
        auxiliary_pool_size=int(cfg.data.auxiliary_pool_size),
    )
    validation_split = split_records(
        load_squad_records(cfg, "validation"),
        seed=int(cfg.runtime.seed),
        namespace="squad_validation",
        sizes={
            "squad_validation_candidates": int(
                cfg.data.squad_validation_candidate_size
            ),
            "squad_validation_population": int(
                cfg.data.squad_validation_population_size
            ),
        },
    )
    trivia_candidates = deterministic_subset(
        load_trivia_records(cfg, "validation", limit=None),
        seed=int(cfg.runtime.seed),
        limit=int(cfg.data.trivia_validation_candidate_size),
        namespace="trivia_validation_candidates",
    )
    split = {
        **train_split,
        **validation_split,
        "trivia_validation_candidates": trivia_candidates,
    }
    for name, rows in split.items():
        write_jsonl(root / f"{name}.jsonl", (record_to_json(row) for row in rows))

    file_hashes = {
        f"{name}.jsonl": file_sha256(root / f"{name}.jsonl") for name in split
    }
    write_json(
        manifest_path,
        {**expected, "file_sha256": file_hashes},
    )
    _log_wandb_stage(
        stage="data",
        metrics={"split_files": float(len(file_hashes))},
        paths={"split_manifest": manifest_path},
    )


def load_squad_records(cfg: DictConfig, split: str) -> list[QARecord]:
    dataset = load_dataset(
        str(cfg.data.squad_dataset),
        str(cfg.data.squad_config),
        revision=str(cfg.data.squad_revision),
        split=split,
    )
    return [squad_row_to_record(dict(row), split) for row in dataset]


def load_trivia_records(
    cfg: DictConfig, split: str, *, limit: int | None
) -> list[QARecord]:
    dataset_split = split if limit is None else f"{split}[:{max(limit * 16, limit)}]"
    data_files = {
        split: (
            f"hf://datasets/{cfg.data.trivia_dataset}@{cfg.data.trivia_revision}/"
            f"{cfg.data.trivia_config}/{split}-*.parquet"
        )
    }
    dataset = load_dataset("parquet", data_files=data_files, split=dataset_split)
    rows: list[QARecord] = []
    for row in dataset:
        record = trivia_row_to_record(dict(row), split)
        if record is not None:
            rows.append(record)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def load_split_file(cfg: DictConfig, project_root: Path, name: str) -> list[QARecord]:
    prepare_data(cfg, project_root)
    from src.llm_mia.data import record_from_json

    path = data_root(cfg, project_root) / f"{name}.jsonl"
    return [record_from_json(row) for row in read_jsonl(path)]


def candidate_limit(cfg: DictConfig, smoke: bool) -> int:
    return int(cfg.data.smoke_candidate_limit if smoke else cfg.data.candidate_limit)


def eval_limit(cfg: DictConfig, smoke: bool) -> int | None:
    value = cfg.data.smoke_eval_limit if smoke else cfg.data.eval_limit
    return None if value is None else int(value)


def shadow_count(cfg: DictConfig, smoke: bool) -> int:
    return int(cfg.shadow.smoke_count if smoke else cfg.shadow.count)


def max_steps(cfg: DictConfig, smoke: bool) -> int | None:
    value = cfg.train.smoke_max_steps if smoke else cfg.train.max_steps
    return None if value is None else int(value)


def train_target(
    cfg: DictConfig,
    *,
    project_root: Path,
    command: str,
    smoke: bool,
) -> Path:
    run_dir = (
        model_root(cfg, project_root, smoke=smoke)
        / "target"
        / f"seed_{int(cfg.runtime.seed)}"
    )
    adapter_dir = run_dir / "adapter"
    if adapter_dir.joinpath("adapter_config.json").exists() and not bool(
        cfg.workflow.force
    ):
        LOGGER.info("Reusing existing target adapter: %s", adapter_dir)
        return adapter_dir
    records = load_split_file(cfg, project_root, "squad_target_train")
    if smoke:
        records = deterministic_subset(
            records,
            seed=int(cfg.runtime.seed),
            limit=max(8, candidate_limit(cfg, smoke)),
            namespace="smoke_target_train",
        )
    return train_lora(
        cfg,
        project_root=project_root,
        train_records=records,
        run_dir=run_dir,
        role="target",
        shadow_index=None,
        command=command,
        max_steps_value=max_steps(cfg, smoke),
    )


def train_shadows(
    cfg: DictConfig,
    *,
    project_root: Path,
    command: str,
    smoke: bool,
) -> list[Path]:
    build_candidate_set(cfg, project_root=project_root, smoke=smoke)
    candidates = read_jsonl(candidate_paths(cfg, project_root, smoke=smoke)["public"])
    masks = read_shadow_masks(candidate_paths(cfg, project_root, smoke=smoke)["masks"])
    auxiliary_pool = load_split_file(cfg, project_root, "squad_attacker_auxiliary_pool")
    adapters: list[Path] = []
    for index in range(shadow_count(cfg, smoke)):
        included = [
            record_from_public_candidate(row)
            for row in candidates
            if masks[row["candidate_id"]][index] == 1
        ]
        target_size = int(cfg.shadow.train_size)
        if smoke:
            target_size = max(8, min(target_size, candidate_limit(cfg, smoke) * 2))
        if len(included) > target_size:
            raise ValueError(
                f"Shadow {index} has {len(included)} included candidates for "
                f"target size {target_size}."
            )

        fill = deterministic_unique_subset(
            auxiliary_pool,
            seed=int(cfg.runtime.seed) + int(cfg.shadow.seed_offset) + index,
            limit=target_size - len(included),
            namespace=f"shadow_{index:02d}_fill",
            excluded_content_hashes={row.content_sha256 for row in included},
        )
        records = included + fill
        content_hashes = [record.content_sha256 for record in records]
        if len(fill) != target_size - len(included) or len(records) != target_size:
            raise ValueError(f"Shadow {index} could not reach {target_size} rows.")
        if len(set(content_hashes)) != len(content_hashes):
            raise ValueError(f"Shadow {index} contains duplicate record content.")

        run_dir = (
            model_root(cfg, project_root, smoke=smoke)
            / "shadows"
            / f"shadow_{index:02d}"
        )
        adapter_dir = run_dir / "adapter"
        if adapter_dir.joinpath("adapter_config.json").exists() and not bool(
            cfg.workflow.force
        ):
            LOGGER.info("Reusing existing shadow adapter: %s", adapter_dir)
            adapters.append(adapter_dir)
            continue
        write_jsonl(
            run_dir / "train_manifest.jsonl", (record_to_json(row) for row in records)
        )
        write_json(
            run_dir / "inclusion_summary.json",
            {
                "shadow_index": index,
                "included_candidates": len(included),
                "filled_from_auxiliary_pool": len(fill),
                "train_size": len(records),
            },
        )
        adapters.append(
            train_lora(
                cfg,
                project_root=project_root,
                train_records=records,
                run_dir=run_dir,
                role="shadow",
                shadow_index=index,
                command=command,
                max_steps_value=max_steps(cfg, smoke),
            )
        )
    return adapters


def train_lora(
    cfg: DictConfig,
    *,
    project_root: Path,
    train_records: list[QARecord],
    run_dir: Path,
    role: str,
    shadow_index: int | None,
    command: str,
    max_steps_value: int | None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(
        run_dir / str(cfg.report.resolved_config_filename),
        OmegaConf.to_container(cfg, resolve=True),
    )
    tokenizer = load_tokenizer(
        str(cfg.model.name_or_path),
        revision=str(cfg.model.revision),
        trust_remote_code=bool(cfg.tokenizer.trust_remote_code),
    )
    train_records = materialize_records(
        train_records,
        tokenizer,
        int(cfg.tokenizer.max_length),
        num_workers=int(cfg.tokenizer.preprocessing_workers),
    )
    write_jsonl(
        run_dir / "train_manifest.jsonl", (record_to_json(row) for row in train_records)
    )
    model = load_causal_lm(
        str(cfg.model.name_or_path),
        bf16=bool(cfg.precision.bf16),
        revision=str(cfg.model.revision),
        trust_remote_code=bool(cfg.tokenizer.trust_remote_code),
    )
    ensure_token_embeddings(model, tokenizer)
    model = attach_lora(model, cfg)
    model.print_trainable_parameters()
    trainer = make_trainer(
        model=model,
        tokenizer=tokenizer,
        train_records=train_records,
        cfg=cfg,
        output_dir=str(run_dir / "trainer"),
        max_steps=max_steps_value,
    )
    result = trainer.train()
    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    metrics = {
        f"train/{key}": float(value)
        for key, value in result.metrics.items()
        if _is_number(value)
    }
    metrics["train/records"] = float(len(train_records))
    metrics["train/trainable_parameters"] = float(
        _trainable_parameter_count(trainer.model)
    )
    write_json(run_dir / "metrics.json", metrics)
    stage_name = role if shadow_index is None else f"shadow_{shadow_index:02d}"
    tracking_artifacts = _log_wandb_stage(
        stage=stage_name,
        metrics=metrics,
        paths={
            "adapter": adapter_dir,
            "metrics": run_dir / "metrics.json",
            "resolved_config": run_dir / str(cfg.report.resolved_config_filename),
        },
    )
    write_experiment_markdown(
        run_dir / str(cfg.report.experiment_filename),
        purpose=f"{role} LoRA fine-tuning for {cfg.model.display_name} on SQuAD QA.",
        hypothesis=str(cfg.experiment.hypothesis),
        command=command,
        overrides=[],
        config_fingerprint_value=config_fingerprint(
            OmegaConf.to_container(cfg, resolve=True)
        ),
        git_state=collect_git_state(project_root),
        environment=extended_environment(project_root),
        artifacts={
            "adapter": str(adapter_dir),
            "metrics": str(run_dir / "metrics.json"),
            "resolved_config": str(run_dir / str(cfg.report.resolved_config_filename)),
            **tracking_artifacts,
        },
        metrics=metrics,
        conclusion=f"{role} LoRA adapter saved from the final epoch/step.",
        achieved_purpose=True,
        next_action="Evaluate utility or score candidates with this final adapter.",
        wandb_run_id=_active_wandb_run_id(),
    )
    del trainer, model
    cleanup_cuda()
    return adapter_dir


def evaluate_all(
    cfg: DictConfig,
    *,
    project_root: Path,
    command: str,
    smoke: bool,
) -> None:
    target_adapter = train_target(
        cfg, project_root=project_root, command=command, smoke=smoke
    )
    evaluate_model(
        cfg,
        project_root=project_root,
        adapter_path=None,
        run_name="base",
        command=command,
        smoke=smoke,
    )
    evaluate_model(
        cfg,
        project_root=project_root,
        adapter_path=target_adapter,
        run_name="target_lora",
        command=command,
        smoke=smoke,
    )


def evaluate_model(
    cfg: DictConfig,
    *,
    project_root: Path,
    adapter_path: Path | None,
    run_name: str,
    command: str,
    smoke: bool,
) -> None:
    run_dir = model_root(cfg, project_root, smoke=smoke) / "eval" / run_name
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not bool(cfg.workflow.force):
        LOGGER.info("Reusing existing eval metrics: %s", metrics_path)
        return
    limit = eval_limit(cfg, smoke)
    squad_validation = deterministic_subset(
        load_squad_records(cfg, "validation"),
        seed=int(cfg.runtime.seed),
        limit=limit,
        namespace="eval_squad_validation",
    )
    trivia_validation = deterministic_subset(
        load_trivia_records(cfg, "validation", limit=limit),
        seed=int(cfg.runtime.seed),
        limit=limit,
        namespace="eval_trivia_validation",
    )
    model, tokenizer = load_model_for_inference(
        cfg,
        adapter_path=str(adapter_path) if adapter_path is not None else None,
    )
    metrics: dict[str, float] = {}
    for name, records in [
        ("squad_validation", squad_validation),
        ("trivia_validation", trivia_validation),
    ]:
        loss_metrics = completion_metrics(
            model,
            tokenizer,
            records,
            batch_size=int(cfg.eval.batch_size),
            max_length=int(cfg.tokenizer.max_length),
            loss_chunk_tokens=int(cfg.eval.loss_chunk_tokens),
            tokenization_workers=int(cfg.tokenizer.preprocessing_workers),
        )
        qa_metrics = generation_metrics(
            model,
            tokenizer,
            records,
            cfg=cfg,
            limit=limit,
        )
        metrics.update({f"{name}/{key}": value for key, value in loss_metrics.items()})
        metrics.update({f"{name}/{key}": value for key, value in qa_metrics.items()})
    write_json(metrics_path, metrics)
    write_yaml(
        run_dir / str(cfg.report.resolved_config_filename),
        OmegaConf.to_container(cfg, resolve=True),
    )
    tracking_artifacts = _log_wandb_stage(
        stage=f"eval_{run_name}",
        metrics=metrics,
        paths={
            "metrics": metrics_path,
            "resolved_config": run_dir / str(cfg.report.resolved_config_filename),
        },
    )
    write_experiment_markdown(
        run_dir / str(cfg.report.experiment_filename),
        purpose=f"Evaluate {run_name} {cfg.model.display_name} on SQuAD IID and TriviaQA OOD QA.",
        hypothesis="Final LoRA should reduce completion loss and improve QA EM/F1 versus the base model.",
        command=command,
        overrides=[],
        config_fingerprint_value=config_fingerprint(
            OmegaConf.to_container(cfg, resolve=True)
        ),
        git_state=collect_git_state(project_root),
        environment=extended_environment(project_root),
        artifacts={
            "metrics": str(metrics_path),
            "adapter": str(adapter_path or "base-model"),
            **tracking_artifacts,
        },
        metrics=metrics,
        conclusion="Evaluation completed without checkpoint selection.",
        achieved_purpose=True,
        next_action="Use final target adapter for generated candidates and RMIA scoring.",
        wandb_run_id=_active_wandb_run_id(),
    )
    del model
    cleanup_cuda()


def completion_metrics(
    model: Any,
    tokenizer: Any,
    records: list[QARecord],
    *,
    batch_size: int,
    max_length: int,
    loss_chunk_tokens: int,
    tokenization_workers: int,
) -> dict[str, float]:
    stats = score_records(
        model,
        tokenizer,
        records,
        batch_size=batch_size,
        max_length=max_length,
        loss_chunk_tokens=loss_chunk_tokens,
        tokenization_workers=tokenization_workers,
    )
    total_logprob = sum(item.token_logprob_sum for item in stats)
    total_tokens = sum(item.token_count for item in stats)
    loss = -total_logprob / max(total_tokens, 1)
    return {
        "completion_loss": float(loss),
        "completion_perplexity": perplexity(loss),
        "completion_tokens": float(total_tokens),
        "examples": float(len(stats)),
    }


def generation_metrics(
    model: Any,
    tokenizer: Any,
    records: list[QARecord],
    *,
    cfg: DictConfig,
    limit: int | None,
) -> dict[str, float]:
    rows = records if limit is None else records[:limit]
    completions = generate_completions(
        model,
        tokenizer,
        rows,
        batch_size=int(cfg.eval.generation_batch_size),
        max_length=int(cfg.tokenizer.max_length),
        max_new_tokens=int(cfg.generation.max_new_tokens),
        do_sample=bool(cfg.generation.do_sample),
        temperature=float(cfg.generation.temperature),
    )
    em = [
        exact_match(prediction, record.answers)
        for prediction, record in zip(completions, rows, strict=True)
    ]
    f1 = [
        token_f1(prediction, record.answers)
        for prediction, record in zip(completions, rows, strict=True)
    ]
    return {
        "generation_exact_match": mean(em),
        "generation_f1": mean(f1),
        "generation_examples": float(len(rows)),
    }


def generate_target_candidates(
    cfg: DictConfig,
    *,
    project_root: Path,
    command: str,
    smoke: bool,
) -> None:
    target_adapter = train_target(
        cfg, project_root=project_root, command=command, smoke=smoke
    )
    run_dir = model_root(cfg, project_root, smoke=smoke) / "generated"
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and not bool(cfg.workflow.force):
        LOGGER.info("Reusing generated candidate artifacts: %s", manifest_path)
        return
    limit = candidate_limit(cfg, smoke)
    target_train = deterministic_subset(
        load_split_file(cfg, project_root, "squad_target_train"),
        seed=int(cfg.runtime.seed),
        limit=limit,
        namespace="gen_from_squad_train",
    )
    squad_validation = deterministic_subset(
        load_split_file(cfg, project_root, "squad_validation_candidates"),
        seed=int(cfg.runtime.seed),
        limit=limit,
        namespace="gen_from_squad_validation",
    )
    trivia_validation = deterministic_subset(
        load_split_file(cfg, project_root, "trivia_validation_candidates"),
        seed=int(cfg.runtime.seed),
        limit=limit,
        namespace="gen_from_trivia_validation",
    )
    model, tokenizer = load_model_for_inference(cfg, adapter_path=str(target_adapter))
    groups = {
        "gen_from_squad_train": materialize_records(
            target_train,
            tokenizer,
            int(cfg.tokenizer.max_length),
            num_workers=int(cfg.tokenizer.preprocessing_workers),
        ),
        "gen_from_squad_validation": materialize_records(
            squad_validation,
            tokenizer,
            int(cfg.tokenizer.max_length),
            num_workers=int(cfg.tokenizer.preprocessing_workers),
        ),
        "gen_from_trivia_validation": materialize_records(
            trivia_validation,
            tokenizer,
            int(cfg.tokenizer.max_length),
            num_workers=int(cfg.tokenizer.preprocessing_workers),
        ),
    }
    train_hashes = {
        record.content_sha256
        for record in materialize_records(
            load_split_file(cfg, project_root, "squad_target_train"),
            tokenizer,
            int(cfg.tokenizer.max_length),
            num_workers=int(cfg.tokenizer.preprocessing_workers),
        )
    }
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    metrics: dict[str, float] = {}
    for group_name, records in groups.items():
        completions = generate_completions(
            model,
            tokenizer,
            records,
            batch_size=int(cfg.generation.batch_size),
            max_length=int(cfg.tokenizer.max_length),
            max_new_tokens=int(cfg.generation.max_new_tokens),
            do_sample=bool(cfg.generation.do_sample),
            temperature=float(cfg.generation.temperature),
        )
        em_values: list[float] = []
        f1_values: list[float] = []
        generated_records: list[QARecord] = []
        exact_reconstructions = 0
        seen: set[str] = set()
        for source, completion_text in zip(records, completions, strict=True):
            completion = f" {completion_text.strip() or '<empty>'}"
            generated = QARecord(
                record_id=f"{group_name}:{source.record_id}",
                dataset="generated",
                split=group_name,
                prompt=source.prompt,
                completion=completion,
                answers=source.answers,
            )
            generated_records.append(generated)
            cid = candidate_id(f"cand_{group_name}", generated)
            is_exact_member = int(generated.content_sha256 in train_hashes)
            exact_reconstructions += is_exact_member
            public_rows.append(
                {
                    **public_candidate(generated, cid),
                    "model_run_id": str(target_adapter),
                    "generation_config": {
                        "do_sample": bool(cfg.generation.do_sample),
                        "temperature": float(cfg.generation.temperature),
                        "max_new_tokens": int(cfg.generation.max_new_tokens),
                    },
                    "tokenizer_id": str(cfg.model.name_or_path),
                }
            )
            private_rows.append(
                {
                    "candidate_id": cid,
                    "private_group": group_name,
                    "source_id": source.record_id,
                    "source_prompt_membership": int(
                        group_name == "gen_from_squad_train"
                    ),
                    "exact_reconstruction": is_exact_member,
                    "record_membership_label": is_exact_member,
                }
            )
            em_values.append(exact_match(completion_text, source.answers))
            f1_values.append(token_f1(completion_text, source.answers))
            seen.add(generated.content_sha256)
        metrics[f"{group_name}/exact_match"] = mean(em_values)
        metrics[f"{group_name}/f1"] = mean(f1_values)
        metrics[f"{group_name}/duplicate_rate"] = 1.0 - (
            len(seen) / max(len(records), 1)
        )
        metrics[f"{group_name}/exact_reconstruction_rate"] = (
            exact_reconstructions / max(len(records), 1)
        )
        generated_completion = completion_metrics(
            model,
            tokenizer,
            generated_records,
            batch_size=int(cfg.eval.batch_size),
            max_length=int(cfg.tokenizer.max_length),
            loss_chunk_tokens=int(cfg.eval.loss_chunk_tokens),
            tokenization_workers=int(cfg.tokenizer.preprocessing_workers),
        )
        metrics.update(
            {
                f"{group_name}/{key}": value
                for key, value in generated_completion.items()
            }
        )
        generated_lengths = [
            len(tokenizer(record.completion, add_special_tokens=False).input_ids)
            for record in generated_records
        ]
        metrics[f"{group_name}/generated_completion_tokens_mean"] = mean(
            [float(length) for length in generated_lengths]
        )
        metrics[f"{group_name}/generated_completion_tokens_min"] = float(
            min(generated_lengths, default=0)
        )
        metrics[f"{group_name}/generated_completion_tokens_max"] = float(
            max(generated_lengths, default=0)
        )
    public_path = run_dir / "public_generated.jsonl"
    private_path = run_dir / "private_generated_labels.jsonl"
    metrics_path = run_dir / "metrics.json"
    write_jsonl(public_path, public_rows)
    write_jsonl(private_path, private_rows)
    write_json(metrics_path, metrics)
    manifest = artifact_manifest(
        cfg,
        project_root,
        {
            "public_generated": public_path,
            "private_generated_labels": private_path,
            "metrics": metrics_path,
        },
        row_counts={
            "public_generated": len(public_rows),
            "private_generated_labels": len(private_rows),
        },
    )
    write_json(manifest_path, manifest)
    tracking_artifacts = _log_wandb_stage(
        stage="generated",
        metrics=metrics,
        paths={
            "public_generated": public_path,
            "private_generated_labels": private_path,
            "metrics": metrics_path,
            "manifest": manifest_path,
        },
    )
    write_experiment_markdown(
        run_dir / str(cfg.report.experiment_filename),
        purpose=f"Generate reusable target-model candidate completions for {cfg.model.display_name}.",
        hypothesis="Generated records will mostly be non-members unless prompt and generated completion exactly reconstruct a target training record.",
        command=command,
        overrides=[],
        config_fingerprint_value=config_fingerprint(
            OmegaConf.to_container(cfg, resolve=True)
        ),
        git_state=collect_git_state(project_root),
        environment=extended_environment(project_root),
        artifacts={
            **{name: str(path) for name, path in manifest["files"].items()},
            **tracking_artifacts,
        },
        metrics=metrics,
        conclusion="Generated artifacts and private provenance labels were saved for reuse.",
        achieved_purpose=True,
        next_action="Build public attack candidates and shadow inclusion masks.",
        wandb_run_id=_active_wandb_run_id(),
    )
    del model
    cleanup_cuda()


def candidate_paths(
    cfg: DictConfig, project_root: Path, *, smoke: bool
) -> dict[str, Path]:
    root = model_root(cfg, project_root, smoke=smoke) / "candidates"
    return {
        "public": root / "public_candidates.jsonl",
        "private": root / "private_labels.jsonl",
        "masks": root / "shadow_masks.csv",
        "manifest": root / "manifest.json",
    }


def build_candidate_set(cfg: DictConfig, *, project_root: Path, smoke: bool) -> None:
    paths = candidate_paths(cfg, project_root, smoke=smoke)
    if paths["manifest"].exists() and not bool(cfg.workflow.force):
        existing_public = read_jsonl(paths["public"])
        existing_private = read_jsonl(paths["private"])
        unique_public, _ = deduplicate_candidate_rows(existing_public, existing_private)
        if len(unique_public) == len(existing_public):
            LOGGER.info("Reusing candidate set: %s", paths["manifest"])
            return
        LOGGER.warning(
            "Rebuilding candidate set because %d duplicate content rows were found.",
            len(existing_public) - len(unique_public),
        )
    generate_target_candidates(
        cfg,
        project_root=project_root,
        command="build_candidate_set dependency generation",
        smoke=smoke,
    )
    limit = candidate_limit(cfg, smoke)
    tokenizer = load_tokenizer(
        str(cfg.model.name_or_path),
        revision=str(cfg.model.revision),
        trust_remote_code=bool(cfg.tokenizer.trust_remote_code),
    )
    target_train = materialize_records(
        deterministic_subset(
            load_split_file(cfg, project_root, "squad_target_train"),
            seed=int(cfg.runtime.seed),
            limit=limit,
            namespace="gold_squad_target_train",
        ),
        tokenizer,
        int(cfg.tokenizer.max_length),
        num_workers=int(cfg.tokenizer.preprocessing_workers),
    )
    squad_validation = materialize_records(
        deterministic_subset(
            load_split_file(cfg, project_root, "squad_validation_candidates"),
            seed=int(cfg.runtime.seed),
            limit=limit,
            namespace="gold_squad_validation",
        ),
        tokenizer,
        int(cfg.tokenizer.max_length),
        num_workers=int(cfg.tokenizer.preprocessing_workers),
    )
    trivia_validation = materialize_records(
        deterministic_subset(
            load_split_file(cfg, project_root, "trivia_validation_candidates"),
            seed=int(cfg.runtime.seed),
            limit=limit,
            namespace="gold_trivia_validation",
        ),
        tokenizer,
        int(cfg.tokenizer.max_length),
        num_workers=int(cfg.tokenizer.preprocessing_workers),
    )
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for group_name, rows, member in [
        ("gold_squad_target_train", target_train, 1),
        ("gold_squad_validation", squad_validation, 0),
        ("gold_trivia_validation", trivia_validation, 0),
    ]:
        for record in rows:
            cid = candidate_id(f"cand_{group_name}", record)
            public_rows.append(public_candidate(record, cid))
            private_rows.append(
                {
                    "candidate_id": cid,
                    "private_group": group_name,
                    "source_id": record.record_id,
                    "source_prompt_membership": int(member),
                    "exact_reconstruction": int(member),
                    "record_membership_label": int(member),
                }
            )
    generated_dir = model_root(cfg, project_root, smoke=smoke) / "generated"
    public_rows.extend(read_jsonl(generated_dir / "public_generated.jsonl"))
    private_rows.extend(read_jsonl(generated_dir / "private_generated_labels.jsonl"))
    raw_candidate_count = len(public_rows)
    public_rows, private_rows = deduplicate_candidate_rows(public_rows, private_rows)
    write_jsonl(paths["public"], public_rows)
    write_jsonl(paths["private"], private_rows)
    write_shadow_masks(
        paths["masks"],
        [row["candidate_id"] for row in public_rows],
        seed=int(cfg.runtime.seed) + int(cfg.shadow.seed_offset),
        shadow_count=shadow_count(cfg, smoke),
        inclusion_probability=float(cfg.shadow.inclusion_probability),
    )
    manifest = artifact_manifest(
        cfg,
        project_root,
        {
            "public_candidates": paths["public"],
            "private_labels": paths["private"],
            "shadow_masks": paths["masks"],
        },
        row_counts={
            "public_candidates": len(public_rows),
            "private_labels": len(private_rows),
            "shadow_models": shadow_count(cfg, smoke),
        },
    )
    write_json(paths["manifest"], manifest)
    _log_wandb_stage(
        stage="candidates",
        metrics={
            "public_candidates": float(len(public_rows)),
            "deduplicated_candidates": float(raw_candidate_count - len(public_rows)),
        },
        paths={**paths},
    )


def run_attack(
    cfg: DictConfig,
    *,
    project_root: Path,
    command: str,
    smoke: bool,
) -> None:
    train_shadows(cfg, project_root=project_root, command=command, smoke=smoke)
    run_dir = model_root(cfg, project_root, smoke=smoke) / "attack"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not bool(cfg.workflow.force):
        LOGGER.info("Reusing attack outputs: %s", metrics_path)
        return

    paths = candidate_paths(cfg, project_root, smoke=smoke)
    public_rows = read_jsonl(paths["public"])
    candidates = [record_from_public_candidate(row) for row in public_rows]
    masks = read_shadow_masks(paths["masks"])
    target_adapter = (
        model_root(cfg, project_root, smoke=smoke)
        / "target"
        / f"seed_{int(cfg.runtime.seed)}"
        / "adapter"
    )
    target_scores = score_with_adapter(
        cfg,
        candidates,
        adapter_path=target_adapter,
        batch_size=int(cfg.attack.batch_size),
    )
    shadow_scores = []
    for index in range(shadow_count(cfg, smoke)):
        adapter = (
            model_root(cfg, project_root, smoke=smoke)
            / "shadows"
            / f"shadow_{index:02d}"
            / "adapter"
        )
        shadow_scores.append(
            score_with_adapter(
                cfg,
                candidates,
                adapter_path=adapter,
                batch_size=int(cfg.attack.batch_size),
            )
        )
    if len(target_scores) != len(public_rows) or any(
        len(scores) != len(public_rows) for scores in shadow_scores
    ):
        raise ValueError(
            "Candidate score count does not match the public candidate table."
        )

    population = load_split_file(cfg, project_root, "squad_validation_population")
    if smoke:
        population = deterministic_subset(
            population,
            seed=int(cfg.runtime.seed),
            limit=candidate_limit(cfg, smoke),
            namespace="smoke_population",
        )
    population_target = score_with_adapter(
        cfg,
        population,
        adapter_path=target_adapter,
        batch_size=int(cfg.attack.batch_size),
    )
    population_shadows = []
    for index in range(shadow_count(cfg, smoke)):
        adapter = (
            model_root(cfg, project_root, smoke=smoke)
            / "shadows"
            / f"shadow_{index:02d}"
            / "adapter"
        )
        population_shadows.append(
            score_with_adapter(
                cfg,
                population,
                adapter_path=adapter,
                batch_size=int(cfg.attack.batch_size),
            )
        )
    population_relative = [
        population_relative_loglikelihood(
            population_target[row_index],
            [scores[row_index] for scores in population_shadows],
        )
        for row_index in range(len(population))
    ]

    score_rows = []
    for row_index, public_row in enumerate(public_rows):
        candidate_id_value = public_row["candidate_id"]
        per_shadow = [scores[row_index] for scores in shadow_scores]
        relative, in_mean, out_mean = candidate_relative_loglikelihood(
            target_scores[row_index], per_shadow, masks[candidate_id_value]
        )
        score_rows.append(
            {
                "candidate_id": candidate_id_value,
                "target_mean_logprob": target_scores[row_index],
                "shadow_in_mean_logprob": in_mean,
                "shadow_out_mean_logprob": out_mean,
                "relative_log_likelihood": relative,
                "online_rmia_score": online_rmia_score(
                    relative, population_relative, gamma=float(cfg.attack.gamma)
                ),
                "num_in_shadows": int(sum(masks[candidate_id_value])),
                "num_out_shadows": int(
                    len(masks[candidate_id_value]) - sum(masks[candidate_id_value])
                ),
            }
        )
    scores_path = run_dir / "online_rmia_scores.jsonl"
    write_jsonl(scores_path, score_rows)

    population_rmia_scores = [
        online_rmia_score(
            value,
            [
                other
                for index, other in enumerate(population_relative)
                if index != row_index
            ],
            gamma=float(cfg.attack.gamma),
        )
        for row_index, value in enumerate(population_relative)
    ]
    private_rows = read_jsonl(paths["private"])
    metrics = attack_metrics(
        score_rows,
        private_rows,
        calibration_scores=population_rmia_scores,
        fpr_thresholds=list(cfg.attack.fpr_thresholds),
        bootstrap_samples=int(cfg.attack.bootstrap_samples),
        seed=int(cfg.runtime.seed),
    )
    write_json(metrics_path, metrics)
    write_json(
        run_dir / "manifest.json",
        artifact_manifest(
            cfg,
            project_root,
            {"scores": scores_path, "metrics": metrics_path},
            row_counts={
                "candidates": len(score_rows),
                "population": len(population),
                "shadow_models": shadow_count(cfg, smoke),
            },
        ),
    )
    tracking_artifacts = _log_wandb_stage(
        stage="attack",
        metrics={key: value for key, value in metrics.items() if _is_number(value)},
        paths={
            "scores": scores_path,
            "metrics": metrics_path,
            "manifest": run_dir / "manifest.json",
        },
    )
    write_experiment_markdown(
        run_dir / str(cfg.report.experiment_filename),
        purpose=f"Analyze text-only RMIA score distributions for {cfg.model.display_name}.",
        hypothesis=str(cfg.experiment.hypothesis),
        command=command,
        overrides=[],
        config_fingerprint_value=config_fingerprint(
            OmegaConf.to_container(cfg, resolve=True)
        ),
        git_state=collect_git_state(project_root),
        environment=extended_environment(project_root),
        artifacts={
            "scores": str(scores_path),
            "metrics": str(metrics_path),
            "public_candidates": str(paths["public"]),
            "shadow_masks": str(paths["masks"]),
            **tracking_artifacts,
        },
        metrics={key: value for key, value in metrics.items() if _is_number(value)},
        conclusion="Scores used text and shadow masks only; evaluator labels were applied after scoring.",
        achieved_purpose=True,
        next_action="Compare model families and true membership-conditioned distributions.",
        wandb_run_id=_active_wandb_run_id(),
    )


def score_with_adapter(
    cfg: DictConfig,
    candidates: list[QARecord],
    *,
    adapter_path: Path,
    batch_size: int,
) -> list[float]:
    model, tokenizer = load_model_for_inference(cfg, adapter_path=str(adapter_path))
    stats = score_records(
        model,
        tokenizer,
        candidates,
        batch_size=batch_size,
        max_length=int(cfg.tokenizer.max_length),
        loss_chunk_tokens=int(cfg.attack.loss_chunk_tokens),
        tokenization_workers=int(cfg.tokenizer.preprocessing_workers),
    )
    scores = [item.mean_logprob for item in stats]
    del model
    cleanup_cuda()
    return scores


def attack_metrics(
    score_rows: list[dict[str, Any]],
    private_rows: list[dict[str, Any]],
    *,
    calibration_scores: list[float],
    fpr_thresholds: list[float],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float]:
    score_by_id = {
        row["candidate_id"]: float(row["online_rmia_score"]) for row in score_rows
    }
    labels_by_id = {
        row["candidate_id"]: int(row["record_membership_label"]) for row in private_rows
    }
    groups_by_id = {
        row["candidate_id"]: str(row["private_group"]) for row in private_rows
    }
    positives = [
        cid
        for cid, label in labels_by_id.items()
        if label == 1 and groups_by_id[cid] == "gold_squad_target_train"
    ]
    metrics: dict[str, float] = {
        "population_calibration_examples": float(len(calibration_scores))
    }
    for target_fpr in fpr_thresholds:
        threshold = threshold_for_fpr(calibration_scores, target_fpr)
        metrics[f"threshold_at_population_fpr_{target_fpr}"] = threshold
        metrics[f"tpr_gold_train_at_population_fpr_{target_fpr}"] = fraction_above(
            [score_by_id[cid] for cid in positives], threshold
        )
        for group in sorted(set(groups_by_id.values())):
            negatives = [
                cid
                for cid, value in groups_by_id.items()
                if value == group and labels_by_id[cid] == 0
            ]
            metrics[f"fpr_{group}_at_population_fpr_{target_fpr}"] = fraction_above(
                [score_by_id[cid] for cid in negatives], threshold
            )
    for group in sorted(set(groups_by_id.values())):
        negatives = [
            cid
            for cid, value in groups_by_id.items()
            if value == group and labels_by_id[cid] == 0
        ]
        if positives and negatives:
            y_true = [1] * len(positives) + [0] * len(negatives)
            values = [score_by_id[cid] for cid in positives + negatives]
            metrics[f"auc_gold_train_vs_{group}"] = float(roc_auc_score(y_true, values))
            metrics.update(
                {
                    f"{key}_gold_train_vs_{group}": value
                    for key, value in bootstrap_binary_metrics(
                        [score_by_id[cid] for cid in positives],
                        [score_by_id[cid] for cid in negatives],
                        fpr_thresholds=fpr_thresholds,
                        samples=bootstrap_samples,
                        seed=seed + sum(ord(char) for char in group),
                    ).items()
                }
            )
    metrics["num_candidates"] = float(len(score_rows))
    return metrics


def threshold_for_fpr(scores: list[float], target_fpr: float) -> float:
    if not scores:
        return float("inf")
    ordered = sorted(scores, reverse=True)
    index = max(0, min(len(ordered) - 1, math.floor(target_fpr * len(ordered))))
    return float(ordered[index])


def fraction_above(scores: list[float], threshold: float) -> float:
    if not scores:
        return float("nan")
    return float(sum(score >= threshold for score in scores) / len(scores))


def artifact_manifest(
    cfg: DictConfig,
    project_root: Path,
    files: dict[str, Path],
    *,
    row_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "source_commit": collect_git_state(project_root),
        "config_sha256": config_fingerprint(OmegaConf.to_container(cfg, resolve=True)),
        "seed": int(cfg.runtime.seed),
        "model": str(cfg.model.name_or_path),
        "tokenizer": str(cfg.model.name_or_path),
        "row_counts": row_counts,
        "files": {name: str(path) for name, path in files.items()},
        "file_sha256": {
            name: text_sha256(path.read_text(encoding="utf-8"))
            for name, path in files.items()
            if path.exists()
        },
    }


def extended_environment(project_root: Path) -> dict[str, Any]:
    info = collect_environment(project_root)
    for module_name in ["transformers", "datasets", "peft", "accelerate"]:
        try:
            module = __import__(module_name)
            info[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            info[f"{module_name}_error"] = repr(exc)
    return info


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _trainable_parameter_count(model: Any) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
