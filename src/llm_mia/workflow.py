from __future__ import annotations

import gc
import json
import logging
import math
import os
import shutil
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from hydra.utils import instantiate
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
    fixed_variance_lira_parameters,
    online_lira_fixed_variance_score,
    online_rmia_score,
    population_relative_loglikelihood,
    tpr_at_fpr,
)
from src.llm_mia.data import (
    QARecord,
    candidate_id,
    canonicalize_candidate_rows,
    deterministic_subset,
    deterministic_unique_subset,
    file_sha256,
    public_candidate,
    read_json,
    read_jsonl,
    read_shadow_masks,
    record_from_json,
    record_from_public_candidate,
    record_to_json,
    split_records,
    split_squad_train,
    squad_row_to_record,
    trivia_row_to_record,
    write_json,
    write_jsonl,
    write_shadow_masks,
    write_single_shadow_smoke_mask,
)
from src.llm_mia.finetuning import (
    FineTuningStrategy,
    LoraFineTuningStrategy,
)
from src.llm_mia.hf import (
    ensure_token_embeddings,
    generate_completions,
    load_causal_lm,
    load_tokenizer,
    make_trainer,
    materialize_records,
    score_records,
)
from src.llm_mia.metrics import exact_match, mean, perplexity, token_f1
from src.llm_mia.plotting import (
    candidate_groups_for_variant,
    plot_group_feature_ecdf,
    sample_group_candidates,
    sample_group_feature_rows,
    summarize_group_features,
)
from src.llm_mia.reuse import (
    ReuseValidationError,
    inherit_lora_candidate_masks,
    prepare_lora_shadow_expansion_inputs,
    validate_base_generation_artifact,
    validate_lora_candidate_artifact,
    validate_lora_mask_reuse_source,
    validate_split_artifact,
)

LOGGER = logging.getLogger(__name__)
_ACTIVE_WANDB_RUN: ContextVar[Any | None] = ContextVar("active_wandb_run", default=None)


def fine_tuning_strategy(cfg: DictConfig) -> FineTuningStrategy:
    target = OmegaConf.select(cfg, "finetuning._target_")
    strategy = (
        LoraFineTuningStrategy()
        if target is None
        else instantiate(cfg.finetuning, _convert_="partial")
    )
    required = (
        "checkpoint_dirname",
        "prepare_model",
        "validate_trainable",
        "save_final",
        "load_for_inference",
    )
    if any(not hasattr(strategy, name) for name in required):
        raise TypeError(f"Invalid fine-tuning strategy: {type(strategy).__name__}")
    return strategy


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
        elif stage == "prepare_shadow_reuse":
            prepare_shadow_expansion_reuse(
                cfg, project_root=project_root, command=command
            )
        elif stage == "train_target":
            train_target(cfg, project_root=project_root, command=command, smoke=False)
        elif stage == "evaluate":
            evaluate_all(cfg, project_root=project_root, command=command, smoke=False)
        elif stage == "generate":
            generate_target_candidates(
                cfg, project_root=project_root, command=command, smoke=False
            )
        elif stage == "generate_base":
            generate_base_candidates(
                cfg, project_root=project_root, command=command, smoke=False
            )
        elif stage == "build_candidates":
            build_candidate_set(cfg, project_root=project_root, smoke=False)
        elif stage == "train_shadows":
            train_shadows(cfg, project_root=project_root, command=command, smoke=False)
        elif stage == "attack":
            run_attack(cfg, project_root=project_root, command=command, smoke=False)
        elif stage == "validate":
            validate_formal_outputs(cfg, project_root=project_root, command=command)
        elif stage == "plot_rmia_feature":
            plot_rmia_feature_distribution(
                cfg, project_root=project_root, command=command
            )
        elif stage == "plot_lira_feature":
            plot_lira_feature_distribution(
                cfg, project_root=project_root, command=command
            )
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
    prepare_reuse_manifest(cfg, project_root=project_root, smoke=smoke)
    if not smoke:
        verify_formal_storage(cfg, project_root=project_root)
    train_target(cfg, project_root=project_root, command=command, smoke=smoke)
    evaluate_all(cfg, project_root=project_root, command=command, smoke=smoke)
    generate_target_candidates(
        cfg, project_root=project_root, command=command, smoke=smoke
    )
    build_candidate_set(cfg, project_root=project_root, smoke=smoke)
    train_shadows(cfg, project_root=project_root, command=command, smoke=smoke)
    if smoke:
        LOGGER.info("Smoke completed; online RMIA requires the five-shadow formal run.")
    else:
        run_attack(cfg, project_root=project_root, command=command, smoke=False)


def profile_name(cfg: DictConfig, smoke: bool) -> str:
    return "smoke" if smoke else str(cfg.workflow.profile)


def model_root(cfg: DictConfig, project_root: Path, *, smoke: bool) -> Path:
    return (
        project_root
        / str(cfg.paths.output_root)
        / profile_name(cfg, smoke)
        / str(cfg.model.key)
    )


def attack_method(cfg: DictConfig) -> str:
    method = str(OmegaConf.select(cfg, "attack.method", default="online_rmia"))
    if method not in {"online_rmia", "online_lira_fixed_variance"}:
        raise ValueError(f"Unsupported text attack method: {method}")
    return method


def attack_score_filename(cfg: DictConfig) -> str:
    if attack_method(cfg) == "online_lira_fixed_variance":
        return "online_lira_scores.jsonl"
    return "online_rmia_scores.jsonl"


def attack_score_field(cfg: DictConfig) -> str:
    if attack_method(cfg) == "online_lira_fixed_variance":
        return "online_lira_log_ratio"
    return "online_rmia_score"


def attack_expected_row_counts(
    cfg: DictConfig, *, candidate_rows: int, smoke: bool
) -> dict[str, int]:
    row_counts = {
        "candidates": candidate_rows,
        "shadow_models": shadow_count(cfg, smoke),
    }
    if attack_method(cfg) == "online_rmia":
        row_counts["population"] = (
            candidate_limit(cfg, smoke)
            if smoke
            else int(cfg.data.squad_validation_population_size)
        )
    return row_counts


def target_checkpoint_path(cfg: DictConfig, project_root: Path, *, smoke: bool) -> Path:
    strategy = fine_tuning_strategy(cfg)
    return (
        model_root(cfg, project_root, smoke=smoke)
        / "target"
        / f"seed_{int(cfg.runtime.seed)}"
        / strategy.checkpoint_dirname
    )


def shadow_checkpoint_paths(
    cfg: DictConfig, project_root: Path, *, smoke: bool
) -> list[Path]:
    strategy = fine_tuning_strategy(cfg)
    root = model_root(cfg, project_root, smoke=smoke) / "shadows"
    return [
        root / f"shadow_{index:02d}" / strategy.checkpoint_dirname
        for index in range(shadow_count(cfg, smoke))
    ]


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
    reference_paths: dict[str, Path] | None = None,
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
    for name, path in (reference_paths or {}).items():
        if not path.exists():
            raise FileNotFoundError(f"Reference artifact path is missing: {path}")
        artifact.add_reference(path.resolve().as_uri(), name=name)
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


def _evaluation_records(cfg: DictConfig, *, smoke: bool) -> dict[str, list[QARecord]]:
    limit = eval_limit(cfg, smoke)
    return {
        "squad_validation": deterministic_subset(
            load_squad_records(cfg, "validation"),
            seed=int(cfg.runtime.seed),
            limit=limit,
            namespace="eval_squad_validation",
        ),
        "trivia_validation": deterministic_subset(
            load_trivia_records(cfg, "validation", limit=limit),
            seed=int(cfg.runtime.seed),
            limit=limit,
            namespace="eval_trivia_validation",
        ),
    }


def _target_eval_each_epoch(cfg: DictConfig) -> bool:
    return bool(OmegaConf.select(cfg, "train.target_eval_each_epoch", default=False))


def _base_generation_source_ids(
    cfg: DictConfig, project_root: Path, *, smoke: bool
) -> dict[str, list[str]]:
    limit = candidate_limit(cfg, smoke)
    specifications = (
        (
            "gen_from_squad_train",
            "squad_target_train",
            "gen_from_squad_train",
        ),
        (
            "gen_from_squad_validation",
            "squad_validation_candidates",
            "gen_from_squad_validation",
        ),
        (
            "gen_from_trivia_validation",
            "trivia_validation_candidates",
            "gen_from_trivia_validation",
        ),
    )
    return {
        group: [
            record.record_id
            for record in deterministic_subset(
                load_split_file(cfg, project_root, split_name),
                seed=int(cfg.runtime.seed),
                limit=limit,
                namespace=namespace,
            )
        ]
        for group, split_name, namespace in specifications
    }


def prepare_reuse_manifest(
    cfg: DictConfig, *, project_root: Path, smoke: bool
) -> dict[str, Any] | None:
    strategy = fine_tuning_strategy(cfg)
    if smoke or strategy.name != "full" or not bool(cfg.reuse.enabled):
        return None

    split_root = project_root / str(cfg.reuse.split_root)
    split_record = validate_split_artifact(split_root, _data_manifest_expected(cfg))
    split_record["path"] = str(split_root.relative_to(project_root))
    base_root = (
        project_root
        / str(cfg.reuse.base_generation_root)
        / str(cfg.model.key)
        / "generated"
        / "base"
    )
    generation_config = {
        "do_sample": bool(cfg.generation.do_sample),
        "temperature": float(cfg.generation.temperature),
        "max_new_tokens": int(cfg.generation.max_new_tokens),
    }
    try:
        base_record = validate_base_generation_artifact(
            base_root,
            expected_model_run_id=f"{cfg.model.name_or_path}@{cfg.model.revision}",
            expected_tokenizer_id=str(cfg.model.name_or_path),
            expected_source_ids=_base_generation_source_ids(
                cfg, project_root, smoke=False
            ),
            expected_generation_config=generation_config,
        )
        base_record["path"] = str(base_root.relative_to(project_root))
    except ReuseValidationError as error:
        base_record = {
            "status": "regenerate",
            "reason": str(error),
            "path": str(
                (
                    model_root(cfg, project_root, smoke=False) / "generated" / "base"
                ).relative_to(project_root)
            ),
        }
    split_hashes = split_record["file_sha256"]
    base_record["source_split_sha256"] = {
        name: split_hashes[name]
        for name in (
            "squad_target_train.jsonl",
            "squad_validation_candidates.jsonl",
            "trivia_validation_candidates.jsonl",
        )
    }
    manifest = {
        "split_artifact": split_record,
        "base_generation": base_record,
    }
    write_json(
        model_root(cfg, project_root, smoke=False) / "reuse_manifest.json",
        manifest,
    )
    return manifest


def prepare_shadow_expansion_reuse(
    cfg: DictConfig,
    *,
    project_root: Path,
    command: str,
) -> dict[str, Any]:
    strategy = fine_tuning_strategy(cfg)
    if strategy.name != "lora":
        raise ValueError("Shadow-expansion reuse is only defined for LoRA runs.")
    if not bool(cfg.reuse.enabled):
        raise ValueError("Shadow-expansion reuse must be explicitly enabled.")

    source_model_root = (
        project_root / str(cfg.reuse.source_output_root) / str(cfg.model.key)
    )
    destination_model_root = model_root(cfg, project_root, smoke=False)
    record = prepare_lora_shadow_expansion_inputs(
        project_root=project_root,
        source_model_root=source_model_root,
        destination_model_root=destination_model_root,
        expected_config=_resolved_config(cfg),
        expected_source_shadow_count=int(cfg.reuse.expected_source_shadow_count),
        seed=int(cfg.runtime.seed),
        target_eval_name=strategy.target_eval_name,
        force=bool(cfg.workflow.force),
    )
    run_dir = destination_model_root / "reuse"
    manifest_path = run_dir / "manifest.json"
    resolved_config_path = run_dir / str(cfg.report.resolved_config_filename)
    write_json(manifest_path, record)
    write_yaml(resolved_config_path, _resolved_config(cfg))
    tracking_artifacts = _log_wandb_stage(
        stage="reuse",
        metrics={
            "source_shadow_models": float(cfg.reuse.expected_source_shadow_count),
            "destination_shadow_models": float(cfg.shadow.count),
        },
        paths={
            "manifest": manifest_path,
            "resolved_config": resolved_config_path,
        },
        reference_paths={
            "source_target": source_model_root / "target",
            "source_eval": source_model_root / "eval",
            "source_generated": source_model_root / "generated",
        },
    )
    write_experiment_markdown(
        run_dir / str(cfg.report.experiment_filename),
        purpose=(
            "Reuse immutable target, evaluation, and generated artifacts while "
            f"expanding {cfg.model.display_name} from "
            f"{cfg.reuse.expected_source_shadow_count} to {cfg.shadow.count} shadows."
        ),
        hypothesis=str(cfg.experiment.hypothesis),
        command=command,
        overrides=[],
        config_fingerprint_value=config_fingerprint(_resolved_config(cfg)),
        git_state=collect_git_state(project_root),
        environment=extended_environment(project_root),
        artifacts={
            "reuse_manifest": str(manifest_path.relative_to(project_root)),
            "source_model_root": str(source_model_root.relative_to(project_root)),
            **tracking_artifacts,
        },
        metrics={
            "source_shadow_models": float(cfg.reuse.expected_source_shadow_count),
            "destination_shadow_models": float(cfg.shadow.count),
        },
        conclusion=(
            "All reused files passed config, presence, and SHA256 validation; "
            "candidates, masks, shadows, and attack outputs remain new artifacts."
        ),
        achieved_purpose=True,
        next_action="Build the 100-shadow candidate masks and train all shadows.",
        wandb_run_id=_active_wandb_run_id(),
    )
    return record


def base_generation_dir(cfg: DictConfig, project_root: Path, *, smoke: bool) -> Path:
    own_path = model_root(cfg, project_root, smoke=smoke) / "generated" / "base"
    if smoke or fine_tuning_strategy(cfg).name != "full":
        return own_path
    manifest = prepare_reuse_manifest(cfg, project_root=project_root, smoke=False)
    if manifest is None:
        return own_path
    record = manifest["base_generation"]
    if record["status"] == "reused":
        return project_root / str(record["path"])
    return own_path


def verify_formal_storage(cfg: DictConfig, *, project_root: Path) -> None:
    strategy = fine_tuning_strategy(cfg)
    if strategy.name != "full":
        return
    smoke_checkpoint = target_checkpoint_path(cfg, project_root, smoke=True)
    if not strategy.checkpoint_is_complete(smoke_checkpoint):
        raise FileNotFoundError(
            "A verified full-FT target smoke checkpoint is required before formal "
            f"launch: {smoke_checkpoint}"
        )
    model_bytes = sum(
        path.stat().st_size for path in smoke_checkpoint.rglob("*") if path.is_file()
    )
    model_equivalents = 6.0 + float(cfg.checkpoint.active_checkpoint_model_equivalents)
    required_bytes = math.ceil(
        model_bytes * model_equivalents * (1.0 + float(cfg.checkpoint.storage_margin))
    )
    output_root = project_root / str(cfg.paths.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_root).free
    record = {
        "smoke_checkpoint": str(smoke_checkpoint.relative_to(project_root)),
        "smoke_model_bytes": model_bytes,
        "model_equivalents": model_equivalents,
        "storage_margin": float(cfg.checkpoint.storage_margin),
        "required_bytes": required_bytes,
        "free_bytes": free_bytes,
        "passed": free_bytes >= required_bytes,
    }
    write_json(
        model_root(cfg, project_root, smoke=False) / "storage_preflight.json", record
    )
    if not record["passed"]:
        raise OSError(
            f"Insufficient storage for formal full-FT run: required={required_bytes}, "
            f"free={free_bytes}."
        )


def shadow_count(cfg: DictConfig, smoke: bool) -> int:
    return int(cfg.shadow.smoke_count if smoke else cfg.shadow.count)


def max_steps(cfg: DictConfig, smoke: bool) -> int | None:
    value = cfg.train.smoke_max_steps if smoke else cfg.train.max_steps
    return None if value is None else int(value)


def training_seed(cfg: DictConfig, shadow_index: int | None) -> int:
    if shadow_index is None:
        return int(cfg.runtime.seed)
    return int(cfg.runtime.seed) + int(cfg.shadow.seed_offset) + shadow_index


def _resolved_config(cfg: DictConfig) -> dict[str, Any]:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Resolved Hydra config must be a mapping.")
    return resolved


def _semantic_config_fingerprint(
    config: dict[str, Any], *, ignored_sections: tuple[str, ...] = ()
) -> str:
    normalized = deepcopy(config)
    workflow = normalized.get("workflow")
    if isinstance(workflow, dict):
        workflow.pop("stage", None)
        workflow.pop("force", None)
    for section in ignored_sections:
        normalized.pop(section, None)
    return config_fingerprint(normalized)


def _run_config_matches(cfg: DictConfig, run_dir: Path) -> bool:
    resolved_config_path = run_dir / str(cfg.report.resolved_config_filename)
    if not resolved_config_path.is_file():
        return False
    try:
        prior = OmegaConf.to_container(
            OmegaConf.load(resolved_config_path), resolve=True
        )
    except (OSError, ValueError):
        return False
    return _semantic_config_fingerprint(prior) == _semantic_config_fingerprint(
        _resolved_config(cfg)
    )


def _artifact_record_is_complete(
    cfg: DictConfig,
    manifest_path: Path,
    files: dict[str, Path],
    *,
    expected_row_counts: dict[str, int],
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError):
        return False
    if manifest.get("semantic_config_sha256") != _semantic_config_fingerprint(
        _resolved_config(cfg)
    ):
        return False
    row_counts = manifest.get("row_counts", {})
    if any(
        row_counts.get(name) != count for name, count in expected_row_counts.items()
    ):
        return False
    hashes = manifest.get("file_sha256", {})
    return all(
        path.is_file() and hashes.get(name) == file_sha256(path)
        for name, path in files.items()
    )


def _training_run_is_complete(
    cfg: DictConfig,
    run_dir: Path,
    strategy: FineTuningStrategy,
    *,
    extra_required_files: tuple[Path, ...] = (),
    ignored_config_sections: tuple[str, ...] = (),
) -> bool:
    checkpoint_path = run_dir / strategy.checkpoint_dirname
    resolved_config_path = run_dir / str(cfg.report.resolved_config_filename)
    required_files = (
        run_dir / "train_manifest.jsonl",
        run_dir / "metrics.json",
        resolved_config_path,
        run_dir / str(cfg.report.experiment_filename),
        *extra_required_files,
    )
    if strategy.name == "full":
        required_files = (*required_files, run_dir / "run_manifest.json")
    if not all(path.is_file() for path in required_files):
        return False
    try:
        prior_config = OmegaConf.to_container(
            OmegaConf.load(resolved_config_path), resolve=True
        )
    except (OSError, ValueError):
        return False
    if not isinstance(prior_config, dict):
        return False
    current_config = _resolved_config(cfg)
    if _semantic_config_fingerprint(
        prior_config, ignored_sections=ignored_config_sections
    ) != _semantic_config_fingerprint(
        current_config, ignored_sections=ignored_config_sections
    ):
        return False
    config_sha256 = config_fingerprint(prior_config)
    return strategy.checkpoint_is_complete(checkpoint_path, config_sha256=config_sha256)


def _resume_checkpoint(
    cfg: DictConfig,
    *,
    project_root: Path,
    trainer_dir: Path,
) -> str | None:
    if bool(cfg.workflow.force):
        _remove_temporary_checkpoints(trainer_dir)
        return None
    value = OmegaConf.select(cfg, "checkpoint.resume_from_checkpoint")
    if value is None or value is False or str(value).lower() in {"none", "false"}:
        return None
    if str(value).lower() == "auto":
        checkpoints: list[tuple[int, Path]] = []
        for path in trainer_dir.glob("checkpoint-*"):
            try:
                step = int(path.name.rsplit("-", maxsplit=1)[1])
            except (IndexError, ValueError):
                continue
            if path.is_dir():
                checkpoints.append((step, path))
        return None if not checkpoints else str(max(checkpoints)[1])
    checkpoint = Path(str(value))
    if not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
    return str(checkpoint)


def _latest_logged_metric(trainer: Any, name: str) -> float | None:
    for row in reversed(trainer.state.log_history):
        value = row.get(name)
        if _is_number(value):
            return float(value)
    return None


def _epoch_validation_metrics(
    log_history: list[dict[str, Any]],
    eval_records: dict[str, list[QARecord]],
    *,
    expected_epochs: int | None,
) -> dict[str, Any]:
    epoch_rows: dict[tuple[float, int], dict[str, Any]] = {}
    for log_row in log_history:
        epoch = log_row.get("epoch")
        step = log_row.get("step")
        if not _is_number(epoch) or not _is_number(step):
            continue
        key = (float(epoch), int(step))
        for dataset_name in eval_records:
            loss_key = f"eval_{dataset_name}_loss"
            loss = log_row.get(loss_key)
            if not _is_number(loss):
                continue
            row = epoch_rows.setdefault(
                key,
                {"epoch": float(epoch), "global_step": int(step)},
            )
            if dataset_name in row:
                raise ValueError(
                    f"Duplicate {dataset_name} validation loss at epoch {epoch}, "
                    f"step {step}."
                )
            loss_value = float(loss)
            row[dataset_name] = {
                "loss": loss_value,
                "perplexity": perplexity(loss_value),
            }

    epochs = [epoch_rows[key] for key in sorted(epoch_rows)]
    for row in epochs:
        missing = [name for name in eval_records if name not in row]
        if missing:
            raise ValueError(
                f"Epoch {row['epoch']} step {row['global_step']} is missing "
                f"validation losses for: {', '.join(missing)}."
            )
    if not epochs:
        raise ValueError("Target epoch evaluation produced no validation losses.")
    if expected_epochs is not None:
        expected = [float(index) for index in range(1, expected_epochs + 1)]
        actual = [float(row["epoch"]) for row in epochs]
        if len(actual) != len(expected) or any(
            not math.isclose(value, target, abs_tol=1e-6)
            for value, target in zip(actual, expected, strict=True)
        ):
            raise ValueError(
                f"Expected target validation at epochs {expected}, got {actual}."
            )

    return {
        "evaluation_strategy": "epoch",
        "datasets": {
            name: {"records": len(records)} for name, records in eval_records.items()
        },
        "epochs": epochs,
    }


def _remove_temporary_checkpoints(trainer_dir: Path) -> None:
    for path in trainer_dir.glob("checkpoint-*"):
        if path.is_dir():
            shutil.rmtree(path)


def _write_training_run_manifest(
    cfg: DictConfig,
    run_dir: Path,
    checkpoint_path: Path,
    *,
    config_sha256: str,
    role: str,
    model_seed: int,
) -> None:
    files = (
        run_dir / "train_manifest.jsonl",
        run_dir / "metrics.json",
        run_dir / str(cfg.report.resolved_config_filename),
        run_dir / str(cfg.report.experiment_filename),
        run_dir / "trainer" / "trainer_state.json",
        checkpoint_path / "manifest.json",
        checkpoint_path / "_SUCCESS",
    )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Training run manifest inputs are missing: {missing}")
    write_json(
        run_dir / "run_manifest.json",
        {
            "config_sha256": config_sha256,
            "role": role,
            "model_seed": model_seed,
            "file_sha256": {
                path.relative_to(run_dir).as_posix(): file_sha256(path)
                for path in files
            },
        },
    )


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
    strategy = fine_tuning_strategy(cfg)
    checkpoint_path = run_dir / strategy.checkpoint_dirname
    epoch_validation_path = run_dir / "epoch_validation_metrics.json"
    target_eval_each_epoch = _target_eval_each_epoch(cfg)
    required_epoch_files = (epoch_validation_path,) if target_eval_each_epoch else ()
    if _training_run_is_complete(
        cfg,
        run_dir,
        strategy,
        extra_required_files=required_epoch_files,
        ignored_config_sections=("experiment", "mask_reuse"),
    ) and not bool(cfg.workflow.force):
        LOGGER.info("Reusing existing target checkpoint: %s", checkpoint_path)
        return checkpoint_path
    records = load_split_file(cfg, project_root, "squad_target_train")
    if smoke:
        records = deterministic_subset(
            records,
            seed=int(cfg.runtime.seed),
            limit=max(8, candidate_limit(cfg, smoke)),
            namespace="smoke_target_train",
        )
    return train_model(
        cfg,
        project_root=project_root,
        train_records=records,
        run_dir=run_dir,
        role="target",
        shadow_index=None,
        command=command,
        max_steps_value=max_steps(cfg, smoke),
        model_seed=training_seed(cfg, None),
        eval_records=(
            _evaluation_records(cfg, smoke=smoke) if target_eval_each_epoch else None
        ),
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
    masks = read_shadow_masks(
        candidate_paths(cfg, project_root, smoke=smoke)["masks"],
        require_in_out=not smoke,
    )
    tokenizer = load_tokenizer(
        str(cfg.model.name_or_path),
        revision=str(cfg.model.revision),
        trust_remote_code=bool(cfg.tokenizer.trust_remote_code),
    )
    auxiliary_pool = materialize_records(
        load_split_file(cfg, project_root, "squad_attacker_auxiliary_pool"),
        tokenizer,
        int(cfg.tokenizer.max_length),
        num_workers=int(cfg.tokenizer.preprocessing_workers),
    )
    target_manifest_path = (
        model_root(cfg, project_root, smoke=smoke)
        / "target"
        / f"seed_{int(cfg.runtime.seed)}"
        / "train_manifest.jsonl"
    )
    target_train_hashes = {
        record_from_json(row).content_sha256 for row in read_jsonl(target_manifest_path)
    }
    checkpoints: list[Path] = []
    strategy = fine_tuning_strategy(cfg)
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
            excluded_content_hashes=target_train_hashes,
        )
        records = included + fill
        content_hashes = [record.content_sha256 for record in records]
        if len(fill) != target_size - len(included) or len(records) != target_size:
            raise ValueError(f"Shadow {index} could not reach {target_size} rows.")
        if len(set(content_hashes)) != len(content_hashes):
            raise ValueError(f"Shadow {index} contains duplicate record content.")
        if {record.content_sha256 for record in fill} & target_train_hashes:
            raise ValueError(f"Shadow {index} filler overlaps target-training content.")

        run_dir = (
            model_root(cfg, project_root, smoke=smoke)
            / "shadows"
            / f"shadow_{index:02d}"
        )
        checkpoint_path = run_dir / strategy.checkpoint_dirname
        if _training_run_is_complete(cfg, run_dir, strategy) and not bool(
            cfg.workflow.force
        ):
            if not _run_config_matches(cfg, run_dir):
                raise ValueError(
                    f"Existing shadow config does not match the current run: {run_dir}"
                )
            LOGGER.info("Reusing existing shadow checkpoint: %s", checkpoint_path)
            checkpoints.append(checkpoint_path)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(
            run_dir / "train_manifest.jsonl", (record_to_json(row) for row in records)
        )
        write_json(
            run_dir / "inclusion_summary.json",
            {
                "shadow_index": index,
                "included_candidates": len(included),
                "filled_from_auxiliary_pool": len(fill),
                "forbidden_target_overlap": 0,
                "train_size": len(records),
            },
        )
        checkpoints.append(
            train_model(
                cfg,
                project_root=project_root,
                train_records=records,
                run_dir=run_dir,
                role="shadow",
                shadow_index=index,
                command=command,
                max_steps_value=max_steps(cfg, smoke),
                model_seed=training_seed(cfg, index),
            )
        )
    return checkpoints


def train_model(
    cfg: DictConfig,
    *,
    project_root: Path,
    train_records: list[QARecord],
    run_dir: Path,
    role: str,
    shadow_index: int | None,
    command: str,
    max_steps_value: int | None,
    model_seed: int,
    eval_records: dict[str, list[QARecord]] | None = None,
) -> Path:
    if role != "target" and eval_records is not None:
        raise ValueError("Only target training may receive evaluation records.")
    run_dir.mkdir(parents=True, exist_ok=True)
    strategy = fine_tuning_strategy(cfg)
    resolved_config = _resolved_config(cfg)
    config_sha256 = config_fingerprint(resolved_config)
    checkpoint_path = run_dir / strategy.checkpoint_dirname
    if checkpoint_path.exists():
        if not bool(cfg.workflow.force):
            raise ValueError(
                f"Existing training output is incomplete: {checkpoint_path}"
            )
        shutil.rmtree(checkpoint_path)
    write_yaml(
        run_dir / str(cfg.report.resolved_config_filename),
        resolved_config,
    )
    seed_everything(model_seed, deterministic=bool(cfg.runtime.deterministic))
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
    model = strategy.prepare_model(model, cfg)
    trainable_parameters, total_parameters = strategy.validate_trainable(model)
    trainer_dir = run_dir / "trainer"
    trainer = make_trainer(
        model=model,
        tokenizer=tokenizer,
        train_records=train_records,
        cfg=cfg,
        output_dir=str(trainer_dir),
        max_steps=max_steps_value,
        seed=model_seed,
        eval_records=eval_records,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    resume_from_checkpoint = _resume_checkpoint(
        cfg, project_root=project_root, trainer_dir=trainer_dir
    )
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()
    epoch_validation_path: Path | None = None
    if eval_records is not None:
        epoch_validation_path = run_dir / "epoch_validation_metrics.json"
        write_json(
            epoch_validation_path,
            _epoch_validation_metrics(
                trainer.state.log_history,
                eval_records,
                expected_epochs=(
                    int(cfg.train.epochs) if max_steps_value is None else None
                ),
            ),
        )
    strategy.validate_trainable(trainer.model)
    checkpoint_path = strategy.save_final(
        trainer.model,
        tokenizer,
        run_dir,
        config_sha256=config_sha256,
    )
    metrics = {
        f"train/{key}": float(value)
        for key, value in result.metrics.items()
        if _is_number(value)
    }
    metrics["train/records"] = float(len(train_records))
    metrics["train/model_seed"] = float(model_seed)
    metrics["train/trainable_parameters"] = float(trainable_parameters)
    metrics["train/total_parameters"] = float(total_parameters)
    if torch.cuda.is_available():
        metrics["train/peak_gpu_memory_bytes"] = float(
            torch.cuda.max_memory_allocated()
        )
    for metric_name in ("learning_rate", "grad_norm"):
        value = _latest_logged_metric(trainer, metric_name)
        if value is not None:
            metrics[f"train/final_{metric_name}"] = value
    if eval_records is not None:
        for dataset_name in eval_records:
            value = _latest_logged_metric(trainer, f"eval_{dataset_name}_loss")
            if value is not None:
                metrics[f"train/final_eval_{dataset_name}_loss"] = value
                metrics[f"train/final_eval_{dataset_name}_perplexity"] = perplexity(
                    value
                )
    write_json(run_dir / "metrics.json", metrics)
    stage_name = role if shadow_index is None else f"shadow_{shadow_index:02d}"
    checkpoint_artifact = {strategy.artifact_name: checkpoint_path}
    tracking_artifacts = _log_wandb_stage(
        stage=stage_name,
        metrics=metrics,
        paths={
            **({} if strategy.name == "full" else checkpoint_artifact),
            "metrics": run_dir / "metrics.json",
            "resolved_config": run_dir / str(cfg.report.resolved_config_filename),
            **(
                {"epoch_validation_metrics": epoch_validation_path}
                if epoch_validation_path is not None
                else {}
            ),
        },
        reference_paths=(checkpoint_artifact if strategy.name == "full" else None),
    )
    if strategy.name == "full":
        _remove_temporary_checkpoints(trainer_dir)
    write_experiment_markdown(
        run_dir / str(cfg.report.experiment_filename),
        purpose=(
            f"{role} {strategy.report_label} fine-tuning for "
            f"{cfg.model.display_name} on SQuAD QA."
        ),
        hypothesis=str(cfg.experiment.hypothesis),
        command=command,
        overrides=[],
        config_fingerprint_value=config_sha256,
        git_state=collect_git_state(project_root),
        environment=extended_environment(project_root),
        artifacts={
            strategy.artifact_name: str(checkpoint_path),
            "metrics": str(run_dir / "metrics.json"),
            "resolved_config": str(run_dir / str(cfg.report.resolved_config_filename)),
            **(
                {"epoch_validation_metrics": str(epoch_validation_path)}
                if epoch_validation_path is not None
                else {}
            ),
            **tracking_artifacts,
        },
        metrics=metrics,
        conclusion=(
            f"{role} {strategy.report_label} checkpoint saved from the final "
            "epoch/step without checkpoint selection."
        ),
        achieved_purpose=True,
        next_action="Evaluate utility or score candidates with this final checkpoint.",
        wandb_run_id=_active_wandb_run_id(),
    )
    if strategy.name == "full":
        _write_training_run_manifest(
            cfg,
            run_dir,
            checkpoint_path,
            config_sha256=config_sha256,
            role=role,
            model_seed=model_seed,
        )
    del trainer, model
    cleanup_cuda()
    return checkpoint_path


def evaluate_all(
    cfg: DictConfig,
    *,
    project_root: Path,
    command: str,
    smoke: bool,
) -> None:
    strategy = fine_tuning_strategy(cfg)
    target_checkpoint = train_target(
        cfg, project_root=project_root, command=command, smoke=smoke
    )
    evaluate_model(
        cfg,
        project_root=project_root,
        checkpoint_path=None,
        run_name="base",
        command=command,
        smoke=smoke,
    )
    evaluate_model(
        cfg,
        project_root=project_root,
        checkpoint_path=target_checkpoint,
        run_name=strategy.target_eval_name,
        command=command,
        smoke=smoke,
    )


def evaluate_model(
    cfg: DictConfig,
    *,
    project_root: Path,
    checkpoint_path: Path | None,
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
    evaluation_records = _evaluation_records(cfg, smoke=smoke)
    strategy = fine_tuning_strategy(cfg)
    model, tokenizer = strategy.load_for_inference(cfg, checkpoint_path)
    metrics: dict[str, float] = {}
    for name, records in evaluation_records.items():
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
        hypothesis=(
            f"Final {strategy.report_label} fine-tuning should reduce completion "
            "loss and improve QA EM/F1 versus the base model."
        ),
        command=command,
        overrides=[],
        config_fingerprint_value=config_fingerprint(
            OmegaConf.to_container(cfg, resolve=True)
        ),
        git_state=collect_git_state(project_root),
        environment=extended_environment(project_root),
        artifacts={
            "metrics": str(metrics_path),
            "checkpoint": str(checkpoint_path or "base-model"),
            **tracking_artifacts,
        },
        metrics=metrics,
        conclusion="Evaluation completed without checkpoint selection.",
        achieved_purpose=True,
        next_action="Use the final target checkpoint for generation and RMIA scoring.",
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
    strategy = fine_tuning_strategy(cfg)
    target_checkpoint = train_target(
        cfg, project_root=project_root, command=command, smoke=smoke
    )
    _generate_candidates(
        cfg,
        project_root=project_root,
        command=command,
        smoke=smoke,
        checkpoint_path=target_checkpoint,
        group_prefix=strategy.generated_group_prefix,
        run_dir=model_root(cfg, project_root, smoke=smoke) / "generated",
        model_run_id=str(target_checkpoint),
        wandb_stage="generated",
        purpose=(
            "Generate reusable target-model candidate completions for "
            f"{cfg.model.display_name}."
        ),
        hypothesis=(
            "Generated records will mostly be non-members unless prompt and "
            "generated completion exactly reconstruct a target training record."
        ),
        conclusion=(
            "Generated artifacts and private provenance labels were saved for reuse."
        ),
        next_action="Build public attack candidates and shadow inclusion masks.",
    )


def generate_base_candidates(
    cfg: DictConfig,
    *,
    project_root: Path,
    command: str,
    smoke: bool,
) -> None:
    prepare_data(cfg, project_root)
    reuse_manifest = prepare_reuse_manifest(cfg, project_root=project_root, smoke=smoke)
    if (
        reuse_manifest is not None
        and reuse_manifest["base_generation"]["status"] == "reused"
    ):
        LOGGER.info(
            "Reusing validated base generation: %s",
            reuse_manifest["base_generation"]["path"],
        )
        return
    model_id = f"{cfg.model.name_or_path}@{cfg.model.revision}"
    _generate_candidates(
        cfg,
        project_root=project_root,
        command=command,
        smoke=smoke,
        checkpoint_path=None,
        group_prefix="gen",
        run_dir=model_root(cfg, project_root, smoke=smoke) / "generated" / "base",
        model_run_id=model_id,
        wandb_stage="generated_base",
        purpose=(
            f"Generate reusable base-model completions for {cfg.model.display_name} "
            "from the SQuAD train, SQuAD validation, and TriviaQA validation sources."
        ),
        hypothesis=(
            "Base-model generation on the same deterministic source subsets provides "
            "a reproducible pre-LoRA reference for generated-data comparisons."
        ),
        conclusion="Base-model generations and source provenance were saved for reuse.",
        next_action="Compare base and fine-tuned generation metrics and distributions.",
    )


def _generate_candidates(
    cfg: DictConfig,
    *,
    project_root: Path,
    command: str,
    smoke: bool,
    checkpoint_path: Path | None,
    group_prefix: str,
    run_dir: Path,
    model_run_id: str,
    wandb_stage: str,
    purpose: str,
    hypothesis: str,
    conclusion: str,
    next_action: str,
) -> None:
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
    strategy = fine_tuning_strategy(cfg)
    model, tokenizer = strategy.load_for_inference(cfg, checkpoint_path)
    group_names = {
        "squad_train": f"{group_prefix}_from_squad_train",
        "squad_validation": f"{group_prefix}_from_squad_validation",
        "trivia_validation": f"{group_prefix}_from_trivia_validation",
    }
    groups = {
        group_names["squad_train"]: materialize_records(
            target_train,
            tokenizer,
            int(cfg.tokenizer.max_length),
            num_workers=int(cfg.tokenizer.preprocessing_workers),
        ),
        group_names["squad_validation"]: materialize_records(
            squad_validation,
            tokenizer,
            int(cfg.tokenizer.max_length),
            num_workers=int(cfg.tokenizer.preprocessing_workers),
        ),
        group_names["trivia_validation"]: materialize_records(
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
                    "model_run_id": model_run_id,
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
                        group_name == group_names["squad_train"]
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
    resolved_config_path = run_dir / str(cfg.report.resolved_config_filename)
    write_yaml(
        resolved_config_path,
        OmegaConf.to_container(cfg, resolve=True),
    )
    manifest = artifact_manifest(
        cfg,
        project_root,
        {
            "public_generated": public_path,
            "private_generated_labels": private_path,
            "metrics": metrics_path,
            "resolved_config": resolved_config_path,
        },
        row_counts={
            "public_generated": len(public_rows),
            "private_generated_labels": len(private_rows),
        },
    )
    manifest["files"] = {
        name: str(Path(path).relative_to(project_root))
        for name, path in manifest["files"].items()
    }
    manifest["generation_source"] = {
        "model": model_run_id,
        "checkpoint": (None if checkpoint_path is None else str(checkpoint_path)),
    }
    write_json(manifest_path, manifest)
    tracking_artifacts = _log_wandb_stage(
        stage=wandb_stage,
        metrics=metrics,
        paths={
            "public_generated": public_path,
            "private_generated_labels": private_path,
            "metrics": metrics_path,
            "manifest": manifest_path,
            "resolved_config": resolved_config_path,
        },
    )
    write_experiment_markdown(
        run_dir / str(cfg.report.experiment_filename),
        purpose=purpose,
        hypothesis=hypothesis,
        command=command,
        overrides=[],
        config_fingerprint_value=config_fingerprint(
            OmegaConf.to_container(cfg, resolve=True)
        ),
        git_state=collect_git_state(project_root),
        environment=extended_environment(project_root),
        artifacts={
            **{name: str(path) for name, path in manifest["files"].items()},
            "model": model_run_id,
            "checkpoint": str(checkpoint_path or "base-model"),
            **tracking_artifacts,
        },
        metrics=metrics,
        conclusion=conclusion,
        achieved_purpose=True,
        next_action=next_action,
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
        "private": root / "evaluator_mapping.jsonl",
        "raw_public": root / "raw_public_candidates.jsonl",
        "raw_private": root / "raw_private_provenance.jsonl",
        "masks": root / "shadow_masks.csv",
        "manifest": root / "manifest.json",
    }


def _build_reused_shadow_expansion_candidates(
    cfg: DictConfig, *, project_root: Path
) -> None:
    paths = candidate_paths(cfg, project_root, smoke=False)
    source_root = (
        project_root
        / str(cfg.reuse.source_output_root)
        / str(cfg.model.key)
        / "candidates"
    )
    source_record = validate_lora_candidate_artifact(
        source_root,
        expected_model=str(cfg.model.name_or_path),
        expected_tokenizer=str(cfg.model.name_or_path),
        expected_seed=int(cfg.runtime.seed),
        expected_shadow_count=int(cfg.reuse.expected_source_shadow_count),
    )
    artifact_files = {
        "public_candidates": paths["public"],
        "evaluator_mapping": paths["private"],
        "shadow_masks": paths["masks"],
    }
    mask_mode = str(OmegaConf.select(cfg, "reuse.mask_mode", default="new"))
    if mask_mode not in {"new", "exact_source"}:
        raise ValueError(f"Unsupported candidate mask reuse mode: {mask_mode}")
    if mask_mode == "exact_source" and int(
        cfg.reuse.expected_source_shadow_count
    ) != shadow_count(cfg, smoke=False):
        raise ValueError(
            "exact_source requires equal source and destination shadow counts."
        )
    source_files = {
        "public_candidates": source_root / "public_candidates.jsonl",
        "evaluator_mapping": source_root / "private_labels.jsonl",
    }
    if mask_mode == "exact_source":
        source_files["shadow_masks"] = source_root / "shadow_masks.csv"

    if paths["manifest"].exists() and not bool(cfg.workflow.force):
        public_rows = read_jsonl(paths["public"]) if paths["public"].is_file() else []
        private_rows = (
            read_jsonl(paths["private"]) if paths["private"].is_file() else []
        )
        existing_manifest = read_json(paths["manifest"])
        if (
            _artifact_record_is_complete(
                cfg,
                paths["manifest"],
                artifact_files,
                expected_row_counts={
                    "public_candidates": len(public_rows),
                    "evaluator_mapping": len(private_rows),
                    "shadow_models": shadow_count(cfg, smoke=False),
                },
            )
            and existing_manifest.get("source_candidate_artifact") == source_record
            and all(
                file_sha256(source) == file_sha256(artifact_files[name])
                for name, source in source_files.items()
            )
        ):
            LOGGER.info("Reusing expanded candidate masks: %s", paths["manifest"])
            return
        raise ValueError(
            "Existing expanded candidate artifacts are incomplete or invalid. "
            "Use workflow.force=true to rebuild them."
        )

    candidate_root = paths["manifest"].parent
    existing_paths = [
        *artifact_files.values(),
        paths["raw_public"],
        paths["raw_private"],
    ]
    if any(path.exists() for path in existing_paths) and not bool(cfg.workflow.force):
        raise ValueError(
            "Partial expanded candidate artifacts exist without a valid manifest. "
            "Use workflow.force=true to rebuild them."
        )
    candidate_root.mkdir(parents=True, exist_ok=True)
    for path in existing_paths:
        if path.exists():
            path.unlink()
    shutil.copy2(source_files["public_candidates"], paths["public"])
    shutil.copy2(source_files["evaluator_mapping"], paths["private"])
    if mask_mode == "exact_source":
        shutil.copy2(source_files["shadow_masks"], paths["masks"])
    for name, source in source_files.items():
        if file_sha256(source) != file_sha256(artifact_files[name]):
            raise ValueError(
                f"Copied candidate artifact failed hash validation: {name}"
            )

    public_rows = read_jsonl(paths["public"])
    private_rows = read_jsonl(paths["private"])
    candidate_ids = [str(row["candidate_id"]) for row in public_rows]
    if mask_mode != "exact_source":
        write_shadow_masks(
            paths["masks"],
            candidate_ids,
            seed=int(cfg.runtime.seed) + int(cfg.shadow.seed_offset),
            shadow_count=shadow_count(cfg, smoke=False),
            inclusion_probability=float(cfg.shadow.inclusion_probability),
        )
    manifest = artifact_manifest(
        cfg,
        project_root,
        artifact_files,
        row_counts={
            "public_candidates": len(public_rows),
            "evaluator_mapping": len(private_rows),
            "shadow_models": shadow_count(cfg, smoke=False),
        },
    )
    manifest["candidate_source_mode"] = (
        "exact_source_tables_exact_masks"
        if mask_mode == "exact_source"
        else "exact_source_tables_new_masks"
    )
    manifest["source_candidate_artifact"] = source_record
    write_json(paths["manifest"], manifest)
    _log_wandb_stage(
        stage="candidates",
        metrics={
            "public_candidates": float(len(public_rows)),
            "source_shadow_models": float(cfg.reuse.expected_source_shadow_count),
            "destination_shadow_models": float(cfg.shadow.count),
        },
        paths={
            "public_candidates": paths["public"],
            "evaluator_mapping": paths["private"],
            "shadow_masks": paths["masks"],
            "manifest": paths["manifest"],
        },
    )


def build_candidate_set(cfg: DictConfig, *, project_root: Path, smoke: bool) -> None:
    if not smoke and bool(OmegaConf.select(cfg, "reuse.enabled", default=False)):
        _build_reused_shadow_expansion_candidates(cfg, project_root=project_root)
        return
    mask_reuse_enabled = not smoke and bool(
        OmegaConf.select(cfg, "mask_reuse.enabled", default=False)
    )
    mask_reuse_source_root = (
        project_root / str(cfg.mask_reuse.source_output_root) / str(cfg.model.key)
        if mask_reuse_enabled
        else None
    )
    paths = candidate_paths(cfg, project_root, smoke=smoke)
    if paths["manifest"].exists() and not bool(cfg.workflow.force):
        required = ("public", "private", "raw_public", "raw_private", "masks")
        artifact_files = {
            "raw_public_candidates": paths["raw_public"],
            "raw_private_provenance": paths["raw_private"],
            "public_candidates": paths["public"],
            "evaluator_mapping": paths["private"],
            "shadow_masks": paths["masks"],
        }
        if all(paths[name].is_file() for name in required):
            existing_public = read_jsonl(paths["public"])
            existing_mapping = read_jsonl(paths["private"])
            public_ids = {str(row["candidate_id"]) for row in existing_public}
            mapping_ids = {str(row["candidate_id"]) for row in existing_mapping}
        else:
            public_ids = set()
            mapping_ids = set()
        existing_manifest = read_json(paths["manifest"])
        mask_reuse_source_matches = True
        if mask_reuse_enabled:
            assert mask_reuse_source_root is not None
            current_source = validate_lora_mask_reuse_source(
                project_root=project_root,
                source_model_root=mask_reuse_source_root,
                expected_model=str(cfg.model.name_or_path),
                expected_tokenizer=str(cfg.model.name_or_path),
                expected_seed=int(cfg.runtime.seed),
                expected_shadow_count=int(cfg.mask_reuse.expected_source_shadow_count),
                expected_group_size=int(cfg.data.candidate_limit),
            )
            mask_reuse_source_matches = (
                existing_manifest.get("mask_reuse", {}).get("source") == current_source
            )
        if (
            public_ids
            and mask_reuse_source_matches
            and public_ids == mapping_ids
            and len(public_ids) == len(existing_public)
            and _artifact_record_is_complete(
                cfg,
                paths["manifest"],
                artifact_files,
                expected_row_counts={
                    "public_candidates": len(existing_public),
                    "evaluator_mapping": len(existing_mapping),
                    "shadow_models": shadow_count(cfg, smoke),
                },
            )
        ):
            LOGGER.info("Reusing candidate set: %s", paths["manifest"])
            return
        raise ValueError(
            "Existing candidate manifest is incomplete or invalid. Use "
            "workflow.force=true to rebuild it."
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
    target_manifest_path = (
        model_root(cfg, project_root, smoke=smoke)
        / "target"
        / f"seed_{int(cfg.runtime.seed)}"
        / "train_manifest.jsonl"
    )
    target_train_hashes = {
        record_from_json(row).content_sha256 for row in read_jsonl(target_manifest_path)
    }
    raw_public_rows: list[dict[str, Any]] = []
    raw_private_rows: list[dict[str, Any]] = []
    for group_name, rows in [
        ("gold_squad_target_train", target_train),
        ("gold_squad_validation", squad_validation),
        ("gold_trivia_validation", trivia_validation),
    ]:
        for record in rows:
            cid = candidate_id(f"cand_{group_name}", record)
            member = int(record.content_sha256 in target_train_hashes)
            raw_public_rows.append(public_candidate(record, cid))
            raw_private_rows.append(
                {
                    "candidate_id": cid,
                    "private_group": group_name,
                    "source_id": record.record_id,
                    "source_prompt_membership": member,
                    "exact_reconstruction": member,
                    "record_membership_label": member,
                }
            )
    generated_dir = model_root(cfg, project_root, smoke=smoke) / "generated"
    raw_public_rows.extend(read_jsonl(generated_dir / "public_generated.jsonl"))
    raw_private_rows.extend(
        read_jsonl(generated_dir / "private_generated_labels.jsonl")
    )
    write_jsonl(paths["raw_public"], raw_public_rows)
    write_jsonl(paths["raw_private"], raw_private_rows)
    public_rows, private_rows = canonicalize_candidate_rows(
        raw_public_rows, raw_private_rows
    )
    write_jsonl(paths["public"], public_rows)
    write_jsonl(paths["private"], private_rows)
    candidate_ids = [row["candidate_id"] for row in public_rows]
    mask_seed = int(cfg.runtime.seed) + int(cfg.shadow.seed_offset)
    mask_reuse_record: dict[str, Any] | None = None
    if mask_reuse_enabled:
        assert mask_reuse_source_root is not None
        mask_reuse_record = inherit_lora_candidate_masks(
            project_root=project_root,
            source_model_root=mask_reuse_source_root,
            destination_candidate_root=paths["manifest"].parent,
            expected_model=str(cfg.model.name_or_path),
            expected_tokenizer=str(cfg.model.name_or_path),
            expected_seed=int(cfg.runtime.seed),
            expected_shadow_count=int(cfg.mask_reuse.expected_source_shadow_count),
            expected_group_size=int(cfg.data.candidate_limit),
        )
    elif smoke:
        write_single_shadow_smoke_mask(
            paths["masks"],
            candidate_ids,
            seed=mask_seed,
            max_included=candidate_limit(cfg, smoke),
        )
    else:
        write_shadow_masks(
            paths["masks"],
            candidate_ids,
            seed=mask_seed,
            shadow_count=shadow_count(cfg, smoke=False),
            inclusion_probability=float(cfg.shadow.inclusion_probability),
        )
    manifest = artifact_manifest(
        cfg,
        project_root,
        {
            "raw_public_candidates": paths["raw_public"],
            "raw_private_provenance": paths["raw_private"],
            "public_candidates": paths["public"],
            "evaluator_mapping": paths["private"],
            "shadow_masks": paths["masks"],
        },
        row_counts={
            "raw_public_candidates": len(raw_public_rows),
            "raw_private_provenance": len(raw_private_rows),
            "public_candidates": len(public_rows),
            "evaluator_mapping": len(private_rows),
            "shadow_models": shadow_count(cfg, smoke),
        },
    )
    if mask_reuse_record is not None:
        manifest["candidate_source_mode"] = "baseline_mask_inheritance"
        manifest["mask_reuse"] = mask_reuse_record
    write_json(paths["manifest"], manifest)
    _log_wandb_stage(
        stage="candidates",
        metrics={
            "raw_candidates": float(len(raw_public_rows)),
            "public_candidates": float(len(public_rows)),
            "deduplicated_candidates": float(len(raw_public_rows) - len(public_rows)),
        },
        paths={**paths},
    )


def attack_scoring_batch_size(cfg: DictConfig) -> int:
    configured = os.environ.get("LLM_MIA_ATTACK_BATCH_SIZE", cfg.attack.batch_size)
    batch_size = int(configured)
    if batch_size <= 0:
        raise ValueError("Attack scoring batch size must be positive.")
    return batch_size


def run_attack(
    cfg: DictConfig,
    *,
    project_root: Path,
    command: str,
    smoke: bool,
) -> None:
    train_shadows(cfg, project_root=project_root, command=command, smoke=smoke)
    method = attack_method(cfg)
    scoring_batch_size = attack_scoring_batch_size(cfg)
    run_dir = model_root(cfg, project_root, smoke=smoke) / "attack"
    metrics_path = run_dir / "metrics.json"
    paths = candidate_paths(cfg, project_root, smoke=smoke)
    scores_path = run_dir / attack_score_filename(cfg)
    manifest_path = run_dir / "manifest.json"
    experiment_path = run_dir / str(cfg.report.experiment_filename)
    if any(
        path.exists() for path in (scores_path, metrics_path, manifest_path)
    ) and not bool(cfg.workflow.force):
        candidate_rows = read_jsonl(paths["public"])
        if experiment_path.is_file() and _artifact_record_is_complete(
            cfg,
            manifest_path,
            {"scores": scores_path, "metrics": metrics_path},
            expected_row_counts=attack_expected_row_counts(
                cfg, candidate_rows=len(candidate_rows), smoke=smoke
            ),
        ):
            prior_manifest = read_json(manifest_path)
            if prior_manifest.get("attack_method", "online_rmia") == method:
                LOGGER.info("Reusing attack outputs: %s", metrics_path)
                return
        raise ValueError(
            "Existing attack outputs do not match the current configuration. "
            "Use workflow.force=true only to intentionally rebuild them."
        )

    public_rows = read_jsonl(paths["public"])
    candidates = [record_from_public_candidate(row) for row in public_rows]
    masks = read_shadow_masks(paths["masks"])
    target_checkpoint = target_checkpoint_path(cfg, project_root, smoke=smoke)
    target_scores = score_with_checkpoint(
        cfg,
        candidates,
        checkpoint_path=target_checkpoint,
        batch_size=scoring_batch_size,
    )
    shadow_checkpoints = shadow_checkpoint_paths(cfg, project_root, smoke=smoke)
    shadow_scores = [
        score_with_checkpoint(
            cfg,
            candidates,
            checkpoint_path=checkpoint,
            batch_size=scoring_batch_size,
        )
        for checkpoint in shadow_checkpoints
    ]
    if len(target_scores) != len(public_rows) or any(
        len(scores) != len(public_rows) for scores in shadow_scores
    ):
        raise ValueError(
            "Candidate score count does not match the public candidate table."
        )

    per_candidate_shadow_scores = [
        [scores[row_index] for scores in shadow_scores]
        for row_index in range(len(public_rows))
    ]
    candidate_masks = [masks[str(row["candidate_id"])] for row in public_rows]
    calibration_scores: list[float] | None = None
    manifest_details: dict[str, Any] = {
        "attack_method": method,
        "scoring_batch_size": scoring_batch_size,
        "source_mask_sha256": file_sha256(paths["masks"]),
    }
    if method == "online_lira_fixed_variance":
        (
            in_means,
            out_means,
            in_variance,
            out_variance,
            in_degrees,
            out_degrees,
        ) = fixed_variance_lira_parameters(per_candidate_shadow_scores, candidate_masks)
        score_rows = []
        for row_index, public_row in enumerate(public_rows):
            mask = candidate_masks[row_index]
            score_rows.append(
                {
                    "candidate_id": public_row["candidate_id"],
                    "target_mean_logprob": target_scores[row_index],
                    "shadow_in_mean_logprob": in_means[row_index],
                    "shadow_out_mean_logprob": out_means[row_index],
                    "pooled_in_variance": in_variance,
                    "pooled_out_variance": out_variance,
                    "online_lira_log_ratio": online_lira_fixed_variance_score(
                        target_scores[row_index],
                        in_means[row_index],
                        out_means[row_index],
                        in_variance,
                        out_variance,
                    ),
                    "num_in_shadows": int(sum(mask)),
                    "num_out_shadows": int(len(mask) - sum(mask)),
                }
            )
        manifest_details.update(
            {
                "variance_estimator": "pooled_within_candidate_unbiased",
                "pooled_in_variance": in_variance,
                "pooled_out_variance": out_variance,
                "in_variance_degrees_of_freedom": in_degrees,
                "out_variance_degrees_of_freedom": out_degrees,
            }
        )
    else:
        population = load_split_file(cfg, project_root, "squad_validation_population")
        if smoke:
            population = deterministic_subset(
                population,
                seed=int(cfg.runtime.seed),
                limit=candidate_limit(cfg, smoke),
                namespace="smoke_population",
            )
        population_target = score_with_checkpoint(
            cfg,
            population,
            checkpoint_path=target_checkpoint,
            batch_size=scoring_batch_size,
        )
        population_shadows = [
            score_with_checkpoint(
                cfg,
                population,
                checkpoint_path=checkpoint,
                batch_size=scoring_batch_size,
            )
            for checkpoint in shadow_checkpoints
        ]
        population_relative = [
            population_relative_loglikelihood(
                population_target[row_index],
                [scores[row_index] for scores in population_shadows],
            )
            for row_index in range(len(population))
        ]
        score_rows = []
        for row_index, public_row in enumerate(public_rows):
            mask = candidate_masks[row_index]
            relative, in_mean, out_mean = candidate_relative_loglikelihood(
                target_scores[row_index],
                per_candidate_shadow_scores[row_index],
                mask,
            )
            score_rows.append(
                {
                    "candidate_id": public_row["candidate_id"],
                    "target_mean_logprob": target_scores[row_index],
                    "shadow_in_mean_logprob": in_mean,
                    "shadow_out_mean_logprob": out_mean,
                    "relative_log_likelihood": relative,
                    "online_rmia_score": online_rmia_score(
                        relative, population_relative, gamma=float(cfg.attack.gamma)
                    ),
                    "num_in_shadows": int(sum(mask)),
                    "num_out_shadows": int(len(mask) - sum(mask)),
                }
            )
        calibration_scores = [
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

    write_jsonl(scores_path, score_rows)
    private_rows = read_jsonl(paths["private"])
    metrics = attack_metrics(
        score_rows,
        private_rows,
        score_field=attack_score_field(cfg),
        calibration_scores=calibration_scores,
        fpr_thresholds=list(cfg.attack.fpr_thresholds),
        bootstrap_samples=int(cfg.attack.bootstrap_samples),
        seed=int(cfg.runtime.seed),
        include_point_tpr=method == "online_lira_fixed_variance",
    )
    write_json(metrics_path, metrics)
    manifest = artifact_manifest(
        cfg,
        project_root,
        {"scores": scores_path, "metrics": metrics_path},
        row_counts=attack_expected_row_counts(
            cfg, candidate_rows=len(score_rows), smoke=smoke
        ),
    )
    manifest.update(manifest_details)
    write_json(manifest_path, manifest)
    tracking_artifacts = _log_wandb_stage(
        stage="attack",
        metrics={key: value for key, value in metrics.items() if _is_number(value)},
        paths={
            "scores": scores_path,
            "metrics": metrics_path,
            "manifest": manifest_path,
        },
    )
    attack_label = (
        "fixed-variance online LiRA"
        if method == "online_lira_fixed_variance"
        else "online RMIA"
    )
    write_experiment_markdown(
        experiment_path,
        purpose=f"Analyze text-only {attack_label} scores for {cfg.model.display_name}.",
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
            "evaluator_mapping": str(paths["private"]),
            "shadow_masks": str(paths["masks"]),
            **tracking_artifacts,
        },
        metrics={key: value for key, value in metrics.items() if _is_number(value)},
        conclusion="Scores used text and shadow masks only; evaluator labels were applied after scoring.",
        achieved_purpose=True,
        next_action="Compare model families and true membership-conditioned distributions.",
        wandb_run_id=_active_wandb_run_id(),
    )


def _validate_mask_reuse_candidate_lineage(
    cfg: DictConfig,
    *,
    project_root: Path,
    candidate_root: Path,
    public_rows: list[dict[str, Any]],
    masks: dict[str, list[int]],
    expected_shadow_count: int,
) -> dict[str, Any]:
    raw_public = read_jsonl(candidate_root / "raw_public_candidates.jsonl")
    raw_private = read_jsonl(candidate_root / "raw_private_provenance.jsonl")
    group_size = int(cfg.data.candidate_limit)
    expected_gold_rows = 3 * group_size
    expected_generated_rows = 3 * group_size
    expected_raw_rows = expected_gold_rows + expected_generated_rows
    if len(raw_public) != expected_raw_rows or len(raw_private) != expected_raw_rows:
        raise ValueError("Mask-reuse raw candidate row count mismatch.")

    source_model_root = (
        project_root / str(cfg.mask_reuse.source_output_root) / str(cfg.model.key)
    )
    source_record = validate_lora_mask_reuse_source(
        project_root=project_root,
        source_model_root=source_model_root,
        expected_model=str(cfg.model.name_or_path),
        expected_tokenizer=str(cfg.model.name_or_path),
        expected_seed=int(cfg.runtime.seed),
        expected_shadow_count=int(cfg.mask_reuse.expected_source_shadow_count),
        expected_group_size=group_size,
    )
    included_counts = [
        sum(mask[index] for mask in masks.values())
        for index in range(expected_shadow_count)
    ]
    expected_record = {
        "mode": "baseline_mask_inheritance",
        "source": source_record,
        "mapping": {
            "gold": "source_candidate_id_with_content_and_identity_check",
            "generated": "raw_position_with_group_and_source_id_check",
            "canonicalization": "content_sha256_with_conflict_rejection",
        },
        "row_counts": {
            "raw_candidates": expected_raw_rows,
            "gold_candidates": expected_gold_rows,
            "generated_candidates": expected_generated_rows,
            "canonical_candidates": len(public_rows),
            "mask_conflicts": 0,
        },
        "included_candidates_by_shadow": included_counts,
        "shadow_masks_sha256": file_sha256(candidate_root / "shadow_masks.csv"),
    }
    candidate_manifest = read_json(candidate_root / "manifest.json")
    if (
        candidate_manifest.get("candidate_source_mode") != "baseline_mask_inheritance"
        or candidate_manifest.get("mask_reuse") != expected_record
    ):
        raise ValueError("Candidate manifest is not bound to the baseline masks.")
    return expected_record


def validate_formal_outputs(
    cfg: DictConfig,
    *,
    project_root: Path,
    command: str,
) -> None:
    strategy = fine_tuning_strategy(cfg)
    method = attack_method(cfg)
    reuse_enabled = bool(OmegaConf.select(cfg, "reuse.enabled", default=False))
    mask_reuse_enabled = bool(
        OmegaConf.select(cfg, "mask_reuse.enabled", default=False)
    )
    if strategy.name != "lora" or reuse_enabled == mask_reuse_enabled:
        raise ValueError(
            "This validation stage requires exactly one LoRA candidate reuse mode."
        )

    root = model_root(cfg, project_root, smoke=False)
    expected_shadow_count = shadow_count(cfg, smoke=False)
    paths = candidate_paths(cfg, project_root, smoke=False)
    public_rows = read_jsonl(paths["public"])
    private_rows = read_jsonl(paths["private"])
    candidate_files = {
        "public_candidates": paths["public"],
        "evaluator_mapping": paths["private"],
        "shadow_masks": paths["masks"],
    }
    expected_candidate_rows = {
        "public_candidates": len(public_rows),
        "evaluator_mapping": len(private_rows),
        "shadow_models": expected_shadow_count,
    }
    if mask_reuse_enabled:
        candidate_files.update(
            {
                "raw_public_candidates": paths["raw_public"],
                "raw_private_provenance": paths["raw_private"],
            }
        )
        expected_candidate_rows.update(
            {
                "raw_public_candidates": len(read_jsonl(paths["raw_public"])),
                "raw_private_provenance": len(read_jsonl(paths["raw_private"])),
            }
        )
    if not _artifact_record_is_complete(
        cfg,
        paths["manifest"],
        candidate_files,
        expected_row_counts=expected_candidate_rows,
    ):
        raise ValueError("Candidate artifacts failed final manifest validation.")

    masks = read_shadow_masks(paths["masks"])
    if set(masks) != {str(row["candidate_id"]) for row in public_rows} or any(
        len(mask) != expected_shadow_count for mask in masks.values()
    ):
        raise ValueError(
            "Candidate masks do not match the configured shadow count: "
            f"{expected_shadow_count}."
        )

    lineage_experiment_artifacts: dict[str, str]
    completion_lineage: dict[str, Any]
    if reuse_enabled:
        source_candidates = (
            project_root
            / str(cfg.reuse.source_output_root)
            / str(cfg.model.key)
            / "candidates"
        )
        source_record = validate_lora_candidate_artifact(
            source_candidates,
            expected_model=str(cfg.model.name_or_path),
            expected_tokenizer=str(cfg.model.name_or_path),
            expected_seed=int(cfg.runtime.seed),
            expected_shadow_count=int(cfg.reuse.expected_source_shadow_count),
        )
        mask_mode = str(OmegaConf.select(cfg, "reuse.mask_mode", default="new"))
        expected_source_mode = (
            "exact_source_tables_exact_masks"
            if mask_mode == "exact_source"
            else "exact_source_tables_new_masks"
        )
        candidate_manifest = read_json(paths["manifest"])
        if (
            candidate_manifest.get("candidate_source_mode") != expected_source_mode
            or candidate_manifest.get("source_candidate_artifact") != source_record
        ):
            raise ValueError("Candidate manifest is not bound to the source tables.")
        source_candidate_hashes = {
            "public_candidates.jsonl": file_sha256(
                source_candidates / "public_candidates.jsonl"
            ),
            "private_labels.jsonl": file_sha256(
                source_candidates / "private_labels.jsonl"
            ),
        }
        destination_candidate_hashes = {
            "public_candidates.jsonl": file_sha256(paths["public"]),
            "private_labels.jsonl": file_sha256(paths["private"]),
        }
        if mask_mode == "exact_source":
            source_candidate_hashes["shadow_masks.csv"] = file_sha256(
                source_candidates / "shadow_masks.csv"
            )
            destination_candidate_hashes["shadow_masks.csv"] = file_sha256(
                paths["masks"]
            )
        if source_candidate_hashes != destination_candidate_hashes:
            raise ValueError(
                "Candidate text, evaluator labels, or exact source masks changed."
            )
        reuse_manifest = root / "reuse" / "manifest.json"
        lineage_experiment_artifacts = {
            "reuse_manifest": str(reuse_manifest.relative_to(project_root))
        }
        completion_lineage = {
            "source_candidate_sha256": source_candidate_hashes,
            "reuse_manifest_sha256": file_sha256(reuse_manifest),
        }
    else:
        mask_reuse_record = _validate_mask_reuse_candidate_lineage(
            cfg,
            project_root=project_root,
            candidate_root=paths["manifest"].parent,
            public_rows=public_rows,
            masks=masks,
            expected_shadow_count=expected_shadow_count,
        )
        source_model_root = (
            project_root / str(cfg.mask_reuse.source_output_root) / str(cfg.model.key)
        )
        lineage_experiment_artifacts = {
            "source_candidate_manifest": str(
                (source_model_root / "candidates" / "manifest.json").relative_to(
                    project_root
                )
            ),
            "source_generation_manifest": str(
                (source_model_root / "generated" / "manifest.json").relative_to(
                    project_root
                )
            ),
        }
        completion_lineage = {
            "candidate_source_mode": "baseline_mask_inheritance",
            "mask_reuse": mask_reuse_record,
        }

    shadow_records = {}
    expected_shadow_names = {
        f"shadow_{index:02d}" for index in range(expected_shadow_count)
    }
    shadow_root = root / "shadows"
    actual_shadow_names = {
        path.name for path in shadow_root.glob("shadow_*") if path.is_dir()
    }
    if actual_shadow_names != expected_shadow_names:
        raise ValueError("Saved shadow directories do not match the configured count.")
    for index in range(expected_shadow_count):
        run_dir = shadow_root / f"shadow_{index:02d}"
        checkpoint = run_dir / strategy.checkpoint_dirname
        if not _training_run_is_complete(cfg, run_dir, strategy):
            raise ValueError(f"Shadow training record is incomplete: {run_dir}")
        if not _run_config_matches(cfg, run_dir):
            raise ValueError(f"Shadow config mismatch: {run_dir}")
        summary = read_json(run_dir / "inclusion_summary.json")
        included = sum(mask[index] for mask in masks.values())
        expected_train_size = int(cfg.shadow.train_size)
        if summary != {
            "shadow_index": index,
            "included_candidates": included,
            "filled_from_auxiliary_pool": expected_train_size - included,
            "forbidden_target_overlap": 0,
            "train_size": expected_train_size,
        }:
            raise ValueError(f"Shadow inclusion summary mismatch: {run_dir}")
        record_files = [
            run_dir / "train_manifest.jsonl",
            run_dir / "inclusion_summary.json",
            run_dir / "metrics.json",
            run_dir / str(cfg.report.resolved_config_filename),
            run_dir / str(cfg.report.experiment_filename),
            *sorted(path for path in checkpoint.rglob("*") if path.is_file()),
        ]
        if not all(path.is_file() for path in record_files):
            raise FileNotFoundError(f"Shadow record files are missing: {run_dir}")
        shadow_records[f"shadow_{index:02d}"] = {
            path.relative_to(run_dir).as_posix(): file_sha256(path)
            for path in record_files
        }

    attack_dir = root / "attack"
    attack_files = {
        "scores": attack_dir / attack_score_filename(cfg),
        "metrics": attack_dir / "metrics.json",
    }
    attack_manifest_path = attack_dir / "manifest.json"
    if (
        not _artifact_record_is_complete(
            cfg,
            attack_manifest_path,
            attack_files,
            expected_row_counts=attack_expected_row_counts(
                cfg, candidate_rows=len(public_rows), smoke=False
            ),
        )
        or not (attack_dir / str(cfg.report.experiment_filename)).is_file()
    ):
        raise ValueError("Attack artifacts failed final manifest validation.")
    attack_manifest = read_json(attack_manifest_path)
    if attack_manifest.get("attack_method", "online_rmia") != method:
        raise ValueError("Attack manifest method does not match the configuration.")
    if method == "online_lira_fixed_variance":
        if attack_manifest.get(
            "variance_estimator"
        ) != "pooled_within_candidate_unbiased" or attack_manifest.get(
            "source_mask_sha256"
        ) != file_sha256(paths["masks"]):
            raise ValueError("LiRA variance or source-mask lineage is invalid.")
        scores = read_jsonl(attack_files["scores"])
        if len(scores) != len(public_rows) or any(
            not math.isfinite(float(row["online_lira_log_ratio"])) for row in scores
        ):
            raise ValueError("LiRA scores are missing or non-finite.")

    analysis_artifact_sha256: dict[str, str] = {}
    if method == "online_lira_fixed_variance":
        analysis_dir = root / "analysis" / "online_lira_log_ratio"
        analysis_files = {
            "figure": analysis_dir / "online_lira_log_ratio_ecdf.png",
            "sampled_candidates": analysis_dir / "sampled_candidates.jsonl",
            "summary": analysis_dir / "summary.json",
        }
        group_order = tuple(
            dict.fromkeys(str(row["private_group"]) for row in private_rows)
        )
        sample_size = int(cfg.analysis.sample_size_per_group)
        if (
            not _artifact_record_is_complete(
                cfg,
                analysis_dir / "manifest.json",
                analysis_files,
                expected_row_counts={
                    "total": sample_size * len(group_order),
                    **{group: sample_size for group in group_order},
                },
            )
            or not (analysis_dir / str(cfg.report.experiment_filename)).is_file()
        ):
            raise ValueError("LiRA analysis artifacts failed final validation.")
        analysis_manifest = read_json(analysis_dir / "manifest.json")
        if analysis_manifest.get("attack_method") != method or analysis_manifest.get(
            "source_scores_sha256"
        ) != file_sha256(attack_files["scores"]):
            raise ValueError("LiRA analysis lineage does not match the attack scores.")
        analysis_artifact_sha256 = {
            **{name: file_sha256(path) for name, path in analysis_files.items()},
            "manifest": file_sha256(analysis_dir / "manifest.json"),
            "experiment": file_sha256(
                analysis_dir / str(cfg.report.experiment_filename)
            ),
        }

    attack_metrics = read_json(attack_dir / "metrics.json")
    summary_metrics = {
        key: float(value) for key, value in attack_metrics.items() if _is_number(value)
    }
    experiment_path = root / str(cfg.report.experiment_filename)
    write_experiment_markdown(
        experiment_path,
        purpose=(
            f"Complete the {expected_shadow_count}-shadow {method} run for "
            f"{cfg.model.display_name}."
        ),
        hypothesis=str(cfg.experiment.hypothesis),
        command=command,
        overrides=[],
        config_fingerprint_value=config_fingerprint(_resolved_config(cfg)),
        git_state=collect_git_state(project_root),
        environment=extended_environment(project_root),
        artifacts={
            **lineage_experiment_artifacts,
            "candidate_manifest": str(paths["manifest"].relative_to(project_root)),
            "attack_manifest": str(
                (attack_dir / "manifest.json").relative_to(project_root)
            ),
        },
        metrics=summary_metrics,
        conclusion=(
            f"All {expected_shadow_count} adapters, masks, candidates, and attack "
            "outputs passed count, configuration, and SHA256 validation."
        ),
        achieved_purpose=True,
        next_action=(
            "Compare LiRA descriptively with the prior RMIA run; safe filler replacements prevent strict attack-only attribution."
            if method == "online_lira_fixed_variance"
            else "Compare the 100-shadow estimator with the prior five-shadow run."
            if reuse_enabled
            else "Compare the 10-epoch run descriptively with the prior 1-epoch run."
        ),
        wandb_run_id=_active_wandb_run_id(),
    )
    completion_manifest_path = root / "completion_manifest.json"
    completion_manifest = {
        "config_sha256": config_fingerprint(_resolved_config(cfg)),
        "semantic_config_sha256": _semantic_config_fingerprint(_resolved_config(cfg)),
        "model": str(cfg.model.name_or_path),
        "attack_method": method,
        "shadow_models": expected_shadow_count,
        "candidate_rows": len(public_rows),
        **completion_lineage,
        "candidate_manifest_sha256": file_sha256(paths["manifest"]),
        "shadow_artifact_sha256": shadow_records,
        "attack_artifact_sha256": {
            **{name: file_sha256(path) for name, path in attack_files.items()},
            "manifest": file_sha256(attack_dir / "manifest.json"),
            "experiment": file_sha256(attack_dir / str(cfg.report.experiment_filename)),
        },
        "analysis_artifact_sha256": analysis_artifact_sha256,
        "experiment_sha256": file_sha256(experiment_path),
    }
    write_json(completion_manifest_path, completion_manifest)
    success_path = root / "_SUCCESS"
    success_path.write_text("verified\n", encoding="utf-8")
    _log_wandb_stage(
        stage="validation",
        metrics={
            "shadow_models": float(expected_shadow_count),
            "candidate_rows": float(len(public_rows)),
        },
        paths={
            "completion_manifest": completion_manifest_path,
            "experiment": experiment_path,
            "success": success_path,
        },
    )


def _analysis_candidate_rows(
    root: Path,
    *,
    base_generation_root: Path,
    generated_variant: str,
    generated_group_prefix: str,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, Path],
]:
    source_files = {
        "fine_tuned_and_gold_public": (root / "candidates" / "public_candidates.jsonl"),
        "fine_tuned_and_gold_private": (
            root / "candidates" / "evaluator_mapping.jsonl"
        ),
        "base_public": base_generation_root / "public_generated.jsonl",
        "base_private": base_generation_root / "private_generated_labels.jsonl",
    }
    for source_path in source_files.values():
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Required formal artifact is missing: {source_path}"
            )

    source_specs = (
        (
            source_files["fine_tuned_and_gold_public"],
            source_files["fine_tuned_and_gold_private"],
            {
                "gold_squad_target_train": "gold_squad_target_train",
                "gold_squad_validation": "gold_squad_validation",
                "gold_trivia_validation": "gold_trivia_validation",
                f"{generated_group_prefix}_from_squad_train": (
                    f"{generated_variant}_gen_from_squad_train"
                ),
                f"{generated_group_prefix}_from_squad_validation": (
                    f"{generated_variant}_gen_from_squad_validation"
                ),
                f"{generated_group_prefix}_from_trivia_validation": (
                    f"{generated_variant}_gen_from_trivia_validation"
                ),
            },
        ),
        (
            source_files["base_public"],
            source_files["base_private"],
            {
                "gen_from_squad_train": "base_gen_from_squad_train",
                "gen_from_squad_validation": "base_gen_from_squad_validation",
                "gen_from_trivia_validation": "base_gen_from_trivia_validation",
            },
        ),
    )

    public_rows: list[dict[str, object]] = []
    private_rows: list[dict[str, object]] = []
    seen_content_by_id: dict[str, str] = {}
    for public_path, private_path, group_mapping in source_specs:
        public_by_id = {
            str(row["candidate_id"]): row for row in read_jsonl(public_path)
        }
        source_private_rows = read_jsonl(private_path)
        if set(public_by_id) != {
            str(row["candidate_id"]) for row in source_private_rows
        }:
            raise ValueError(
                f"Public and private candidate IDs differ for {public_path}."
            )
        for private_row in source_private_rows:
            source_group = str(private_row["private_group"])
            if source_group not in group_mapping:
                continue
            public_row = public_by_id[str(private_row["candidate_id"])]
            group = group_mapping[source_group]
            content_sha256 = str(public_row["content_sha256"])
            analysis_id = f"analysis_{group}_{content_sha256[:20]}"
            previous_content = seen_content_by_id.get(analysis_id)
            if previous_content is not None:
                if previous_content != content_sha256:
                    raise ValueError(f"Analysis candidate ID collision: {analysis_id}")
                continue
            seen_content_by_id[analysis_id] = content_sha256
            public_rows.append({**public_row, "candidate_id": analysis_id})
            private_rows.append(
                {
                    "candidate_id": analysis_id,
                    "private_group": group,
                }
            )
    return public_rows, private_rows, source_files


def plot_rmia_feature_distribution(
    cfg: DictConfig,
    *,
    project_root: Path,
    command: str,
) -> None:
    feature = str(cfg.analysis.feature)
    if feature != "relative_log_likelihood":
        raise ValueError(
            "The formal RMIA feature plot requires relative_log_likelihood."
        )

    strategy = fine_tuning_strategy(cfg)
    group_order = candidate_groups_for_variant(strategy.generated_variant)
    root = model_root(cfg, project_root, smoke=False)
    public_rows, private_rows, source_files = _analysis_candidate_rows(
        root,
        base_generation_root=base_generation_dir(cfg, project_root, smoke=False),
        generated_variant=strategy.generated_variant,
        generated_group_prefix=strategy.generated_group_prefix,
    )
    sample_size = int(cfg.analysis.sample_size_per_group)
    sampling_seed = int(cfg.analysis.sampling_seed)
    sampled_public, sampled_private = sample_group_candidates(
        public_rows,
        private_rows,
        sample_size=sample_size,
        seed=sampling_seed,
        group_order=group_order,
    )
    candidates = [record_from_public_candidate(row) for row in sampled_public]

    target_checkpoint = target_checkpoint_path(cfg, project_root, smoke=False)
    shadow_checkpoints = shadow_checkpoint_paths(cfg, project_root, smoke=False)
    output_dir = root / "analysis" / feature
    figure_path = output_dir / "relative_log_likelihood_ecdf.png"
    sampled_path = output_dir / "sampled_candidates.jsonl"
    manifest_path = output_dir / "manifest.json"
    resolved_config_path = output_dir / str(cfg.report.resolved_config_filename)
    experiment_path = output_dir / str(cfg.report.experiment_filename)
    feature_definition = (
        "target_mean_logprob - logmeanexp(mean_logprob from every saved shadow "
        f"{strategy.artifact_name})"
    )

    sampled_rows: list[dict[str, object]] | None = None
    if (
        sampled_path.is_file()
        and manifest_path.is_file()
        and not bool(cfg.workflow.force)
    ):
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sample_sha = prior_manifest.get("artifact_sha256", {}).get(
            "sampled_candidates"
        )
        reusable = (
            prior_manifest.get("feature") == feature
            and prior_manifest.get("feature_definition") == feature_definition
            and prior_manifest.get("group_order") == list(group_order)
            and prior_manifest.get("sample_size_per_group") == sample_size
            and prior_manifest.get("reference_shadow_count") == len(shadow_checkpoints)
            and prior_manifest.get("row_counts", {}).get("total")
            == sample_size * len(group_order)
            and expected_sample_sha == file_sha256(sampled_path)
        )
        if reusable:
            sampled_rows = read_jsonl(sampled_path)
            LOGGER.info("Reusing formal RMIA feature scores: %s", sampled_path)

    if sampled_rows is None:
        target_scores = score_with_checkpoint(
            cfg,
            candidates,
            checkpoint_path=target_checkpoint,
            batch_size=int(cfg.attack.batch_size),
        )
        shadow_scores = [
            score_with_checkpoint(
                cfg,
                candidates,
                checkpoint_path=checkpoint,
                batch_size=int(cfg.attack.batch_size),
            )
            for checkpoint in shadow_checkpoints
        ]
        if len(target_scores) != len(candidates) or any(
            len(scores) != len(candidates) for scores in shadow_scores
        ):
            raise ValueError(
                "RMIA feature score count does not match sampled candidates."
            )

        sampled_rows = []
        for row_index, private_row in enumerate(sampled_private):
            per_shadow = [scores[row_index] for scores in shadow_scores]
            relative = population_relative_loglikelihood(
                target_scores[row_index], per_shadow
            )
            sampled_rows.append(
                {
                    "candidate_id": private_row["candidate_id"],
                    "private_group": private_row["private_group"],
                    "sample_rank": private_row["sample_rank"],
                    "target_mean_logprob": target_scores[row_index],
                    "shadow_reference_mean_logprob": (
                        target_scores[row_index] - relative
                    ),
                    "relative_log_likelihood": relative,
                    "num_reference_shadows": len(per_shadow),
                }
            )

    expected_rows = sample_size * len(group_order)
    if len(sampled_rows) != expected_rows or any(
        sum(row["private_group"] == group for row in sampled_rows) != sample_size
        for group in group_order
    ):
        raise ValueError("Reusable RMIA feature scores have invalid group counts.")
    if any(
        int(row.get("num_reference_shadows", 0)) != len(shadow_checkpoints)
        or not math.isfinite(float(row[feature]))
        for row in sampled_rows
    ):
        raise ValueError("Reusable RMIA feature scores contain invalid values.")
    summaries = summarize_group_features(
        sampled_rows, feature=feature, group_order=group_order
    )
    write_jsonl(sampled_path, sampled_rows)
    plot_group_feature_ecdf(
        sampled_rows,
        feature=feature,
        model_display_name=str(cfg.model.display_name),
        output_path=figure_path,
        dpi=int(cfg.analysis.figure_dpi),
        group_order=group_order,
    )
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_config, dict):
        raise TypeError("Resolved Hydra config must be a mapping.")
    write_yaml(resolved_config_path, resolved_config)
    config_sha256 = config_fingerprint(resolved_config)

    artifacts = {
        "figure": figure_path,
        "sampled_candidates": sampled_path,
        "resolved_config": resolved_config_path,
    }
    implementation_files = {
        "plotting": project_root / "src" / "llm_mia" / "plotting.py",
        "workflow": Path(__file__).resolve(),
    }
    manifest = {
        "source_state": collect_git_state(project_root),
        "implementation_files": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in implementation_files.items()
        },
        "config_sha256": config_sha256,
        "seed": int(cfg.runtime.seed),
        "sampling_seed": sampling_seed,
        "sampling_algorithm": (
            "ascending_sha256(seed:rmia-feature:group:candidate_id)"
        ),
        "sampling_without_replacement": True,
        "sampling_before_model_scoring": True,
        "feature": feature,
        "feature_definition": feature_definition,
        "model": str(cfg.model.name_or_path),
        "target_checkpoint": str(target_checkpoint),
        "shadow_checkpoints": [str(path) for path in shadow_checkpoints],
        "reference_shadow_count": len(shadow_checkpoints),
        "group_order": list(group_order),
        "sample_size_per_group": sample_size,
        "row_counts": {
            **{group: sample_size for group in group_order},
            "total": len(sampled_rows),
        },
        "source_files": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in source_files.items()
        },
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "artifact_sha256": {
            name: file_sha256(path) for name, path in artifacts.items()
        },
        "group_summaries": summaries,
    }
    write_json(manifest_path, manifest)

    metrics = {
        f"{group}_{name}": value
        for group, summary in summaries.items()
        for name, value in summary.items()
        if name in {"count", "mean", "median"}
    }
    tracking_artifacts = _log_wandb_stage(
        stage="rmia_feature_plot",
        metrics=metrics,
        paths={**artifacts, "manifest": manifest_path},
    )
    write_experiment_markdown(
        experiment_path,
        purpose=(
            f"Compare nine candidate-group {feature} distributions for "
            f"{cfg.model.display_name}."
        ),
        hypothesis=(
            "Data source and text origin change the target-versus-shadow "
            "relative log-likelihood distribution."
        ),
        command=command,
        overrides=[],
        config_fingerprint_value=config_sha256,
        git_state=collect_git_state(project_root),
        environment=extended_environment(project_root),
        artifacts={
            **{name: str(path) for name, path in artifacts.items()},
            "manifest": str(manifest_path),
            **tracking_artifacts,
        },
        metrics=metrics,
        conclusion=(
            "The figure is descriptive: color encodes the three data sources, "
            f"line style encodes gold/base-generated/{strategy.report_label}-generated "
            "text, and "
            "every curve uses an equal-size deterministic sample."
        ),
        achieved_purpose=True,
        next_action=(
            "Interpret group separation alongside token length and reference scores."
        ),
        wandb_run_id=_active_wandb_run_id(),
    )


def plot_lira_feature_distribution(
    cfg: DictConfig,
    *,
    project_root: Path,
    command: str,
) -> None:
    if attack_method(cfg) != "online_lira_fixed_variance":
        raise ValueError("The LiRA plot stage requires online_lira_fixed_variance.")
    feature = str(cfg.analysis.feature)
    if feature != "online_lira_log_ratio":
        raise ValueError("The LiRA plot stage requires online_lira_log_ratio.")

    root = model_root(cfg, project_root, smoke=False)
    candidate = candidate_paths(cfg, project_root, smoke=False)
    score_path = root / "attack" / attack_score_filename(cfg)
    score_rows = read_jsonl(score_path)
    private_rows = read_jsonl(candidate["private"])
    group_order = tuple(
        dict.fromkeys(str(row["private_group"]) for row in private_rows)
    )
    sampled_rows = sample_group_feature_rows(
        score_rows,
        private_rows,
        feature=feature,
        sample_size=int(cfg.analysis.sample_size_per_group),
        seed=int(cfg.analysis.sampling_seed),
        group_order=group_order,
    )
    summaries = summarize_group_features(
        sampled_rows, feature=feature, group_order=group_order
    )
    output_dir = root / "analysis" / feature
    figure_path = output_dir / "online_lira_log_ratio_ecdf.png"
    sampled_path = output_dir / "sampled_candidates.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"
    experiment_path = output_dir / str(cfg.report.experiment_filename)
    write_jsonl(sampled_path, sampled_rows)
    write_json(summary_path, summaries)
    plot_group_feature_ecdf(
        sampled_rows,
        feature=feature,
        model_display_name=str(cfg.model.display_name),
        output_path=figure_path,
        dpi=int(cfg.analysis.figure_dpi),
        group_order=group_order,
        title="Fixed-variance online LiRA score distributions",
        x_label="LiRA log-likelihood ratio (IN minus OUT)",
    )
    artifacts = {
        "figure": figure_path,
        "sampled_candidates": sampled_path,
        "summary": summary_path,
    }
    manifest = artifact_manifest(
        cfg,
        project_root,
        artifacts,
        row_counts={
            "total": len(sampled_rows),
            **{
                group: sum(row["private_group"] == group for row in sampled_rows)
                for group in group_order
            },
        },
    )
    manifest.update(
        {
            "attack_method": attack_method(cfg),
            "feature": feature,
            "group_order": list(group_order),
            "sample_size_per_group": int(cfg.analysis.sample_size_per_group),
            "source_scores_sha256": file_sha256(score_path),
            "source_private_labels_sha256": file_sha256(candidate["private"]),
        }
    )
    write_json(manifest_path, manifest)
    metrics = {
        f"{group}_{name}": value
        for group, summary in summaries.items()
        for name, value in summary.items()
        if name in {"count", "mean", "median"}
    }
    tracking_artifacts = _log_wandb_stage(
        stage="lira_feature_plot",
        metrics=metrics,
        paths={**artifacts, "manifest": manifest_path},
    )
    write_experiment_markdown(
        experiment_path,
        purpose=f"Compare fixed-variance LiRA score distributions for {cfg.model.display_name}.",
        hypothesis=str(cfg.experiment.hypothesis),
        command=command,
        overrides=[],
        config_fingerprint_value=config_fingerprint(_resolved_config(cfg)),
        git_state=collect_git_state(project_root),
        environment=extended_environment(project_root),
        artifacts={
            **{name: str(path) for name, path in artifacts.items()},
            "manifest": str(manifest_path),
            **tracking_artifacts,
        },
        metrics=metrics,
        conclusion="The ECDF uses an equal-size deterministic sample from each available candidate group.",
        achieved_purpose=True,
        next_action="Interpret LiRA separation together with per-group AUC and low-FPR TPR.",
        wandb_run_id=_active_wandb_run_id(),
    )


def score_with_checkpoint(
    cfg: DictConfig,
    candidates: list[QARecord],
    *,
    checkpoint_path: Path,
    batch_size: int,
) -> list[float]:
    strategy = fine_tuning_strategy(cfg)
    model, tokenizer = strategy.load_for_inference(cfg, checkpoint_path)
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
    score_field: str = "online_rmia_score",
    calibration_scores: list[float] | None,
    fpr_thresholds: list[float],
    bootstrap_samples: int,
    seed: int,
    include_point_tpr: bool = False,
) -> dict[str, float]:
    score_by_id = {}
    for row in score_rows:
        candidate_id_value = str(row["candidate_id"])
        value = float(row[score_field])
        if candidate_id_value in score_by_id:
            raise ValueError(f"Duplicate scored candidate: {candidate_id_value}")
        if not math.isfinite(value):
            raise ValueError(f"Non-finite attack score: {candidate_id_value}")
        score_by_id[candidate_id_value] = value
    labels_by_id: dict[str, int] = {}
    candidate_ids_by_group: dict[str, set[str]] = {}
    for row in private_rows:
        candidate_id_value = str(row["candidate_id"])
        if candidate_id_value not in score_by_id:
            raise ValueError(
                f"Evaluator mapping references an unscored candidate: "
                f"{candidate_id_value}"
            )
        label = int(row["record_membership_label"])
        previous_label = labels_by_id.get(candidate_id_value)
        if previous_label is not None and previous_label != label:
            raise ValueError("Canonical candidate has conflicting membership labels.")
        labels_by_id[candidate_id_value] = label
        group = str(row["private_group"])
        candidate_ids_by_group.setdefault(group, set()).add(candidate_id_value)
    if set(score_by_id) != set(labels_by_id):
        raise ValueError("Scores and evaluator mapping candidate IDs must match.")
    positives = sorted(
        candidate_id_value
        for candidate_id_value in candidate_ids_by_group.get(
            "gold_squad_target_train", set()
        )
        if labels_by_id[candidate_id_value] == 1
    )
    metrics: dict[str, float] = {}
    if calibration_scores is not None:
        metrics["population_calibration_examples"] = float(len(calibration_scores))
        for target_fpr in fpr_thresholds:
            threshold = threshold_for_fpr(calibration_scores, target_fpr)
            metrics[f"threshold_at_population_fpr_{target_fpr}"] = threshold
            metrics[f"tpr_gold_train_at_population_fpr_{target_fpr}"] = fraction_above(
                [score_by_id[cid] for cid in positives], threshold
            )
            for group, group_candidate_ids in sorted(candidate_ids_by_group.items()):
                negatives = sorted(
                    candidate_id_value
                    for candidate_id_value in group_candidate_ids
                    if labels_by_id[candidate_id_value] == 0
                )
                metrics[f"fpr_{group}_at_population_fpr_{target_fpr}"] = fraction_above(
                    [score_by_id[cid] for cid in negatives], threshold
                )
    for group, group_candidate_ids in sorted(candidate_ids_by_group.items()):
        negatives = sorted(
            candidate_id_value
            for candidate_id_value in group_candidate_ids
            if labels_by_id[candidate_id_value] == 0
        )
        if positives and negatives:
            positive_scores = [score_by_id[cid] for cid in positives]
            negative_scores = [score_by_id[cid] for cid in negatives]
            y_true = [1] * len(positives) + [0] * len(negatives)
            values = positive_scores + negative_scores
            metrics[f"auc_gold_train_vs_{group}"] = float(roc_auc_score(y_true, values))
            if include_point_tpr:
                for target_fpr in fpr_thresholds:
                    metrics[f"tpr_at_fpr_{target_fpr}_gold_train_vs_{group}"] = (
                        tpr_at_fpr(values, y_true, target_fpr)
                    )
            metrics.update(
                {
                    f"{key}_gold_train_vs_{group}": value
                    for key, value in bootstrap_binary_metrics(
                        positive_scores,
                        negative_scores,
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
    resolved_config = _resolved_config(cfg)
    return {
        "source_commit": collect_git_state(project_root),
        "config_sha256": config_fingerprint(resolved_config),
        "semantic_config_sha256": _semantic_config_fingerprint(resolved_config),
        "seed": int(cfg.runtime.seed),
        "model": str(cfg.model.name_or_path),
        "tokenizer": str(cfg.model.name_or_path),
        "row_counts": row_counts,
        "files": {name: str(path) for name, path in files.items()},
        "file_sha256": {
            name: file_sha256(path) for name, path in files.items() if path.exists()
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
