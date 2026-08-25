from __future__ import annotations

import math
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.experiment import (
    collect_git_state,
    config_fingerprint,
    write_experiment_markdown,
)
from src.llm_mia.data import (
    QARecord,
    file_sha256,
    read_json,
    read_jsonl,
    record_from_public_candidate,
    write_json,
    write_jsonl,
)
from src.llm_mia.feature_matrix import (
    FEATURE_NAME,
    common_feature_limits,
    feature_cache_key,
    plot_feature_matrix_ecdf,
    population_center,
)
from src.llm_mia.plotting import (
    sample_group_candidates,
    summarize_group_features,
)


def _setting_specs(cfg: DictConfig, project_root: Path) -> list[dict[str, object]]:
    specs = [
        {
            "key": str(key),
            "label": str(value.label),
            "root": project_root / str(value.source_output_root) / str(cfg.model.key),
            "expected_epochs": int(value.expected_epochs),
        }
        for key, value in cfg.analysis.settings.items()
    ]
    if len(specs) != 5 or len({str(spec["key"]) for spec in specs}) != 5:
        raise ValueError("Feature matrix requires exactly five distinct settings.")
    return specs


def _tree_file_sha256(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(f"Artifact directory is missing: {root}")
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _validate_checkpoint_root(
    cfg: DictConfig,
    *,
    root: Path,
    expected_epochs: int,
) -> dict[str, object]:
    completion_path = root / "completion_manifest.json"
    success_path = root / "_SUCCESS"
    if not completion_path.is_file() or not success_path.is_file():
        raise FileNotFoundError(f"Checkpoint setting is not complete: {root}")
    if success_path.read_text(encoding="utf-8").strip() != "verified":
        raise ValueError(f"Checkpoint success marker is invalid: {root}")
    completion = read_json(completion_path)
    shadow_total = int(cfg.analysis.reference_shadow_count)
    if (
        completion.get("model") != str(cfg.model.name_or_path)
        or completion.get("shadow_models") != shadow_total
    ):
        raise ValueError(f"Checkpoint completion manifest is incompatible: {root}")

    target_run = root / "target" / f"seed_{int(cfg.runtime.seed)}"
    shadow_runs = [
        root / "shadows" / f"shadow_{index:02d}" for index in range(shadow_total)
    ]
    for run in [target_run, *shadow_runs]:
        resolved_path = run / str(cfg.report.resolved_config_filename)
        adapter_path = run / "adapter" / "adapter_model.safetensors"
        if not resolved_path.is_file() or not adapter_path.is_file():
            raise FileNotFoundError(f"Incomplete checkpoint run: {run}")
        resolved = OmegaConf.load(resolved_path)
        if int(resolved.train.epochs) != expected_epochs:
            raise ValueError(
                f"Expected {expected_epochs} epochs in {resolved_path}, "
                f"found {resolved.train.epochs}."
            )

    actual_shadow_hashes = {
        f"shadow_{index:02d}": _tree_file_sha256(run)
        for index, run in enumerate(shadow_runs)
    }
    recorded_shadow_hashes = completion.get("shadow_artifact_sha256", {})
    if set(recorded_shadow_hashes) != set(actual_shadow_hashes) or any(
        not isinstance(recorded, dict)
        or any(
            actual_shadow_hashes[name].get(path) != digest
            for path, digest in recorded.items()
        )
        for name, recorded in recorded_shadow_hashes.items()
    ):
        raise ValueError(f"Completion manifest shadow hashes differ: {root}")
    target_hashes = _tree_file_sha256(root / "target")
    recorded_target = completion.get("target_artifact_sha256")
    if recorded_target is not None and any(
        target_hashes.get(path) != digest for path, digest in recorded_target.items()
    ):
        raise ValueError(f"Completion manifest target hashes differ: {root}")
    return {
        "completion_manifest_sha256": file_sha256(completion_path),
        "success_sha256": file_sha256(success_path),
        "target_checkpoint": str(target_run / "adapter"),
        "shadow_checkpoints": [str(run / "adapter") for run in shadow_runs],
        "target_tree_sha256": target_hashes,
        "shadow_tree_sha256": actual_shadow_hashes,
        "expected_epochs": expected_epochs,
    }


def _validate_candidate_source(candidate_source_root: Path) -> dict[str, Path]:
    candidate_root = candidate_source_root / "candidates"
    generated_root = candidate_source_root / "generated"
    base_root = generated_root / "base"
    files = {
        "candidate_manifest": candidate_root / "manifest.json",
        "candidate_public": candidate_root / "public_candidates.jsonl",
        "candidate_private": candidate_root / "private_labels.jsonl",
        "candidate_masks": candidate_root / "shadow_masks.csv",
        "generated_manifest": generated_root / "manifest.json",
        "generated_public": generated_root / "public_generated.jsonl",
        "generated_private": generated_root / "private_generated_labels.jsonl",
        "generated_metrics": generated_root / "metrics.json",
        "base_manifest": base_root / "manifest.json",
        "base_public": base_root / "public_generated.jsonl",
        "base_private": base_root / "private_generated_labels.jsonl",
        "base_metrics": base_root / "metrics.json",
        "base_resolved_config": base_root / "resolved_config.yaml",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Matrix candidate source is missing: {path}")
    manifest_specs = (
        (
            files["candidate_manifest"],
            {
                "public_candidates": files["candidate_public"],
                "private_labels": files["candidate_private"],
                "shadow_masks": files["candidate_masks"],
            },
        ),
        (
            files["generated_manifest"],
            {
                "public_generated": files["generated_public"],
                "private_generated_labels": files["generated_private"],
                "metrics": files["generated_metrics"],
            },
        ),
        (
            files["base_manifest"],
            {
                "public_generated": files["base_public"],
                "private_generated_labels": files["base_private"],
                "metrics": files["base_metrics"],
                "resolved_config": files["base_resolved_config"],
            },
        ),
    )
    for manifest_path, bound_files in manifest_specs:
        recorded = read_json(manifest_path).get("file_sha256", {})
        for name, path in bound_files.items():
            if recorded.get(name) != file_sha256(path):
                raise ValueError(f"Source manifest hash mismatch: {path}")
    return files


def _analysis_candidate_rows(
    candidate_source_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source_specs = (
        (
            candidate_source_root / "candidates" / "public_candidates.jsonl",
            candidate_source_root / "candidates" / "private_labels.jsonl",
            {
                "gold_squad_target_train": "gold_squad_target_train",
                "gold_squad_validation": "gold_squad_validation",
                "gold_trivia_validation": "gold_trivia_validation",
                "gen_from_squad_train": "lora_gen_from_squad_train",
                "gen_from_squad_validation": "lora_gen_from_squad_validation",
                "gen_from_trivia_validation": "lora_gen_from_trivia_validation",
            },
        ),
        (
            candidate_source_root / "generated" / "base" / "public_generated.jsonl",
            candidate_source_root
            / "generated"
            / "base"
            / "private_generated_labels.jsonl",
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
        source_private = read_jsonl(private_path)
        if set(public_by_id) != {str(row["candidate_id"]) for row in source_private}:
            raise ValueError(f"Public and private candidate IDs differ: {public_path}")
        for private_row in source_private:
            source_group = str(private_row["private_group"])
            if source_group not in group_mapping:
                continue
            public_row = public_by_id[str(private_row["candidate_id"])]
            group = group_mapping[source_group]
            content_sha256 = str(public_row["content_sha256"])
            analysis_id = f"analysis_{group}_{content_sha256[:20]}"
            previous = seen_content_by_id.get(analysis_id)
            if previous is not None:
                if previous != content_sha256:
                    raise ValueError(f"Analysis candidate ID collision: {analysis_id}")
                continue
            seen_content_by_id[analysis_id] = content_sha256
            public_rows.append({**public_row, "candidate_id": analysis_id})
            private_rows.append({"candidate_id": analysis_id, "private_group": group})
    return public_rows, private_rows


def _score_checkpoint(
    cfg: DictConfig,
    *,
    records: list[QARecord],
    input_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: dict[str, str],
    cache_root: Path,
) -> list[float]:
    from src.llm_mia.workflow import attack_scoring_batch_size, score_with_checkpoint

    cache_key = feature_cache_key(
        {
            "input_sha256": input_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "max_length": int(cfg.tokenizer.max_length),
            "loss_chunk_tokens": int(cfg.attack.loss_chunk_tokens),
        }
    )
    cache_dir = cache_root / "checkpoints" / cache_key
    scores_path = cache_dir / "scores.jsonl"
    manifest_path = cache_dir / "manifest.json"
    if (
        scores_path.is_file()
        and manifest_path.is_file()
        and not bool(cfg.workflow.force)
    ):
        manifest = read_json(manifest_path)
        rows = read_jsonl(scores_path)
        if (
            manifest.get("cache_key") == cache_key
            and manifest.get("row_count") == len(records)
            and manifest.get("scores_sha256") == file_sha256(scores_path)
            and len(rows) == len(records)
            and all(
                int(row.get("row_index", -1)) == index
                and math.isfinite(float(row.get("mean_logprob", math.nan)))
                for index, row in enumerate(rows)
            )
        ):
            return [float(row["mean_logprob"]) for row in rows]
        raise ValueError(f"Existing checkpoint cache is invalid: {cache_dir}")

    scores = score_with_checkpoint(
        cfg,
        records,
        checkpoint_path=checkpoint_path,
        batch_size=attack_scoring_batch_size(cfg),
    )
    if len(scores) != len(records) or any(
        not math.isfinite(float(value)) for value in scores
    ):
        raise ValueError(f"Invalid checkpoint scores: {checkpoint_path}")
    rows = [
        {"row_index": index, "mean_logprob": float(value)}
        for index, value in enumerate(scores)
    ]
    write_jsonl(scores_path, rows)
    write_json(
        manifest_path,
        {
            "cache_key": cache_key,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "input_sha256": input_sha256,
            "row_count": len(rows),
            "scores_sha256": file_sha256(scores_path),
        },
    )
    return scores


def _score_setting(
    cfg: DictConfig,
    *,
    records: list[QARecord],
    candidate_count: int,
    sampled_private: list[dict[str, object]],
    population: list[QARecord],
    input_sha256: str,
    lineage: dict[str, object],
    cache_root: Path,
) -> dict[str, object]:
    from src.llm_mia.analysis import population_relative_loglikelihood

    target_scores = _score_checkpoint(
        cfg,
        records=records,
        input_sha256=input_sha256,
        checkpoint_path=Path(str(lineage["target_checkpoint"])),
        checkpoint_sha256=lineage["target_tree_sha256"],
        cache_root=cache_root,
    )
    shadow_scores = [
        _score_checkpoint(
            cfg,
            records=records,
            input_sha256=input_sha256,
            checkpoint_path=Path(str(checkpoint)),
            checkpoint_sha256=lineage["shadow_tree_sha256"][f"shadow_{index:02d}"],
            cache_root=cache_root,
        )
        for index, checkpoint in enumerate(lineage["shadow_checkpoints"])
    ]
    relative = [
        population_relative_loglikelihood(
            target_scores[index], [scores[index] for scores in shadow_scores]
        )
        for index in range(len(records))
    ]
    candidate_relative = relative[:candidate_count]
    population_relative = relative[candidate_count:]
    center, centered_population = population_center(population_relative)
    candidate_rows = [
        {
            "candidate_id": private["candidate_id"],
            "private_group": private["private_group"],
            "sample_rank": private["sample_rank"],
            "target_mean_logprob": target_scores[index],
            "shadow_reference_mean_logprob": (
                target_scores[index] - candidate_relative[index]
            ),
            "relative_log_likelihood": candidate_relative[index],
            FEATURE_NAME: candidate_relative[index] - center,
            "num_reference_shadows": 5,
        }
        for index, private in enumerate(sampled_private)
    ]
    population_rows = [
        {
            "record_id": record.record_id,
            "target_mean_logprob": target_scores[candidate_count + index],
            "relative_log_likelihood": population_relative[index],
            FEATURE_NAME: centered_population[index],
            "num_reference_shadows": 5,
        }
        for index, record in enumerate(population)
    ]
    return {
        "candidate_rows": candidate_rows,
        "population_rows": population_rows,
        "center": center,
    }


def plot_population_centered_feature_matrix(
    cfg: DictConfig,
    *,
    project_root: Path,
    command: str,
) -> None:
    from src.llm_mia.workflow import (
        _active_wandb_run_id,
        _log_wandb_stage,
        data_root,
        extended_environment,
        load_split_file,
        model_root,
    )

    if str(cfg.analysis.feature) != FEATURE_NAME:
        raise ValueError(f"Feature matrix requires {FEATURE_NAME}.")
    if int(cfg.analysis.reference_shadow_count) != 5:
        raise ValueError("Feature matrix requires exactly five reference shadows.")

    root = model_root(cfg, project_root, smoke=False)
    candidate_source_root = (
        project_root
        / str(cfg.analysis.candidate_source_output_root)
        / str(cfg.model.key)
    )
    candidate_files = _validate_candidate_source(candidate_source_root)
    public_rows, private_rows = _analysis_candidate_rows(candidate_source_root)
    sample_size = int(cfg.analysis.sample_size_per_group)
    sampled_public, sampled_private = sample_group_candidates(
        public_rows,
        private_rows,
        sample_size=sample_size,
        seed=int(cfg.analysis.sampling_seed),
    )
    candidate_count = 9 * sample_size
    if len(sampled_public) != candidate_count:
        raise ValueError("Feature matrix sampled candidate count differs.")
    sampled_candidates = [record_from_public_candidate(row) for row in sampled_public]

    population_path = data_root(cfg, project_root) / "squad_validation_population.jsonl"
    population_sha256 = str(cfg.analysis.population_sha256)
    if file_sha256(population_path) != population_sha256:
        raise ValueError("Feature matrix population SHA256 differs.")
    population = load_split_file(cfg, project_root, "squad_validation_population")
    if len(population) != int(cfg.data.squad_validation_population_size):
        raise ValueError("Feature matrix population row count differs.")
    all_records = [*sampled_candidates, *population]
    score_input_sha256 = feature_cache_key(
        {
            "sampled_candidates": [
                {
                    "candidate_id": row["candidate_id"],
                    "content_sha256": row["content_sha256"],
                    "private_group": private["private_group"],
                    "sample_rank": private["sample_rank"],
                }
                for row, private in zip(sampled_public, sampled_private, strict=True)
            ],
            "population_sha256": population_sha256,
        }
    )

    suite_dir = root / "candidate_suite"
    sampled_public_path = suite_dir / "sampled_public_candidates.jsonl"
    sampled_private_path = suite_dir / "sampled_private_groups.jsonl"
    suite_manifest_path = suite_dir / "manifest.json"
    write_jsonl(sampled_public_path, sampled_public)
    write_jsonl(sampled_private_path, sampled_private)
    write_json(
        suite_manifest_path,
        {
            "source_root": str(candidate_source_root.relative_to(project_root)),
            "source_artifact_sha256": {
                name: file_sha256(path) for name, path in candidate_files.items()
            },
            "sampling_seed": int(cfg.analysis.sampling_seed),
            "sampling_algorithm": (
                "ascending_sha256(seed:rmia-feature:group:candidate_id)"
            ),
            "sample_size_per_group": sample_size,
            "row_count": candidate_count,
            "score_input_sha256": score_input_sha256,
            "sampled_public_sha256": file_sha256(sampled_public_path),
            "sampled_private_sha256": file_sha256(sampled_private_path),
            "population_path": str(population_path.relative_to(project_root)),
            "population_sha256": population_sha256,
            "population_rows": len(population),
        },
    )

    settings = _setting_specs(cfg, project_root)
    cache_root = root / "cache"
    scored: dict[str, dict[str, object]] = {}
    checkpoint_lineage: dict[str, dict[str, object]] = {}
    for spec in settings:
        key = str(spec["key"])
        lineage = _validate_checkpoint_root(
            cfg,
            root=Path(spec["root"]),
            expected_epochs=int(spec["expected_epochs"]),
        )
        checkpoint_lineage[key] = lineage
        scored[key] = _score_setting(
            cfg,
            records=all_records,
            candidate_count=candidate_count,
            sampled_private=sampled_private,
            population=population,
            input_sha256=score_input_sha256,
            lineage=lineage,
            cache_root=cache_root,
        )

    alias_keys = ("one_epoch_rmia", "one_epoch_lira")
    if (
        checkpoint_lineage[alias_keys[0]]["target_tree_sha256"]
        != checkpoint_lineage[alias_keys[1]]["target_tree_sha256"]
        or checkpoint_lineage[alias_keys[0]]["shadow_tree_sha256"]
        != checkpoint_lineage[alias_keys[1]]["shadow_tree_sha256"]
        or scored[alias_keys[0]] != scored[alias_keys[1]]
    ):
        raise ValueError("1-epoch RMIA and LiRA feature data must be identical.")

    x_limits = common_feature_limits(
        {key: value["candidate_rows"] for key, value in scored.items()}
    )
    setting_manifests: dict[str, str] = {}
    figures: dict[str, Path] = {}
    metrics: dict[str, float] = {}
    for spec in settings:
        key = str(spec["key"])
        setting_dir = root / "settings" / key
        figure_path = setting_dir / "population_centered_3x3_ecdf.png"
        candidates_path = setting_dir / "sampled_scores.jsonl"
        population_scores_path = setting_dir / "population_scores.jsonl"
        summary_path = setting_dir / "summary.json"
        manifest_path = setting_dir / "manifest.json"
        experiment_path = setting_dir / str(cfg.report.experiment_filename)
        candidate_rows = scored[key]["candidate_rows"]
        population_rows = scored[key]["population_rows"]
        summaries = summarize_group_features(candidate_rows, feature=FEATURE_NAME)
        write_jsonl(candidates_path, candidate_rows)
        write_jsonl(population_scores_path, population_rows)
        write_json(
            summary_path,
            {
                "population_median_relative_log_likelihood": scored[key]["center"],
                "shared_x_limits": list(x_limits),
                "groups": summaries,
            },
        )
        plot_feature_matrix_ecdf(
            candidate_rows,
            model_display_name=str(cfg.model.display_name),
            setting_label=str(spec["label"]),
            output_path=figure_path,
            dpi=int(cfg.analysis.figure_dpi),
            x_limits=x_limits,
        )
        write_json(
            manifest_path,
            {
                "setting": key,
                "setting_label": str(spec["label"]),
                "feature": FEATURE_NAME,
                "feature_definition": (
                    "target_mean_logprob - logmeanexp(five shadow mean_logprobs) "
                    "- median(population relative log-likelihood)"
                ),
                "source_root": str(Path(spec["root"]).relative_to(project_root)),
                "checkpoint_lineage": checkpoint_lineage[key],
                "candidate_suite_manifest_sha256": file_sha256(suite_manifest_path),
                "sample_size_per_group": sample_size,
                "candidate_rows": len(candidate_rows),
                "population_rows": len(population_rows),
                "population_sha256": population_sha256,
                "population_center": scored[key]["center"],
                "shared_x_limits": list(x_limits),
                "artifacts_sha256": {
                    "figure": file_sha256(figure_path),
                    "sampled_scores": file_sha256(candidates_path),
                    "population_scores": file_sha256(population_scores_path),
                    "summary": file_sha256(summary_path),
                },
            },
        )
        write_experiment_markdown(
            experiment_path,
            purpose=(
                f"Compare population-centered 3x3 MIA feature distributions for "
                f"{cfg.model.display_name} under {spec['label']}."
            ),
            hypothesis=str(cfg.experiment.hypothesis),
            command=command,
            overrides=[],
            config_fingerprint_value=config_fingerprint(
                OmegaConf.to_container(cfg, resolve=True)
            ),
            git_state=collect_git_state(project_root),
            environment=extended_environment(project_root),
            artifacts={
                "figure": str(figure_path),
                "sampled_scores": str(candidates_path),
                "population_scores": str(population_scores_path),
                "summary": str(summary_path),
                "manifest": str(manifest_path),
            },
            metrics={
                "population_center": float(scored[key]["center"]),
                "sampled_candidates": float(len(candidate_rows)),
            },
            conclusion=(
                "All nine cells use the same deterministic candidate suite; only "
                "the target and five-shadow checkpoint ensemble varies by setting."
            ),
            achieved_purpose=True,
            next_action="Compare source and text-origin shifts across settings.",
            wandb_run_id=_active_wandb_run_id(),
        )
        setting_manifests[key] = file_sha256(manifest_path)
        figures[key] = figure_path
        metrics[f"{key}/population_center"] = float(scored[key]["center"])

    alias_candidate_hashes = [
        file_sha256(root / "settings" / key / "sampled_scores.jsonl")
        for key in alias_keys
    ]
    alias_population_hashes = [
        file_sha256(root / "settings" / key / "population_scores.jsonl")
        for key in alias_keys
    ]
    if len(set(alias_candidate_hashes)) != 1 or len(set(alias_population_hashes)) != 1:
        raise ValueError("1-epoch RMIA and LiRA written score hashes differ.")

    completion_path = root / "completion_manifest.json"
    success_path = root / "_SUCCESS"
    write_json(
        completion_path,
        {
            "model": str(cfg.model.name_or_path),
            "feature": FEATURE_NAME,
            "settings": setting_manifests,
            "figures_sha256": {key: file_sha256(path) for key, path in figures.items()},
            "candidate_suite_manifest_sha256": file_sha256(suite_manifest_path),
            "population_sha256": population_sha256,
            "shared_x_limits": list(x_limits),
            "rmia_lira_shared_score_sha256": {
                "sampled_scores": alias_candidate_hashes[0],
                "population_scores": alias_population_hashes[0],
            },
        },
    )
    success_path.write_text("verified\n", encoding="utf-8")
    _log_wandb_stage(
        stage="population_centered_feature_matrix",
        metrics=metrics,
        paths={
            **{f"figure_{key}": path for key, path in figures.items()},
            "completion_manifest": completion_path,
            "success": success_path,
        },
    )
