from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.experiment import config_fingerprint
from src.llm_mia.data import file_sha256, read_json, read_jsonl


class ReuseValidationError(ValueError):
    """Raised when a prior experiment artifact cannot be reused safely."""


_LORA_TARGET_REUSE_CONFIG_SECTIONS = (
    "runtime",
    "model",
    "data",
    "tokenizer",
    "lora",
    "optimizer",
    "scheduler",
    "loss",
    "precision",
    "train",
    "checkpoint",
)
_LORA_INFERENCE_REUSE_CONFIG_SECTIONS = (
    "runtime",
    "model",
    "data",
    "tokenizer",
    "lora",
    "precision",
    "eval",
    "generation",
)
_HISTORICAL_TARGET_IGNORED_CONFIG_KEYS = {
    "tokenizer": {"preprocessing_workers"},
}
_MASK_REUSE_GOLD_GROUPS = (
    "gold_squad_target_train",
    "gold_squad_validation",
    "gold_trivia_validation",
)
_MASK_REUSE_GENERATED_GROUPS = (
    "gen_from_squad_train",
    "gen_from_squad_validation",
    "gen_from_trivia_validation",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReuseValidationError(message)


def _resolved_yaml(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Missing resolved config: {path}")
    resolved = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    _require(isinstance(resolved, dict), f"Resolved config is not a mapping: {path}")
    return resolved


def _section_without_ignored_keys(
    config: dict[str, Any], section: str, ignored_keys: set[str]
) -> Any:
    value = config.get(section)
    if not ignored_keys or not isinstance(value, dict):
        return value
    return {key: item for key, item in value.items() if key not in ignored_keys}


def _require_matching_config_sections(
    source: dict[str, Any],
    expected: dict[str, Any],
    sections: tuple[str, ...],
    *,
    context: str,
    ignored_keys: dict[str, set[str]] | None = None,
) -> None:
    ignored_keys = ignored_keys or {}
    for section in sections:
        ignored = ignored_keys.get(section, set())
        _require(
            _section_without_ignored_keys(source, section, ignored)
            == _section_without_ignored_keys(expected, section, ignored),
            f"Reusable {context} config mismatch: {section}",
        )


def _tree_hashes(root: Path, *, require_nonempty: bool = True) -> dict[str, str]:
    _require(root.is_dir(), f"Missing reusable artifact directory: {root}")
    hashes = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if require_nonempty:
        _require(bool(hashes), f"Reusable artifact directory is empty: {root}")
    return hashes


def _copy_tree_immutable(
    source: Path,
    destination: Path,
    *,
    expected_hashes: dict[str, str],
    force: bool,
) -> None:
    if destination.exists():
        if _tree_hashes(destination, require_nonempty=False) == expected_hashes:
            return
        _require(
            force,
            f"Existing reused artifact differs from its source: {destination}",
        )
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    _require(
        _tree_hashes(destination) == expected_hashes,
        f"Copied artifact failed hash verification: {destination}",
    )


def _copy_files_immutable(
    source: Path,
    destination: Path,
    names: tuple[str, ...],
    *,
    force: bool,
) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    expected = {}
    for name in names:
        source_path = source / name
        destination_path = destination / name
        _require(source_path.is_file(), f"Missing reusable artifact: {source_path}")
        digest = file_sha256(source_path)
        expected[name] = digest
        if destination_path.exists() and file_sha256(destination_path) != digest:
            _require(
                force,
                f"Existing reused artifact differs from its source: {destination_path}",
            )
            destination_path.unlink()
        if not destination_path.exists():
            shutil.copy2(source_path, destination_path)
        _require(
            file_sha256(destination_path) == digest,
            f"Copied artifact failed hash verification: {destination_path}",
        )
    return expected


def prepare_lora_shadow_expansion_inputs(
    *,
    project_root: Path,
    source_model_root: Path,
    destination_model_root: Path,
    expected_config: dict[str, Any],
    expected_source_shadow_count: int,
    seed: int,
    target_eval_name: str,
    force: bool,
) -> dict[str, Any]:
    """Validate and copy immutable non-shadow inputs from a prior LoRA run."""
    _require(
        source_model_root.resolve() != destination_model_root.resolve(),
        "Source and destination model roots must differ.",
    )
    source_target = source_model_root / "target" / f"seed_{seed}"
    source_config_path = source_target / "resolved_config.yaml"
    source_config = _resolved_yaml(source_config_path)
    _require_matching_config_sections(
        source_config,
        expected_config,
        _LORA_TARGET_REUSE_CONFIG_SECTIONS,
        context="LoRA target",
        ignored_keys=_HISTORICAL_TARGET_IGNORED_CONFIG_KEYS,
    )
    _require(
        source_config.get("paths", {}).get("data_dir")
        == expected_config.get("paths", {}).get("data_dir"),
        "Reusable LoRA run data path mismatch.",
    )
    source_shadow = dict(source_config.get("shadow", {}))
    expected_shadow = dict(expected_config.get("shadow", {}))
    _require(
        source_shadow.pop("count", None) == expected_source_shadow_count,
        "Reusable LoRA run has the wrong shadow count.",
    )
    expected_shadow.pop("count", None)
    _require(
        source_shadow == expected_shadow,
        "Reusable LoRA run shadow settings differ beyond shadow.count.",
    )

    target_required = (
        source_target / "adapter" / "adapter_config.json",
        source_target / "train_manifest.jsonl",
        source_target / "metrics.json",
        source_target / "resolved_config.yaml",
        source_target / "experiment.md",
    )
    _require(
        all(path.is_file() for path in target_required),
        f"Reusable target record is incomplete: {source_target}",
    )
    _require(
        any((source_target / "adapter").glob("*.safetensors")),
        f"Reusable target adapter weights are missing: {source_target / 'adapter'}",
    )

    source_eval = source_model_root / "eval"
    for eval_name in ("base", target_eval_name):
        eval_root = source_eval / eval_name
        _require(
            all(
                (eval_root / name).is_file()
                for name in ("metrics.json", "resolved_config.yaml", "experiment.md")
            ),
            f"Reusable evaluation record is incomplete: {eval_root}",
        )
        eval_config = _resolved_yaml(eval_root / "resolved_config.yaml")
        _require_matching_config_sections(
            eval_config,
            expected_config,
            _LORA_INFERENCE_REUSE_CONFIG_SECTIONS,
            context=eval_name,
        )

    source_generated = source_model_root / "generated"
    generated_files = (
        "public_generated.jsonl",
        "private_generated_labels.jsonl",
        "metrics.json",
        "manifest.json",
        "experiment.md",
    )
    generated_manifest = read_json(source_generated / "manifest.json")
    generated_config_path = source_generated / "resolved_config.yaml"
    if generated_config_path.is_file():
        generated_config = _resolved_yaml(generated_config_path)
        generated_files = (*generated_files, "resolved_config.yaml")
        generated_config_validation = "generated_resolved_config"
    else:
        generated_config_path = source_eval / target_eval_name / "resolved_config.yaml"
        generated_config = _resolved_yaml(generated_config_path)
        generated_fingerprint = generated_manifest.get("config_sha256")
        _require(
            generated_fingerprint == config_fingerprint(generated_config),
            "Historical generated config fingerprint does not match the "
            "target evaluation resolved config.",
        )
        experiment_text = (source_generated / "experiment.md").read_text(
            encoding="utf-8"
        )
        _require(
            f"`{generated_fingerprint}`" in experiment_text,
            "Historical generated experiment record has a different config "
            "fingerprint.",
        )
        generated_config_validation = "target_eval_fingerprint"
    _require_matching_config_sections(
        generated_config,
        expected_config,
        _LORA_INFERENCE_REUSE_CONFIG_SECTIONS,
        context="generated",
    )
    generated_hashes = generated_manifest.get("file_sha256", {})
    generated_manifest_names = {
        "public_generated.jsonl": "public_generated",
        "private_generated_labels.jsonl": "private_generated_labels",
        "metrics.json": "metrics",
    }
    for name, manifest_name in generated_manifest_names.items():
        path = source_generated / name
        _require(path.is_file(), f"Missing reusable generated artifact: {path}")
        _require(
            generated_hashes.get(manifest_name) == file_sha256(path),
            f"Reusable generated artifact hash mismatch: {path}",
        )

    target_hashes = _tree_hashes(source_model_root / "target")
    eval_hashes = _tree_hashes(source_eval)
    _copy_tree_immutable(
        source_model_root / "target",
        destination_model_root / "target",
        expected_hashes=target_hashes,
        force=force,
    )
    _copy_tree_immutable(
        source_eval,
        destination_model_root / "eval",
        expected_hashes=eval_hashes,
        force=force,
    )
    copied_generated_hashes = _copy_files_immutable(
        source_generated,
        destination_model_root / "generated",
        generated_files,
        force=force,
    )
    return {
        "status": "reused",
        "source_model_root": str(source_model_root.relative_to(project_root)),
        "destination_model_root": str(destination_model_root.relative_to(project_root)),
        "expected_source_shadow_count": expected_source_shadow_count,
        "destination_shadow_count": int(expected_config["shadow"]["count"]),
        "validated_config_sections": {
            "target": list(_LORA_TARGET_REUSE_CONFIG_SECTIONS),
            "evaluation": list(_LORA_INFERENCE_REUSE_CONFIG_SECTIONS),
            "generated": list(_LORA_INFERENCE_REUSE_CONFIG_SECTIONS),
        },
        "historical_config_compatibility": {
            "target_ignored_keys": {
                section: sorted(keys)
                for section, keys in _HISTORICAL_TARGET_IGNORED_CONFIG_KEYS.items()
            },
            "generated_config_validation": generated_config_validation,
            "generated_config_source": str(
                generated_config_path.relative_to(project_root)
            ),
        },
        "source_config_sha256": file_sha256(source_config_path),
        "source_state": generated_manifest.get("source_commit"),
        "artifact_sha256": {
            "target": target_hashes,
            "eval": eval_hashes,
            "generated": copied_generated_hashes,
        },
    }


def validate_lora_candidate_artifact(
    root: Path,
    *,
    expected_model: str,
    expected_tokenizer: str,
    expected_seed: int,
    expected_shadow_count: int,
) -> dict[str, Any]:
    """Validate the exact candidate tables reused by a shadow expansion."""
    manifest_path = root / "manifest.json"
    public_path = root / "public_candidates.jsonl"
    private_path = root / "private_labels.jsonl"
    masks_path = root / "shadow_masks.csv"
    manifest = read_json(manifest_path)
    _require(manifest.get("model") == expected_model, "Candidate model mismatch.")
    _require(
        manifest.get("tokenizer") == expected_tokenizer,
        "Candidate tokenizer mismatch.",
    )
    _require(manifest.get("seed") == expected_seed, "Candidate seed mismatch.")

    file_hashes = manifest.get("file_sha256", {})
    files = {
        "public_candidates": public_path,
        "private_labels": private_path,
        "shadow_masks": masks_path,
    }
    for name, path in files.items():
        _require(path.is_file(), f"Missing candidate artifact: {path}")
        _require(
            file_hashes.get(name) == file_sha256(path),
            f"Candidate artifact hash mismatch: {path}",
        )

    public_rows = read_jsonl(public_path)
    private_rows = read_jsonl(private_path)
    public_ids = [str(row.get("candidate_id", "")) for row in public_rows]
    private_ids = [str(row.get("candidate_id", "")) for row in private_rows]
    _require(bool(public_ids), "Candidate table is empty.")
    _require(
        len(public_ids) == len(set(public_ids)),
        "Candidate IDs are missing or duplicated.",
    )
    _require(public_ids == private_ids, "Public/private candidate order differs.")

    row_counts = manifest.get("row_counts", {})
    _require(
        row_counts.get("public_candidates") == len(public_rows)
        and row_counts.get("private_labels") == len(private_rows)
        and row_counts.get("shadow_models") == expected_shadow_count,
        "Candidate manifest row counts differ.",
    )
    with masks_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        mask_rows = list(reader)
    expected_header = [
        "candidate_id",
        *[f"shadow_{index:02d}" for index in range(expected_shadow_count)],
    ]
    _require(header == expected_header, "Candidate mask header differs.")
    _require(
        [row[0] for row in mask_rows] == public_ids,
        "Candidate mask order differs from the public table.",
    )
    _require(
        all(
            len(row) == expected_shadow_count + 1 and set(row[1:]) == {"0", "1"}
            for row in mask_rows
        ),
        "Candidate masks contain invalid values.",
    )
    return {
        "status": "reused",
        "path": str(root),
        "source_shadow_count": expected_shadow_count,
        "candidate_rows": len(public_rows),
        "manifest_sha256": file_sha256(manifest_path),
        "file_sha256": {name: file_hashes[name] for name in files},
        "source_state": manifest.get("source_commit"),
    }


def _validate_mask_reuse_generated_artifact(
    root: Path,
    *,
    project_root: Path,
    expected_model: str,
    expected_tokenizer: str,
    expected_seed: int,
    expected_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    public_path = root / "public_generated.jsonl"
    private_path = root / "private_generated_labels.jsonl"
    _require(manifest_path.is_file(), f"Missing generated manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    _require(manifest.get("model") == expected_model, "Generated model mismatch.")
    _require(
        manifest.get("tokenizer") == expected_tokenizer,
        "Generated tokenizer mismatch.",
    )
    _require(manifest.get("seed") == expected_seed, "Generated seed mismatch.")
    row_counts = manifest.get("row_counts", {})
    _require(
        row_counts.get("public_generated") == expected_rows
        and row_counts.get("private_generated_labels") == expected_rows,
        "Generated row-count mismatch.",
    )
    file_hashes = manifest.get("file_sha256", {})
    files = {
        "public_generated": public_path,
        "private_generated_labels": private_path,
    }
    for name, path in files.items():
        _require(path.is_file(), f"Missing generated artifact: {path}")
        _require(
            file_hashes.get(name) == file_sha256(path),
            f"Generated artifact hash mismatch: {path}",
        )

    public_rows = read_jsonl(public_path)
    private_rows = read_jsonl(private_path)
    _require(
        len(public_rows) == expected_rows and len(private_rows) == expected_rows,
        "Generated table length mismatch.",
    )
    _require(
        all(
            str(public.get("candidate_id", "")) == str(private.get("candidate_id", ""))
            for public, private in zip(public_rows, private_rows, strict=True)
        ),
        "Generated public/private candidate order differs.",
    )
    return (
        {
            "root": root.relative_to(project_root).as_posix(),
            "manifest_sha256": file_sha256(manifest_path),
            "file_sha256": {name: file_hashes[name] for name in files},
            "rows": expected_rows,
            "source_state": manifest.get("source_commit"),
        },
        public_rows,
        private_rows,
    )


def validate_lora_mask_reuse_source(
    *,
    project_root: Path,
    source_model_root: Path,
    expected_model: str,
    expected_tokenizer: str,
    expected_seed: int,
    expected_shadow_count: int,
    expected_group_size: int,
) -> dict[str, Any]:
    """Validate the baseline candidate masks and ordered generated rows."""
    candidate_root = source_model_root / "candidates"
    candidate_record = validate_lora_candidate_artifact(
        candidate_root,
        expected_model=expected_model,
        expected_tokenizer=expected_tokenizer,
        expected_seed=expected_seed,
        expected_shadow_count=expected_shadow_count,
    )
    generated_record, _, _ = _validate_mask_reuse_generated_artifact(
        source_model_root / "generated",
        project_root=project_root,
        expected_model=expected_model,
        expected_tokenizer=expected_tokenizer,
        expected_seed=expected_seed,
        expected_rows=len(_MASK_REUSE_GENERATED_GROUPS) * expected_group_size,
    )
    return {
        "model_root": source_model_root.relative_to(project_root).as_posix(),
        "candidate_artifact": {
            "manifest_sha256": candidate_record["manifest_sha256"],
            "file_sha256": candidate_record["file_sha256"],
            "rows": candidate_record["candidate_rows"],
            "shadow_count": candidate_record["source_shadow_count"],
            "source_state": candidate_record["source_state"],
        },
        "generated_artifact": generated_record,
    }


def inherit_lora_candidate_masks(
    *,
    project_root: Path,
    source_model_root: Path,
    destination_candidate_root: Path,
    expected_model: str,
    expected_tokenizer: str,
    expected_seed: int,
    expected_shadow_count: int,
    expected_group_size: int,
) -> dict[str, Any]:
    """Inherit baseline masks through exact gold and generated-row alignment."""
    source_record = validate_lora_mask_reuse_source(
        project_root=project_root,
        source_model_root=source_model_root,
        expected_model=expected_model,
        expected_tokenizer=expected_tokenizer,
        expected_seed=expected_seed,
        expected_shadow_count=expected_shadow_count,
        expected_group_size=expected_group_size,
    )
    source_candidate_root = source_model_root / "candidates"
    source_public = read_jsonl(source_candidate_root / "public_candidates.jsonl")
    source_private = read_jsonl(source_candidate_root / "private_labels.jsonl")
    with (source_candidate_root / "shadow_masks.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        source_masks = {
            str(row["candidate_id"]): [
                int(row[f"shadow_{index:02d}"])
                for index in range(expected_shadow_count)
            ]
            for row in reader
        }

    source_by_id: dict[str, tuple[dict[str, Any], dict[str, Any], list[int]]] = {}
    source_by_content: dict[str, tuple[dict[str, Any], dict[str, Any], list[int]]] = {}
    for public, private in zip(source_public, source_private, strict=True):
        candidate = str(public.get("candidate_id", ""))
        content_hash = str(public.get("content_sha256", ""))
        _require(
            candidate == str(private.get("candidate_id", "")),
            "Baseline candidate public/private order differs.",
        )
        _require(candidate in source_masks, f"Missing baseline mask: {candidate}")
        _require(
            content_hash not in source_by_content,
            f"Duplicate baseline candidate content: {content_hash}",
        )
        entry = (public, private, source_masks[candidate])
        source_by_id[candidate] = entry
        source_by_content[content_hash] = entry

    _, source_generated_public, source_generated_private = (
        _validate_mask_reuse_generated_artifact(
            source_model_root / "generated",
            project_root=project_root,
            expected_model=expected_model,
            expected_tokenizer=expected_tokenizer,
            expected_seed=expected_seed,
            expected_rows=len(_MASK_REUSE_GENERATED_GROUPS) * expected_group_size,
        )
    )
    destination_files = {
        "raw public candidates": destination_candidate_root
        / "raw_public_candidates.jsonl",
        "raw private provenance": destination_candidate_root
        / "raw_private_provenance.jsonl",
        "public candidates": destination_candidate_root / "public_candidates.jsonl",
        "evaluator mapping": destination_candidate_root / "evaluator_mapping.jsonl",
    }
    for name, path in destination_files.items():
        _require(path.is_file(), f"Missing destination {name}: {path}")
    raw_public = read_jsonl(destination_files["raw public candidates"])
    raw_private = read_jsonl(destination_files["raw private provenance"])
    canonical_public = read_jsonl(destination_files["public candidates"])
    evaluator_mapping = read_jsonl(destination_files["evaluator mapping"])

    expected_groups = [
        group
        for group in (*_MASK_REUSE_GOLD_GROUPS, *_MASK_REUSE_GENERATED_GROUPS)
        for _ in range(expected_group_size)
    ]
    expected_raw_rows = len(expected_groups)
    _require(
        len(raw_public) == expected_raw_rows
        and len(raw_private) == expected_raw_rows
        and len(evaluator_mapping) == expected_raw_rows,
        "Destination candidate row-count mismatch.",
    )
    _require(
        [str(row.get("private_group", "")) for row in raw_private] == expected_groups,
        "Destination candidate groups or order differ.",
    )

    expected_canonical: list[dict[str, Any]] = []
    canonical_by_id: dict[str, dict[str, Any]] = {}
    for raw_index, (public, private, mapping) in enumerate(
        zip(raw_public, raw_private, evaluator_mapping, strict=True)
    ):
        source_candidate_id = str(public.get("candidate_id", ""))
        _require(
            source_candidate_id == str(private.get("candidate_id", "")),
            "Destination raw public/private candidate order differs.",
        )
        content_hash = str(public.get("content_sha256", ""))
        _require(bool(content_hash), "Destination candidate content hash is missing.")
        canonical_id = f"candidate_{content_hash[:20]}"
        model_row = {
            "candidate_id": canonical_id,
            "prompt": str(public.get("prompt", "")),
            "completion": str(public.get("completion", "")),
            "content_sha256": content_hash,
        }
        previous = canonical_by_id.get(canonical_id)
        if previous is None:
            canonical_by_id[canonical_id] = model_row
            expected_canonical.append(model_row)
        else:
            _require(
                previous == model_row,
                f"Canonical destination candidate collision: {canonical_id}",
            )
        expected_mapping = {
            **private,
            "candidate_id": canonical_id,
            "source_candidate_id": source_candidate_id,
            "raw_row_index": raw_index,
        }
        _require(
            mapping == expected_mapping,
            f"Destination evaluator mapping differs at raw row {raw_index}.",
        )
    _require(
        canonical_public == expected_canonical,
        "Destination canonical candidate table differs from its raw rows.",
    )

    inherited_by_canonical: dict[str, list[int]] = {}
    gold_rows = len(_MASK_REUSE_GOLD_GROUPS) * expected_group_size
    for raw_index, (public, private) in enumerate(
        zip(raw_public, raw_private, strict=True)
    ):
        content_hash = str(public["content_sha256"])
        if raw_index < gold_rows:
            source_candidate_id = str(public["candidate_id"])
            source = source_by_id.get(source_candidate_id)
            _require(
                source is not None,
                f"Gold candidate is absent from the baseline: {source_candidate_id}",
            )
            _require(
                str(source[0].get("content_sha256", "")) == content_hash,
                f"Gold candidate content differs: {source_candidate_id}",
            )
            _require(
                str(source[1].get("private_group", ""))
                == str(private.get("private_group", ""))
                and str(source[1].get("source_id", ""))
                == str(private.get("source_id", "")),
                f"Gold candidate identity differs: {source_candidate_id}",
            )
        else:
            generated_index = raw_index - gold_rows
            source_generated = source_generated_public[generated_index]
            source_generated_label = source_generated_private[generated_index]
            _require(
                str(source_generated_label.get("private_group", ""))
                == str(private.get("private_group", ""))
                and str(source_generated_label.get("source_id", ""))
                == str(private.get("source_id", "")),
                f"Generated source identity differs at row {generated_index}.",
            )
            source_content_hash = str(source_generated.get("content_sha256", ""))
            source = source_by_content.get(source_content_hash)
            _require(
                source is not None,
                f"Generated baseline content is absent from candidates: "
                f"{source_content_hash}",
            )
        mask = source[2]
        canonical_id = f"candidate_{content_hash[:20]}"
        previous_mask = inherited_by_canonical.get(canonical_id)
        _require(
            previous_mask is None or previous_mask == mask,
            f"Canonical candidate inherits conflicting masks: {canonical_id}",
        )
        inherited_by_canonical[canonical_id] = mask

    canonical_ids = [str(row["candidate_id"]) for row in canonical_public]
    _require(
        set(inherited_by_canonical) == set(canonical_ids),
        "Inherited masks do not cover the canonical candidate table.",
    )
    destination_mask_path = destination_candidate_root / "shadow_masks.csv"
    fieldnames = [
        "candidate_id",
        *[f"shadow_{index:02d}" for index in range(expected_shadow_count)],
    ]
    with destination_mask_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in canonical_ids:
            mask = inherited_by_canonical[candidate]
            writer.writerow(
                {
                    "candidate_id": candidate,
                    **{
                        f"shadow_{index:02d}": value for index, value in enumerate(mask)
                    },
                }
            )
    included_counts = [
        sum(inherited_by_canonical[candidate][index] for candidate in canonical_ids)
        for index in range(expected_shadow_count)
    ]
    return {
        "mode": "baseline_mask_inheritance",
        "source": source_record,
        "mapping": {
            "gold": "source_candidate_id_with_content_and_identity_check",
            "generated": "raw_position_with_group_and_source_id_check",
            "canonicalization": "content_sha256_with_conflict_rejection",
        },
        "row_counts": {
            "raw_candidates": expected_raw_rows,
            "gold_candidates": gold_rows,
            "generated_candidates": len(source_generated_public),
            "canonical_candidates": len(canonical_ids),
            "mask_conflicts": 0,
        },
        "included_candidates_by_shadow": included_counts,
        "shadow_masks_sha256": file_sha256(destination_mask_path),
    }


def validate_split_artifact(
    root: Path, expected_manifest: dict[str, Any]
) -> dict[str, Any]:
    manifest_path = root / "split_manifest.json"
    _require(manifest_path.is_file(), f"Missing split manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    for key, expected_value in expected_manifest.items():
        _require(
            manifest.get(key) == expected_value,
            f"Split manifest field mismatch: {key}",
        )
    file_hashes = manifest.get("file_sha256")
    _require(
        isinstance(file_hashes, dict) and bool(file_hashes), "Missing split hashes."
    )
    for name, digest in file_hashes.items():
        path = root / str(name)
        _require(path.is_file(), f"Missing split file: {path}")
        _require(file_sha256(path) == digest, f"Split hash mismatch: {path}")
    return {
        "status": "reused",
        "path": str(root),
        "manifest_sha256": file_sha256(manifest_path),
        "file_sha256": dict(file_hashes),
        "validated_fields": sorted(expected_manifest),
    }


def validate_base_generation_artifact(
    root: Path,
    *,
    expected_model_run_id: str,
    expected_tokenizer_id: str,
    expected_source_ids: dict[str, list[str]],
    expected_generation_config: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file(), f"Missing generation manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    _require(
        manifest.get("generation_source", {}).get("model") == expected_model_run_id,
        "Base-generation model revision mismatch.",
    )
    expected_rows = sum(len(values) for values in expected_source_ids.values())
    row_counts = manifest.get("row_counts", {})
    _require(
        row_counts.get("public_generated") == expected_rows
        and row_counts.get("private_generated_labels") == expected_rows,
        "Base-generation row-count mismatch.",
    )

    artifact_files = {
        "public_generated": root / "public_generated.jsonl",
        "private_generated_labels": root / "private_generated_labels.jsonl",
        "metrics": root / "metrics.json",
        "resolved_config": root / "resolved_config.yaml",
    }
    file_hashes = manifest.get("file_sha256")
    _require(
        isinstance(file_hashes, dict) and set(artifact_files) <= set(file_hashes),
        "Base-generation manifest is missing file hashes.",
    )
    for name, path in artifact_files.items():
        _require(path.is_file(), f"Missing base-generation file: {path}")
        _require(
            file_sha256(path) == file_hashes[name],
            f"Base-generation hash mismatch: {path}",
        )

    public_rows = read_jsonl(artifact_files["public_generated"])
    private_rows = read_jsonl(artifact_files["private_generated_labels"])
    public_ids = [str(row.get("candidate_id", "")) for row in public_rows]
    private_ids = [str(row.get("candidate_id", "")) for row in private_rows]
    _require(
        len(public_ids) == expected_rows and len(set(public_ids)) == expected_rows,
        "Base-generation public candidate IDs are missing or duplicated.",
    )
    _require(public_ids == private_ids, "Base-generation public/private order differs.")
    _require(
        all(row.get("tokenizer_id") == expected_tokenizer_id for row in public_rows),
        "Base-generation tokenizer mismatch.",
    )
    _require(
        all(
            row.get("generation_config") == expected_generation_config
            for row in public_rows
        ),
        "Base-generation settings mismatch.",
    )

    actual_source_ids = {group: [] for group in expected_source_ids}
    for row in private_rows:
        group = str(row.get("private_group", ""))
        _require(group in actual_source_ids, f"Unexpected generation group: {group}")
        actual_source_ids[group].append(str(row.get("source_id", "")))
    _require(
        actual_source_ids == expected_source_ids,
        "Base-generation source IDs or order mismatch.",
    )
    return {
        "status": "reused",
        "path": str(root),
        "source_commit": manifest.get("source_commit"),
        "manifest_sha256": file_sha256(manifest_path),
        "file_sha256": {name: file_hashes[name] for name in artifact_files},
        "validated_fields": [
            "model_revision",
            "tokenizer",
            "generation_config",
            "row_counts",
            "ordered_source_ids",
            "file_sha256",
        ],
    }
