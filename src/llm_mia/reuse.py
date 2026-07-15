from __future__ import annotations

from pathlib import Path
from typing import Any

from src.llm_mia.data import file_sha256, read_json, read_jsonl


class ReuseValidationError(ValueError):
    """Raised when a prior experiment artifact cannot be reused safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReuseValidationError(message)


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
