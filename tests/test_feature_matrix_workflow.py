from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.llm_mia.data import (
    QARecord,
    file_sha256,
    record_to_json,
    write_json,
    write_jsonl,
)
from src.llm_mia.feature_matrix_workflow import (
    _load_verified_population,
    _tree_file_sha256,
    _validate_candidate_source,
    _validate_checkpoint_root,
)


def _config():
    return OmegaConf.create(
        {
            "analysis": {"reference_shadow_count": 5},
            "model": {"name_or_path": "test/model"},
            "runtime": {"seed": 42},
            "report": {"resolved_config_filename": "resolved_config.yaml"},
        }
    )


def _write_run(run: Path, epochs: int) -> None:
    adapter = run / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (run / "resolved_config.yaml").write_text(
        f"train:\n  epochs: {epochs}\n", encoding="utf-8"
    )


def _checkpoint_root(tmp_path: Path, epochs: int = 1) -> Path:
    root = tmp_path / "setting"
    _write_run(root / "target" / "seed_42", epochs)
    shadow_hashes = {}
    for index in range(5):
        run = root / "shadows" / f"shadow_{index:02d}"
        _write_run(run, epochs)
        shadow_hashes[f"shadow_{index:02d}"] = _tree_file_sha256(run)
    write_json(
        root / "completion_manifest.json",
        {
            "model": "test/model",
            "shadow_models": 5,
            "shadow_artifact_sha256": shadow_hashes,
        },
    )
    (root / "_SUCCESS").write_text("verified\n", encoding="utf-8")
    return root


def test_checkpoint_validation_accepts_manifest_bound_tree(tmp_path: Path) -> None:
    root = _checkpoint_root(tmp_path)

    lineage = _validate_checkpoint_root(_config(), root=root, expected_epochs=1)

    assert lineage["expected_epochs"] == 1
    assert len(lineage["shadow_checkpoints"]) == 5


def test_checkpoint_validation_rejects_missing_adapter(tmp_path: Path) -> None:
    root = _checkpoint_root(tmp_path)
    (root / "shadows" / "shadow_00" / "adapter" / "adapter_model.safetensors").unlink()

    with pytest.raises(FileNotFoundError, match="Incomplete checkpoint run"):
        _validate_checkpoint_root(_config(), root=root, expected_epochs=1)


def test_checkpoint_validation_rejects_epoch_mismatch(tmp_path: Path) -> None:
    root = _checkpoint_root(tmp_path)

    with pytest.raises(ValueError, match="Expected 10 epochs"):
        _validate_checkpoint_root(_config(), root=root, expected_epochs=10)


def test_checkpoint_validation_rejects_hash_mismatch(tmp_path: Path) -> None:
    root = _checkpoint_root(tmp_path)
    adapter = root / "shadows" / "shadow_00" / "adapter" / "adapter_model.safetensors"
    adapter.write_bytes(b"changed")

    with pytest.raises(ValueError, match="shadow hashes differ"):
        _validate_checkpoint_root(_config(), root=root, expected_epochs=1)


def _candidate_source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    candidate = root / "candidates"
    generated = root / "generated"
    base = generated / "base"
    for directory in (candidate, generated, base):
        directory.mkdir(parents=True, exist_ok=True)
    file_groups = (
        (
            candidate / "manifest.json",
            {
                "public_candidates": candidate / "public_candidates.jsonl",
                "private_labels": candidate / "private_labels.jsonl",
                "shadow_masks": candidate / "shadow_masks.csv",
            },
        ),
        (
            generated / "manifest.json",
            {
                "public_generated": generated / "public_generated.jsonl",
                "private_generated_labels": generated
                / "private_generated_labels.jsonl",
                "metrics": generated / "metrics.json",
            },
        ),
        (
            base / "manifest.json",
            {
                "public_generated": base / "public_generated.jsonl",
                "private_generated_labels": base / "private_generated_labels.jsonl",
                "metrics": base / "metrics.json",
                "resolved_config": base / "resolved_config.yaml",
            },
        ),
    )
    for manifest, files in file_groups:
        for path in files.values():
            path.write_text("{}\n", encoding="utf-8")
        write_json(
            manifest,
            {"file_sha256": {name: file_sha256(path) for name, path in files.items()}},
        )
    return root


def test_candidate_source_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    root = _candidate_source(tmp_path)
    (root / "generated" / "base" / "public_generated.jsonl").write_text(
        "changed\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Source manifest hash mismatch"):
        _validate_candidate_source(root)


def test_verified_population_loads_without_training_data_config(tmp_path: Path) -> None:
    path = tmp_path / "population.jsonl"
    record = QARecord(
        record_id="population-0",
        dataset="squad",
        split="validation",
        prompt="Question\n",
        completion=" answer",
        answers=("answer",),
    )
    write_jsonl(path, [record_to_json(record)])

    loaded = _load_verified_population(
        path, expected_sha256=file_sha256(path), expected_rows=1
    )

    assert loaded == [record]


def test_verified_population_rejects_hash_or_row_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "population.jsonl"
    record = QARecord(
        record_id="population-0",
        dataset="squad",
        split="validation",
        prompt="Question\n",
        completion=" answer",
        answers=("answer",),
    )
    write_jsonl(path, [record_to_json(record)])

    with pytest.raises(ValueError, match="SHA256 differs"):
        _load_verified_population(path, expected_sha256="0" * 64, expected_rows=1)
    with pytest.raises(ValueError, match="expected 2"):
        _load_verified_population(
            path, expected_sha256=file_sha256(path), expected_rows=2
        )
