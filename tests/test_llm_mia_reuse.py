from __future__ import annotations

from pathlib import Path

import pytest

from src.llm_mia.data import file_sha256, write_json, write_jsonl
from src.llm_mia.reuse import (
    ReuseValidationError,
    validate_base_generation_artifact,
    validate_split_artifact,
)

GENERATION_CONFIG = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 64,
}


def test_split_reuse_requires_matching_fields_and_hashes(tmp_path: Path) -> None:
    split_path = tmp_path / "squad_target_train.jsonl"
    write_jsonl(split_path, [{"record_id": "row-0"}])
    expected = {
        "seed": 42,
        "split_rule": "stable",
        "counts": {"squad_target_train": 1},
    }
    write_json(
        tmp_path / "split_manifest.json",
        {
            **expected,
            "file_sha256": {split_path.name: file_sha256(split_path)},
        },
    )

    record = validate_split_artifact(tmp_path, expected)
    assert record["status"] == "reused"

    split_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ReuseValidationError, match="hash mismatch"):
        validate_split_artifact(tmp_path, expected)


def _write_base_generation_fixture(root: Path) -> dict[str, list[str]]:
    source_ids = {
        "gen_from_squad_train": ["train-0", "train-1"],
        "gen_from_squad_validation": ["validation-0", "validation-1"],
        "gen_from_trivia_validation": ["trivia-0", "trivia-1"],
    }
    public_rows = []
    private_rows = []
    for group, group_source_ids in source_ids.items():
        for index, source_id in enumerate(group_source_ids):
            candidate_id = f"{group}-{index}"
            public_rows.append(
                {
                    "candidate_id": candidate_id,
                    "prompt": "prompt",
                    "completion": " completion",
                    "content_sha256": f"hash-{candidate_id}",
                    "tokenizer_id": "owner/model",
                    "generation_config": GENERATION_CONFIG,
                }
            )
            private_rows.append(
                {
                    "candidate_id": candidate_id,
                    "private_group": group,
                    "source_id": source_id,
                }
            )

    files = {
        "public_generated": root / "public_generated.jsonl",
        "private_generated_labels": root / "private_generated_labels.jsonl",
        "metrics": root / "metrics.json",
        "resolved_config": root / "resolved_config.yaml",
    }
    write_jsonl(files["public_generated"], public_rows)
    write_jsonl(files["private_generated_labels"], private_rows)
    write_json(files["metrics"], {"metric": 1.0})
    files["resolved_config"].write_text("runtime:\n  seed: 42\n", encoding="utf-8")
    write_json(
        root / "manifest.json",
        {
            "generation_source": {"model": "owner/model@revision"},
            "row_counts": {
                "public_generated": 6,
                "private_generated_labels": 6,
            },
            "file_sha256": {name: file_sha256(path) for name, path in files.items()},
            "source_commit": {"commit": "abc123"},
        },
    )
    return source_ids


def test_base_generation_reuse_validates_revision_config_sources_and_hashes(
    tmp_path: Path,
) -> None:
    source_ids = _write_base_generation_fixture(tmp_path)

    record = validate_base_generation_artifact(
        tmp_path,
        expected_model_run_id="owner/model@revision",
        expected_tokenizer_id="owner/model",
        expected_source_ids=source_ids,
        expected_generation_config=GENERATION_CONFIG,
    )

    assert record["status"] == "reused"
    assert record["source_commit"] == {"commit": "abc123"}


def test_base_generation_reuse_rejects_wrong_ordered_source_ids(
    tmp_path: Path,
) -> None:
    source_ids = _write_base_generation_fixture(tmp_path)
    source_ids["gen_from_squad_train"] = list(
        reversed(source_ids["gen_from_squad_train"])
    )

    with pytest.raises(ReuseValidationError, match="source IDs or order"):
        validate_base_generation_artifact(
            tmp_path,
            expected_model_run_id="owner/model@revision",
            expected_tokenizer_id="owner/model",
            expected_source_ids=source_ids,
            expected_generation_config=GENERATION_CONFIG,
        )


def test_base_generation_reuse_rejects_generation_config_change(
    tmp_path: Path,
) -> None:
    source_ids = _write_base_generation_fixture(tmp_path)
    changed_config = {**GENERATION_CONFIG, "max_new_tokens": 32}

    with pytest.raises(ReuseValidationError, match="settings mismatch"):
        validate_base_generation_artifact(
            tmp_path,
            expected_model_run_id="owner/model@revision",
            expected_tokenizer_id="owner/model",
            expected_source_ids=source_ids,
            expected_generation_config=changed_config,
        )
