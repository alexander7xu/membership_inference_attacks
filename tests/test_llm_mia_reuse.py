from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.experiment import config_fingerprint
from src.llm_mia.data import (
    canonicalize_candidate_rows,
    file_sha256,
    read_json,
    read_jsonl,
    read_shadow_masks,
    write_json,
    write_jsonl,
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
from src.llm_mia.workflow import (
    _artifact_record_is_complete,
    _semantic_config_fingerprint,
    _validate_mask_reuse_candidate_lineage,
    artifact_manifest,
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


def _write_lora_expansion_fixture(root: Path) -> dict[str, object]:
    expected_config: dict[str, object] = {
        "runtime": {"seed": 42, "deterministic": False},
        "paths": {"data_dir": "data/squad_lora_rmia"},
        "tokenizer": {
            "max_length": 1024,
            "trust_remote_code": False,
            "preprocessing_workers": 8,
        },
        "shadow": {
            "count": 100,
            "smoke_count": 1,
            "train_size": 4,
            "inclusion_probability": 0.5,
            "seed_offset": 1000,
        },
    }
    source_config = {
        **expected_config,
        "tokenizer": {
            "max_length": 1024,
            "trust_remote_code": False,
        },
        "shadow": {**expected_config["shadow"], "count": 5},
    }
    target = root / "target" / "seed_42"
    (target / "adapter").mkdir(parents=True)
    (target / "adapter" / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (target / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")
    write_jsonl(target / "train_manifest.jsonl", [{"record_id": "row-0"}])
    write_json(target / "metrics.json", {"train/loss": 1.0})
    OmegaConf.save(source_config, target / "resolved_config.yaml")
    (target / "experiment.md").write_text("target\n", encoding="utf-8")

    for eval_name in ("base", "target_lora"):
        eval_root = root / "eval" / eval_name
        write_json(eval_root / "metrics.json", {"loss": 1.0})
        OmegaConf.save(expected_config, eval_root / "resolved_config.yaml")
        (eval_root / "experiment.md").write_text("eval\n", encoding="utf-8")

    generated = root / "generated"
    generated_files = {
        "public_generated": generated / "public_generated.jsonl",
        "private_generated_labels": generated / "private_generated_labels.jsonl",
        "metrics": generated / "metrics.json",
    }
    write_jsonl(generated_files["public_generated"], [{"candidate_id": "c0"}])
    write_jsonl(generated_files["private_generated_labels"], [{"candidate_id": "c0"}])
    write_json(generated_files["metrics"], {"rows": 1})
    fingerprint = config_fingerprint(expected_config)
    (generated / "experiment.md").write_text(
        f"Config SHA256: `{fingerprint}`\n", encoding="utf-8"
    )
    write_json(
        generated / "manifest.json",
        {
            "config_sha256": fingerprint,
            "file_sha256": {
                name: file_sha256(path) for name, path in generated_files.items()
            },
            "source_commit": {"commit": "abc123"},
        },
    )
    return expected_config


def test_lora_shadow_expansion_reuse_is_hash_verified_and_idempotent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "model"
    destination = tmp_path / "destination" / "model"
    expected_config = _write_lora_expansion_fixture(source)

    first = prepare_lora_shadow_expansion_inputs(
        project_root=tmp_path,
        source_model_root=source,
        destination_model_root=destination,
        expected_config=expected_config,
        expected_source_shadow_count=5,
        seed=42,
        target_eval_name="target_lora",
        force=False,
    )
    second = prepare_lora_shadow_expansion_inputs(
        project_root=tmp_path,
        source_model_root=source,
        destination_model_root=destination,
        expected_config=expected_config,
        expected_source_shadow_count=5,
        seed=42,
        target_eval_name="target_lora",
        force=False,
    )

    assert first == second
    assert first["destination_shadow_count"] == 100
    assert (destination / "target" / "seed_42" / "adapter").is_dir()
    assert (destination / "eval" / "target_lora" / "metrics.json").is_file()
    assert (destination / "generated" / "manifest.json").is_file()

    (destination / "generated" / "metrics.json").write_text(
        "changed\n", encoding="utf-8"
    )
    with pytest.raises(ReuseValidationError, match="differs from its source"):
        prepare_lora_shadow_expansion_inputs(
            project_root=tmp_path,
            source_model_root=source,
            destination_model_root=destination,
            expected_config=expected_config,
            expected_source_shadow_count=5,
            seed=42,
            target_eval_name="target_lora",
            force=False,
        )
    repaired = prepare_lora_shadow_expansion_inputs(
        project_root=tmp_path,
        source_model_root=source,
        destination_model_root=destination,
        expected_config=expected_config,
        expected_source_shadow_count=5,
        seed=42,
        target_eval_name="target_lora",
        force=True,
    )
    assert repaired == first


def test_lora_shadow_expansion_rejects_semantic_tokenizer_change(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "model"
    destination = tmp_path / "destination" / "model"
    expected_config = _write_lora_expansion_fixture(source)
    changed_config = deepcopy(expected_config)
    changed_config["tokenizer"]["max_length"] = 512

    with pytest.raises(ReuseValidationError, match="target config mismatch: tokenizer"):
        prepare_lora_shadow_expansion_inputs(
            project_root=tmp_path,
            source_model_root=source,
            destination_model_root=destination,
            expected_config=changed_config,
            expected_source_shadow_count=5,
            seed=42,
            target_eval_name="target_lora",
            force=False,
        )


def test_lora_shadow_expansion_rejects_unlinked_historical_generation_config(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "model"
    destination = tmp_path / "destination" / "model"
    expected_config = _write_lora_expansion_fixture(source)
    manifest_path = source / "generated" / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["config_sha256"] = "wrong"
    write_json(manifest_path, manifest)

    with pytest.raises(ReuseValidationError, match="Historical generated config"):
        prepare_lora_shadow_expansion_inputs(
            project_root=tmp_path,
            source_model_root=source,
            destination_model_root=destination,
            expected_config=expected_config,
            expected_source_shadow_count=5,
            seed=42,
            target_eval_name="target_lora",
            force=False,
        )


def _write_lora_candidate_fixture(root: Path) -> None:
    public_rows = [
        {
            "candidate_id": "candidate-0",
            "prompt": "prompt 0",
            "completion": " completion 0",
        },
        {
            "candidate_id": "candidate-1",
            "prompt": "prompt 1",
            "completion": " completion 1",
        },
    ]
    private_rows = [
        {"candidate_id": "candidate-0", "record_membership_label": 1},
        {"candidate_id": "candidate-1", "record_membership_label": 0},
    ]
    public_path = root / "public_candidates.jsonl"
    private_path = root / "private_labels.jsonl"
    masks_path = root / "shadow_masks.csv"
    write_jsonl(public_path, public_rows)
    write_jsonl(private_path, private_rows)
    with masks_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["candidate_id", *[f"shadow_{index:02d}" for index in range(5)]]
        )
        writer.writerow(["candidate-0", 1, 0, 1, 0, 1])
        writer.writerow(["candidate-1", 0, 1, 0, 1, 0])
    write_json(
        root / "manifest.json",
        {
            "model": "owner/model",
            "tokenizer": "owner/model",
            "seed": 42,
            "row_counts": {
                "public_candidates": 2,
                "private_labels": 2,
                "shadow_models": 5,
            },
            "file_sha256": {
                "public_candidates": file_sha256(public_path),
                "private_labels": file_sha256(private_path),
                "shadow_masks": file_sha256(masks_path),
            },
            "source_commit": {"commit": "abc123"},
        },
    )


def test_lora_candidate_reuse_validates_exact_tables_and_masks(tmp_path: Path) -> None:
    _write_lora_candidate_fixture(tmp_path)

    record = validate_lora_candidate_artifact(
        tmp_path,
        expected_model="owner/model",
        expected_tokenizer="owner/model",
        expected_seed=42,
        expected_shadow_count=5,
    )

    assert record["candidate_rows"] == 2
    assert record["source_shadow_count"] == 5


def test_lora_candidate_reuse_rejects_changed_private_labels(tmp_path: Path) -> None:
    _write_lora_candidate_fixture(tmp_path)
    (tmp_path / "private_labels.jsonl").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ReuseValidationError, match="hash mismatch"):
        validate_lora_candidate_artifact(
            tmp_path,
            expected_model="owner/model",
            expected_tokenizer="owner/model",
            expected_seed=42,
            expected_shadow_count=5,
        )


def test_semantic_config_fingerprint_ignores_only_workflow_controls() -> None:
    build = {
        "workflow": {"stage": "build_candidates", "profile": "formal", "force": False},
        "shadow": {"count": 100},
        "train": {"learning_rate": 1.0e-4},
    }
    attack = {
        **build,
        "workflow": {"stage": "attack", "profile": "formal", "force": True},
    }
    changed_training = {
        **attack,
        "train": {"learning_rate": 2.0e-4},
    }

    assert _semantic_config_fingerprint(build) == _semantic_config_fingerprint(attack)
    assert _semantic_config_fingerprint(build) != _semantic_config_fingerprint(
        changed_training
    )


def test_artifact_manifest_hashes_exact_file_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "src.llm_mia.workflow.collect_git_state",
        lambda _project_root: {"commit": "test"},
    )
    cfg = OmegaConf.create(
        {
            "runtime": {"seed": 42},
            "model": {"name_or_path": "owner/model"},
            "workflow": {
                "stage": "build_candidates",
                "profile": "formal",
                "force": False,
            },
        }
    )
    masks_path = tmp_path / "shadow_masks.csv"
    masks_path.write_bytes(b"candidate_id,shadow_00\r\ncandidate-0,1\r\n")
    manifest = artifact_manifest(
        cfg,
        tmp_path,
        {"shadow_masks": masks_path},
        row_counts={"shadow_models": 1},
    )
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)

    assert manifest["file_sha256"]["shadow_masks"] == file_sha256(masks_path)
    assert _artifact_record_is_complete(
        cfg,
        manifest_path,
        {"shadow_masks": masks_path},
        expected_row_counts={"shadow_models": 1},
    )


def test_more_shadows_config_changes_only_declared_experiment_fields() -> None:
    project_root = Path(__file__).resolve().parents[1]
    base = OmegaConf.to_container(
        OmegaConf.load(project_root / "conf" / "squad_lora_rmia.yaml"),
        resolve=False,
    )
    expanded = OmegaConf.to_container(
        OmegaConf.load(project_root / "conf" / "squad_lora_rmia_more_shadows.yaml"),
        resolve=False,
    )
    assert isinstance(base, dict)
    assert isinstance(expanded, dict)

    expanded["experiment"] = base["experiment"]
    expanded["paths"]["output_root"] = base["paths"]["output_root"]
    expanded["shadow"]["count"] = base["shadow"]["count"]
    expanded["wandb"]["tags"] = base["wandb"]["tags"]
    expanded.pop("reuse")

    assert expanded == base


def test_more_epochs_config_changes_only_declared_experiment_fields() -> None:
    project_root = Path(__file__).resolve().parents[1]
    base = OmegaConf.to_container(
        OmegaConf.load(project_root / "conf" / "squad_lora_rmia.yaml"),
        resolve=False,
    )
    extended = OmegaConf.to_container(
        OmegaConf.load(project_root / "conf" / "squad_lora_rmia_more_epochs.yaml"),
        resolve=False,
    )
    assert isinstance(base, dict)
    assert isinstance(extended, dict)
    assert extended["train"]["epochs"] == 10
    assert extended["train"]["target_eval_each_epoch"] is True
    assert extended["shadow"]["count"] == 5

    extended["experiment"] = base["experiment"]
    extended["paths"]["output_root"] = base["paths"]["output_root"]
    extended["train"]["epochs"] = base["train"]["epochs"]
    extended["train"].pop("target_eval_each_epoch")
    extended["wandb"]["tags"] = base["wandb"]["tags"]
    extended.pop("mask_reuse")

    assert extended == base


_MASK_REUSE_GROUPS = (
    "gold_squad_target_train",
    "gold_squad_validation",
    "gold_trivia_validation",
    "gen_from_squad_train",
    "gen_from_squad_validation",
    "gen_from_trivia_validation",
)
_MASK_PATTERNS = (
    (1, 0, 1, 0, 1),
    (0, 1, 0, 1, 0),
    (1, 1, 0, 0, 1),
    (0, 0, 1, 1, 0),
    (1, 0, 0, 1, 1),
    (0, 1, 1, 0, 0),
)


def _write_mask_csv(path: Path, candidate_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["candidate_id", *[f"shadow_{index:02d}" for index in range(5)]]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate_id, mask in zip(candidate_ids, _MASK_PATTERNS, strict=True):
            writer.writerow(
                {
                    "candidate_id": candidate_id,
                    **{
                        f"shadow_{index:02d}": value for index, value in enumerate(mask)
                    },
                }
            )


def _write_mask_reuse_fixture(
    tmp_path: Path,
    *,
    conflicting_canonical_mask: bool = False,
    repeated_generated_source_ids: bool = False,
) -> tuple[Path, Path]:
    source_model_root = tmp_path / "baseline" / "model"
    source_candidate_root = source_model_root / "candidates"
    destination_root = tmp_path / "destination" / "candidates"
    source_candidate_root.mkdir(parents=True)
    destination_root.mkdir(parents=True)

    source_public = []
    source_private = []
    for index, group in enumerate(_MASK_REUSE_GROUPS):
        candidate_id = f"baseline-{index}"
        content_hash = f"{index + 1:020x}" + "0" * 44
        source_id = (
            "shared-generated-source"
            if repeated_generated_source_ids and index >= 3
            else f"source-{index}"
        )
        source_public.append(
            {
                "candidate_id": candidate_id,
                "prompt": f"source prompt {index}",
                "completion": f" source completion {index}",
                "content_sha256": content_hash,
            }
        )
        source_private.append(
            {
                "candidate_id": candidate_id,
                "private_group": group,
                "source_id": source_id,
                "record_membership_label": int(index == 0),
            }
        )
    write_jsonl(source_candidate_root / "public_candidates.jsonl", source_public)
    write_jsonl(source_candidate_root / "private_labels.jsonl", source_private)
    _write_mask_csv(
        source_candidate_root / "shadow_masks.csv",
        [str(row["candidate_id"]) for row in source_public],
    )
    candidate_files = {
        "public_candidates": source_candidate_root / "public_candidates.jsonl",
        "private_labels": source_candidate_root / "private_labels.jsonl",
        "shadow_masks": source_candidate_root / "shadow_masks.csv",
    }
    write_json(
        source_candidate_root / "manifest.json",
        {
            "model": "owner/model",
            "tokenizer": "owner/model",
            "seed": 42,
            "row_counts": {
                "public_candidates": 6,
                "private_labels": 6,
                "shadow_models": 5,
            },
            "file_sha256": {
                name: file_sha256(path) for name, path in candidate_files.items()
            },
            "source_commit": {"commit": "baseline"},
        },
    )

    generated_root = source_model_root / "generated"
    generated_public = [dict(row) for row in source_public[3:]]
    generated_private = [dict(row) for row in source_private[3:]]
    write_jsonl(generated_root / "public_generated.jsonl", generated_public)
    write_jsonl(generated_root / "private_generated_labels.jsonl", generated_private)
    generated_files = {
        "public_generated": generated_root / "public_generated.jsonl",
        "private_generated_labels": generated_root / "private_generated_labels.jsonl",
    }
    write_json(
        generated_root / "manifest.json",
        {
            "model": "owner/model",
            "tokenizer": "owner/model",
            "seed": 42,
            "row_counts": {
                "public_generated": 3,
                "private_generated_labels": 3,
            },
            "file_sha256": {
                name: file_sha256(path) for name, path in generated_files.items()
            },
            "source_commit": {"commit": "baseline"},
        },
    )

    raw_public = []
    raw_private = []
    for index, group in enumerate(_MASK_REUSE_GROUPS):
        if index < 3:
            public = dict(source_public[index])
        else:
            content_index = (
                10 if conflicting_canonical_mask and index in (3, 4) else index + 10
            )
            public = {
                "candidate_id": f"current-{index}",
                "prompt": (
                    "shared prompt"
                    if conflicting_canonical_mask and index in (3, 4)
                    else f"current prompt {index}"
                ),
                "completion": (
                    " shared completion"
                    if conflicting_canonical_mask and index in (3, 4)
                    else f" current completion {index}"
                ),
                "content_sha256": f"{content_index:020x}" + "1" * 44,
            }
        private = {
            "candidate_id": public["candidate_id"],
            "private_group": group,
            "source_id": source_private[index]["source_id"],
            "record_membership_label": int(index == 0),
        }
        raw_public.append(public)
        raw_private.append(private)
    canonical_public, evaluator_mapping = canonicalize_candidate_rows(
        raw_public, raw_private
    )
    write_jsonl(destination_root / "raw_public_candidates.jsonl", raw_public)
    write_jsonl(destination_root / "raw_private_provenance.jsonl", raw_private)
    write_jsonl(destination_root / "public_candidates.jsonl", canonical_public)
    write_jsonl(destination_root / "evaluator_mapping.jsonl", evaluator_mapping)
    return source_model_root, destination_root


def _inherit_fixture_masks(
    tmp_path: Path, **fixture_kwargs: bool
) -> tuple[dict[str, object], Path, Path]:
    source_root, destination_root = _write_mask_reuse_fixture(
        tmp_path, **fixture_kwargs
    )
    record = inherit_lora_candidate_masks(
        project_root=tmp_path,
        source_model_root=source_root,
        destination_candidate_root=destination_root,
        expected_model="owner/model",
        expected_tokenizer="owner/model",
        expected_seed=42,
        expected_shadow_count=5,
        expected_group_size=1,
    )
    return record, source_root, destination_root


def test_mask_reuse_inherits_gold_and_positionally_aligned_generated_masks(
    tmp_path: Path,
) -> None:
    record, source_root, destination_root = _inherit_fixture_masks(
        tmp_path, repeated_generated_source_ids=True
    )
    masks = read_shadow_masks(destination_root / "shadow_masks.csv")
    gold_id = f"candidate_{1:020x}"

    assert masks[gold_id] == list(_MASK_PATTERNS[0])
    assert record["row_counts"] == {
        "raw_candidates": 6,
        "gold_candidates": 3,
        "generated_candidates": 3,
        "canonical_candidates": 6,
        "mask_conflicts": 0,
    }
    assert record["source"] == validate_lora_mask_reuse_source(
        project_root=tmp_path,
        source_model_root=source_root,
        expected_model="owner/model",
        expected_tokenizer="owner/model",
        expected_seed=42,
        expected_shadow_count=5,
        expected_group_size=1,
    )


def test_mask_reuse_final_validation_checks_bound_manifest(tmp_path: Path) -> None:
    record, _, destination_root = _inherit_fixture_masks(tmp_path)
    write_json(
        destination_root / "manifest.json",
        {
            "candidate_source_mode": "baseline_mask_inheritance",
            "mask_reuse": record,
        },
    )
    cfg = OmegaConf.create(
        {
            "runtime": {"seed": 42},
            "model": {"key": "model", "name_or_path": "owner/model"},
            "data": {"candidate_limit": 1},
            "mask_reuse": {
                "source_output_root": "baseline",
                "expected_source_shadow_count": 5,
            },
        }
    )
    public_rows = read_jsonl(destination_root / "public_candidates.jsonl")
    masks = read_shadow_masks(destination_root / "shadow_masks.csv")

    assert (
        _validate_mask_reuse_candidate_lineage(
            cfg,
            project_root=tmp_path,
            candidate_root=destination_root,
            public_rows=public_rows,
            masks=masks,
            expected_shadow_count=5,
        )
        == record
    )

    invalid_manifest = deepcopy(read_json(destination_root / "manifest.json"))
    invalid_manifest["mask_reuse"]["included_candidates_by_shadow"][0] += 1
    write_json(destination_root / "manifest.json", invalid_manifest)
    with pytest.raises(ValueError, match="not bound to the baseline masks"):
        _validate_mask_reuse_candidate_lineage(
            cfg,
            project_root=tmp_path,
            candidate_root=destination_root,
            public_rows=public_rows,
            masks=masks,
            expected_shadow_count=5,
        )


def test_mask_reuse_rejects_conflicting_canonical_masks(tmp_path: Path) -> None:
    with pytest.raises(ReuseValidationError, match="inherits conflicting masks"):
        _inherit_fixture_masks(tmp_path, conflicting_canonical_mask=True)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing", "Missing generated artifact"),
        ("hash", "Generated artifact hash mismatch"),
        ("identity", "Generated source identity differs"),
    ],
)
def test_mask_reuse_fails_closed_on_invalid_dependencies_or_identity(
    tmp_path: Path, failure: str, message: str
) -> None:
    source_root, destination_root = _write_mask_reuse_fixture(tmp_path)
    if failure == "missing":
        (source_root / "generated/private_generated_labels.jsonl").unlink()
    elif failure == "hash":
        (source_root / "generated/public_generated.jsonl").write_text(
            "changed\n", encoding="utf-8"
        )
    else:
        raw_private_path = destination_root / "raw_private_provenance.jsonl"
        evaluator_path = destination_root / "evaluator_mapping.jsonl"
        raw_private = read_jsonl(raw_private_path)
        evaluator = read_jsonl(evaluator_path)
        raw_private[3]["source_id"] = "different-source"
        evaluator[3]["source_id"] = "different-source"
        write_jsonl(raw_private_path, raw_private)
        write_jsonl(evaluator_path, evaluator)

    with pytest.raises(ReuseValidationError, match=message):
        inherit_lora_candidate_masks(
            project_root=tmp_path,
            source_model_root=source_root,
            destination_candidate_root=destination_root,
            expected_model="owner/model",
            expected_tokenizer="owner/model",
            expected_seed=42,
            expected_shadow_count=5,
            expected_group_size=1,
        )
