from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.llm_mia.analysis import (
    fixed_variance_lira_parameters,
    online_lira_fixed_variance_score,
)
from src.llm_mia.data import file_sha256, read_json, write_json, write_jsonl
from src.llm_mia.workflow import (
    _build_reused_shadow_expansion_candidates,
    attack_metrics,
)


def test_fixed_variance_lira_matches_hand_calculation() -> None:
    result = fixed_variance_lira_parameters(
        [[1.0, 3.0, 2.0, 4.0], [2.0, 4.0, 1.0, 5.0]],
        [[1, 1, 0, 0], [1, 1, 0, 0]],
    )

    in_means, out_means, in_variance, out_variance, in_df, out_df = result
    assert in_means == [2.0, 3.0]
    assert out_means == [3.0, 3.0]
    assert in_variance == pytest.approx(2.0)
    assert out_variance == pytest.approx(5.0)
    assert (in_df, out_df) == (2, 2)

    expected = -0.5 * (
        math.log(2.0 * math.pi * 2.0) + (2.25 - 2.0) ** 2 / 2.0
    ) + 0.5 * (math.log(2.0 * math.pi * 5.0) + (2.25 - 3.0) ** 2 / 5.0)
    assert online_lira_fixed_variance_score(2.25, 2.0, 3.0, 2.0, 5.0) == pytest.approx(
        expected
    )


def test_fixed_variance_lira_score_direction() -> None:
    near_in = online_lira_fixed_variance_score(1.1, 1.0, 4.0, 1.0, 1.0)
    near_out = online_lira_fixed_variance_score(3.9, 1.0, 4.0, 1.0, 1.0)
    assert near_in > 0
    assert near_out < 0


def test_fixed_variance_lira_is_permutation_invariant_and_allows_singletons() -> None:
    scores = [[1.0, 2.0, 4.0], [2.0, 4.0, 1.0]]
    masks = [[1, 0, 0], [1, 1, 0]]
    original = fixed_variance_lira_parameters(scores, masks)
    candidate_permuted = fixed_variance_lira_parameters(
        list(reversed(scores)), list(reversed(masks))
    )
    shadow_permuted = fixed_variance_lira_parameters(
        [[row[index] for index in (2, 0, 1)] for row in scores],
        [[row[index] for index in (2, 0, 1)] for row in masks],
    )

    assert candidate_permuted[:2] == tuple(
        list(reversed(values)) for values in original[:2]
    )
    assert candidate_permuted[2:] == pytest.approx(original[2:])
    assert shadow_permuted == pytest.approx(original)


@pytest.mark.parametrize(
    ("scores", "masks", "message"),
    [
        ([[1.0, 2.0]], [[1, 1]], "at least one IN and one OUT"),
        ([[1.0, 2.0]], [[1, 2]], "only zero or one"),
        ([[1.0, 1.0], [2.0, 2.0]], [[1, 0], [1, 0]], "freedom"),
        (
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[1, 1, 0], [1, 0, 0]],
            "variances must be finite and positive",
        ),
    ],
)
def test_fixed_variance_lira_fails_closed(
    scores: list[list[float]], masks: list[list[int]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fixed_variance_lira_parameters(scores, masks)


def test_lira_metrics_have_no_population_calibration() -> None:
    score_rows = [
        {"candidate_id": "member-0", "online_lira_log_ratio": 3.0},
        {"candidate_id": "member-1", "online_lira_log_ratio": 2.0},
        {"candidate_id": "nonmember-0", "online_lira_log_ratio": -1.0},
        {"candidate_id": "nonmember-1", "online_lira_log_ratio": -2.0},
    ]
    private_rows = [
        {
            "candidate_id": "member-0",
            "record_membership_label": 1,
            "private_group": "gold_squad_target_train",
        },
        {
            "candidate_id": "member-1",
            "record_membership_label": 1,
            "private_group": "gold_squad_target_train",
        },
        {
            "candidate_id": "nonmember-0",
            "record_membership_label": 0,
            "private_group": "gold_squad_validation",
        },
        {
            "candidate_id": "nonmember-1",
            "record_membership_label": 0,
            "private_group": "gold_squad_validation",
        },
    ]

    metrics = attack_metrics(
        score_rows,
        private_rows,
        score_field="online_lira_log_ratio",
        calibration_scores=None,
        fpr_thresholds=[0.01, 0.05],
        bootstrap_samples=20,
        seed=42,
        include_point_tpr=True,
    )

    assert metrics["auc_gold_train_vs_gold_squad_validation"] == 1.0
    assert "population_calibration_examples" not in metrics
    assert "tpr_at_fpr_0.01_gold_train_vs_gold_squad_validation" in metrics


def _write_source_candidates(root: Path) -> bytes:
    root.mkdir(parents=True)
    public_rows = [
        {"candidate_id": "candidate-0", "prompt": "p0", "completion": " c0"},
        {"candidate_id": "candidate-1", "prompt": "p1", "completion": " c1"},
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
        writer = csv.writer(handle, lineterminator="\r\n")
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
    return masks_path.read_bytes()


def _exact_reuse_config(shadow_count: int = 5):
    return OmegaConf.create(
        {
            "runtime": {"seed": 42},
            "model": {"key": "model", "name_or_path": "owner/model"},
            "workflow": {
                "stage": "build_candidates",
                "profile": "formal",
                "force": False,
            },
            "paths": {"output_root": "outputs/squad_lora_lira"},
            "reuse": {
                "source_output_root": "outputs/squad_lora_rmia/formal",
                "expected_source_shadow_count": 5,
                "mask_mode": "exact_source",
            },
            "shadow": {
                "count": shadow_count,
                "smoke_count": 5,
                "seed_offset": 1000,
                "inclusion_probability": 0.5,
            },
        }
    )


def test_exact_source_reuse_copies_crlf_mask_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "outputs/squad_lora_rmia/formal/model/candidates"
    source_bytes = _write_source_candidates(source)
    monkeypatch.setattr(
        "src.llm_mia.workflow.collect_git_state", lambda _root: {"commit": "test"}
    )

    _build_reused_shadow_expansion_candidates(
        _exact_reuse_config(), project_root=tmp_path
    )

    destination = tmp_path / "outputs/squad_lora_lira/formal/model/candidates"
    assert (destination / "shadow_masks.csv").read_bytes() == source_bytes
    assert read_json(destination / "manifest.json")["candidate_source_mode"] == (
        "exact_source_tables_exact_masks"
    )


def test_exact_source_reuse_requires_equal_shadow_counts(tmp_path: Path) -> None:
    source = tmp_path / "outputs/squad_lora_rmia/formal/model/candidates"
    _write_source_candidates(source)

    with pytest.raises(ValueError, match="equal source and destination"):
        _build_reused_shadow_expansion_candidates(
            _exact_reuse_config(shadow_count=6), project_root=tmp_path
        )


def test_lira_config_changes_only_declared_fields() -> None:
    project_root = Path(__file__).resolve().parents[1]
    base = OmegaConf.to_container(
        OmegaConf.load(project_root / "conf/squad_lora_rmia.yaml"), resolve=False
    )
    lira = OmegaConf.to_container(
        OmegaConf.load(project_root / "conf/squad_lora_lira.yaml"), resolve=False
    )
    assert isinstance(base, dict)
    assert isinstance(lira, dict)

    lira["experiment"] = base["experiment"]
    lira["paths"]["output_root"] = base["paths"]["output_root"]
    lira["shadow"]["smoke_count"] = base["shadow"]["smoke_count"]
    lira["attack"] = base["attack"]
    lira["analysis"]["feature"] = base["analysis"]["feature"]
    lira["wandb"]["tags"] = base["wandb"]["tags"]
    lira.pop("reuse")

    assert lira == base


def test_lira_submitter_routes_models_and_propagates_exit_status() -> None:
    script = Path("scripts/submit_squad_lora_lira.sh").read_text(encoding="utf-8")

    assert "pythia-full" in script
    assert "olmo-full" in script
    assert 'partition="xe8545"' in script
    assert 'partition="tmp"' in script
    assert "run_squad_lora_lira_formal.sh $model" in script
    assert "exec bash" in script
    assert "--wrap=\"exec bash -lc '$job_command'\"" in script
    assert "mysubmit" not in script
