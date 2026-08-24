from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.llm_mia.analysis import paired_bootstrap_binary_metric_differences
from src.llm_mia.data import (
    canonicalize_candidate_rows,
    read_shadow_masks,
    write_explicit_shadow_masks,
)
from src.llm_mia.plotting import (
    TARGET_GENERATED_GROUP_COLORS,
    plot_group_feature_ecdf,
)
from src.llm_mia.workflow import (
    _inherit_target_generated_masks,
    _validate_unique_generation_source_prompts,
    attack_metrics,
)


def _source_rows() -> tuple[list[dict[str, object]], dict[str, list[int]]]:
    rows = [
        {
            "candidate_id": "member",
            "private_group": "gold_squad_target_train",
            "source_id": "member-source",
            "record_membership_label": 1,
        },
        {
            "candidate_id": "gold-00",
            "private_group": "gold_squad_validation_00",
            "source_id": "validation-00",
            "record_membership_label": 0,
        },
        {
            "candidate_id": "gold-01",
            "private_group": "gold_squad_validation_01",
            "source_id": "validation-01",
            "record_membership_label": 0,
        },
    ]
    masks = {
        "member": [1, 0, 1, 0, 0],
        "gold-00": [0, 1, 0, 1, 0],
        "gold-01": [1, 0, 0, 0, 1],
    }
    return rows, masks


def test_target_generated_masks_inherit_by_member_id_and_source_identity() -> None:
    source, masks = _source_rows()
    destination = [
        dict(source[0]),
        {
            "candidate_id": "generated-00",
            "private_group": "target_gen_squad_validation_00",
            "source_id": "validation-00",
            "source_candidate_id": "gold-00",
            "record_membership_label": 0,
        },
        {
            "candidate_id": "generated-01",
            "private_group": "target_gen_squad_validation_01",
            "source_id": "validation-01",
            "source_candidate_id": "gold-01",
            "record_membership_label": 0,
        },
    ]

    inherited = _inherit_target_generated_masks(
        destination, source, masks, group_count=2
    )

    assert inherited == {
        "member": masks["member"],
        "generated-00": masks["gold-00"],
        "generated-01": masks["gold-01"],
    }


def test_target_generated_canonicalization_preserves_gold_lineage() -> None:
    source, masks = _source_rows()
    public = [
        {
            "candidate_id": "generated-00",
            "prompt": "Question: generated?\nAnswer:",
            "completion": " generated",
            "content_sha256": "generated-content",
        }
    ]
    private = [
        {
            "candidate_id": "generated-00",
            "private_group": "target_gen_squad_validation_00",
            "source_id": "validation-00",
            "source_candidate_id": "gold-00",
            "record_membership_label": 0,
        }
    ]

    _, mapping = canonicalize_candidate_rows(
        public, private, preserve_source_candidate_ids=True
    )
    inherited = _inherit_target_generated_masks(mapping, source, masks, group_count=2)

    assert mapping[0]["source_candidate_id"] == "gold-00"
    assert inherited == {"generated-00": masks["gold-00"]}


@pytest.mark.parametrize(
    "mutation",
    [
        {"source_id": "missing"},
        {"source_candidate_id": "wrong"},
        {"private_group": "target_gen_squad_validation_02"},
    ],
)
def test_target_generated_masks_fail_closed_on_identity_mismatch(
    mutation: dict[str, str],
) -> None:
    source, masks = _source_rows()
    row = {
        "candidate_id": "generated-00",
        "private_group": "target_gen_squad_validation_00",
        "source_id": "validation-00",
        "source_candidate_id": "gold-00",
        "record_membership_label": 0,
        **mutation,
    }

    with pytest.raises(ValueError):
        _inherit_target_generated_masks(
            [dict(source[0]), row], source, masks, group_count=2
        )


def test_target_generated_masks_reject_duplicate_source_identity() -> None:
    source, masks = _source_rows()
    source.append({**source[-1], "candidate_id": "duplicate-gold"})
    masks["duplicate-gold"] = [1, 0, 1, 0, 0]

    with pytest.raises(ValueError, match="duplicated"):
        _inherit_target_generated_masks([], source, masks, group_count=2)


def test_explicit_shadow_masks_round_trip_and_reject_invalid_rows(
    tmp_path: Path,
) -> None:
    output = tmp_path / "masks.csv"
    masks = {"a": [1, 0, 1], "b": [0, 1, 0]}
    write_explicit_shadow_masks(output, masks)
    assert read_shadow_masks(output) == masks

    with pytest.raises(ValueError):
        write_explicit_shadow_masks(output, {"bad": [1, 1, 1]})


def test_paired_bootstrap_uses_shared_pairs_and_reports_direction() -> None:
    control_positive = [0.7, 0.8, 0.9, 1.0]
    control_negative = [0.1, 0.2, 0.3, 0.4]
    unchanged = paired_bootstrap_binary_metric_differences(
        control_positive,
        control_negative,
        control_positive,
        control_negative,
        fpr_thresholds=[0.05],
        samples=50,
        seed=42,
    )
    assert unchanged["auc_difference"] == 0.0
    assert unchanged["auc_difference_ci_lower"] == 0.0
    assert unchanged["auc_difference_ci_upper"] == 0.0

    degraded = paired_bootstrap_binary_metric_differences(
        [0.1, 0.2, 0.3, 0.4],
        [0.7, 0.8, 0.9, 1.0],
        control_positive,
        control_negative,
        fpr_thresholds=[0.05],
        samples=50,
        seed=42,
    )
    assert degraded["auc_difference"] < 0.0


def test_target_generated_attack_metrics_include_slices_and_pooled() -> None:
    scores = [{"candidate_id": "member", "online_rmia_score": 0.9}]
    mapping = [
        {
            "candidate_id": "member",
            "private_group": "gold_squad_target_train",
            "record_membership_label": 1,
        }
    ]
    for group in range(2):
        for item, score in enumerate((0.1, 0.2)):
            candidate = f"generated-{group}-{item}"
            scores.append(
                {"candidate_id": candidate, "online_rmia_score": score + group / 10}
            )
            mapping.append(
                {
                    "candidate_id": candidate,
                    "private_group": f"target_gen_squad_validation_{group:02d}",
                    "record_membership_label": 0,
                }
            )

    metrics = attack_metrics(
        scores,
        mapping,
        calibration_scores=[0.0, 0.05],
        fpr_thresholds=[0.01, 0.05],
        bootstrap_samples=10,
        seed=42,
        include_point_tpr=True,
    )

    assert metrics["auc_gold_train_vs_target_gen_squad_validation_00"] == 1.0
    assert metrics["auc_gold_train_vs_target_gen_squad_validation_01"] == 1.0
    assert metrics["auc_gold_train_vs_target_gen_squad_validation_pooled"] == 1.0
    assert (
        "tpr_at_fpr_0.05_ci_upper_gold_train_vs_"
        "target_gen_squad_validation_pooled" in metrics
    )


def test_target_generated_plot_supports_all_groups(tmp_path: Path) -> None:
    group_order = tuple(TARGET_GENERATED_GROUP_COLORS)
    rows = [
        {
            "candidate_id": f"candidate-{index}",
            "private_group": group,
            "relative_log_likelihood": float(index),
        }
        for index, group in enumerate(group_order)
    ]
    output = tmp_path / "target-generated-ecdf.png"

    plot_group_feature_ecdf(
        rows,
        feature="relative_log_likelihood",
        model_display_name="test-model",
        output_path=output,
        dpi=72,
        group_order=group_order,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_target_generated_config_and_runners_preserve_scientific_settings() -> None:
    config = OmegaConf.load("conf/squad_lora_rmia_target_generated_ablation.yaml")
    assert config.candidate_suite.mode == "target_generated_squad_validation"
    assert config.candidate_suite.group_size == 1024
    assert config.candidate_suite.nonmember_group_count == 5
    assert config.reuse.mode == "target_eval_only"
    assert (
        config.reuse.source_output_root == "outputs/squad_lora_rmia_iid_ablation/formal"
    )
    assert (
        config.control.source_output_root
        == "outputs/squad_lora_rmia_iid_ablation/formal"
    )
    assert config.shadow.count == 5
    assert config.shadow.train_size == 43799
    assert config.train.epochs == 1
    assert config.train.per_device_train_batch_size == 16
    assert config.train.gradient_accumulation_steps == 2
    assert config.precision.bf16 is True
    assert config.generation.do_sample is False
    assert config.generation.temperature == 0.0
    assert config.generation.max_new_tokens == 64
    assert config.attack.method == "online_rmia"
    assert config.attack.gamma == 0.5
    assert config.attack.bootstrap_samples == 2000

    runner = Path(
        "scripts/run_squad_lora_rmia_target_generated_ablation_formal.sh"
    ).read_text(encoding="utf-8")
    expected_order = [
        "prepare_shadow_reuse",
        "generate",
        "build_candidates",
        "train_shadows",
        "attack",
        "compare_gold_iid",
        "plot_rmia_feature",
        "validate",
    ]
    stage_block = runner.split("for stage in \\\n", maxsplit=1)[1].split(
        "; do", maxsplit=1
    )[0]
    assert [stage_block.index(stage) for stage in expected_order] == sorted(
        stage_block.index(stage) for stage in expected_order
    )
    assert "attack_control" not in runner
    assert "control/gold_iid/comparison.json" in runner

    submitter = Path(
        "scripts/submit_squad_lora_rmia_target_generated_ablation.sh"
    ).read_text(encoding="utf-8")
    assert 'partition="xe8545"' in submitter
    assert 'partition="tmp"' in submitter
    assert "--parsable" in submitter
    assert (
        "exec bash ./scripts/run_squad_lora_rmia_target_generated_ablation_formal.sh "
        "$model" in submitter
    )


def test_target_generation_rejects_duplicate_source_prompts_before_loading_model() -> (
    None
):
    public = {
        "a": {"prompt": "same prompt"},
        "b": {"prompt": "same prompt"},
    }
    private = [
        {
            "candidate_id": candidate,
            "private_group": f"gold_squad_validation_0{index}",
        }
        for index, candidate in enumerate(("a", "b"))
    ]

    with pytest.raises(ValueError, match="globally unique"):
        _validate_unique_generation_source_prompts(public, private, expected_rows=2)


def test_prompt_unique_target_generated_config_and_runners() -> None:
    config = OmegaConf.load(
        "conf/squad_lora_rmia_target_generated_prompt_unique_ablation.yaml"
    )
    source_root = "outputs/squad_lora_rmia_iid_prompt_unique_control/formal"
    assert config.paths.output_root == (
        "outputs/squad_lora_rmia_target_generated_prompt_unique_ablation"
    )
    assert config.reuse.source_output_root == source_root
    assert config.control.source_output_root == source_root
    assert config.shadow.count == 5
    assert config.train.epochs == 1
    assert config.attack.method == "online_rmia"

    runner = Path(
        "scripts/run_squad_lora_rmia_target_generated_prompt_unique_ablation_formal.sh"
    ).read_text(encoding="utf-8")
    assert "squad_lora_rmia_target_generated_prompt_unique_ablation" in runner
    submitter = Path(
        "scripts/submit_squad_lora_rmia_target_generated_prompt_unique_ablation.sh"
    ).read_text(encoding="utf-8")
    assert 'dependency_args+=(--dependency="afterok:$AFTEROK_JOB_ID")' in submitter
    assert (
        "run_squad_lora_rmia_target_generated_prompt_unique_ablation_formal.sh"
        in submitter
    )
