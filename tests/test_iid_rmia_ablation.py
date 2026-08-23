from pathlib import Path

from omegaconf import OmegaConf

from src.llm_mia.data import QARecord, build_squad_validation_iid_groups
from src.llm_mia.plotting import IID_GROUP_COLORS, plot_group_feature_ecdf
from src.llm_mia.workflow import attack_metrics


def _unique_records(prefix: str, count: int, *, split: str) -> list[QARecord]:
    return [
        QARecord(
            record_id=f"{prefix}-{index}",
            dataset="squad",
            split=split,
            prompt=f"Context:\n{prefix} context {index}\n\nQuestion:\nq{index}\n\nAnswer:\n",
            completion=f" answer {prefix} {index}",
            answers=(f"answer {prefix} {index}",),
        )
        for index in range(count)
    ]


def test_iid_validation_groups_are_deterministic_exact_and_disjoint() -> None:
    validation = _unique_records("validation", 16, split="validation")
    baseline = validation[:2]
    population = validation[2:4]
    target = _unique_records("target", 4, split="train")

    first = build_squad_validation_iid_groups(
        validation,
        baseline_candidates=baseline,
        population=population,
        target_train=target,
        seed=42,
        group_size=2,
        group_count=3,
    )
    second = build_squad_validation_iid_groups(
        validation,
        baseline_candidates=baseline,
        population=population,
        target_train=target,
        seed=42,
        group_size=2,
        group_count=3,
    )

    assert first == second
    assert first["gold_squad_validation_00"] == baseline
    assert {name: len(rows) for name, rows in first.items()} == {
        "gold_squad_validation_00": 2,
        "gold_squad_validation_01": 2,
        "gold_squad_validation_02": 2,
    }
    selected = [record for rows in first.values() for record in rows]
    assert len({record.record_id for record in selected}) == 6
    assert len({record.content_sha256 for record in selected}) == 6
    forbidden = {record.content_sha256 for record in [*population, *target]}
    assert not ({record.content_sha256 for record in selected} & forbidden)


def test_iid_validation_groups_reject_nonunique_baseline() -> None:
    validation = _unique_records("validation", 8, split="validation")
    duplicate = [validation[0], validation[0]]

    try:
        build_squad_validation_iid_groups(
            validation,
            baseline_candidates=duplicate,
            population=validation[2:4],
            target_train=_unique_records("target", 2, split="train"),
            seed=42,
            group_size=2,
            group_count=2,
        )
    except ValueError as error:
        assert "not unique" in str(error)
    else:
        raise AssertionError("A duplicate baseline IID group must be rejected.")


def test_iid_attack_metrics_include_slices_and_pooled_tpr() -> None:
    scores = [
        {"candidate_id": "member", "online_rmia_score": 0.9},
        {"candidate_id": "negative-00-a", "online_rmia_score": 0.1},
        {"candidate_id": "negative-00-b", "online_rmia_score": 0.2},
        {"candidate_id": "negative-01-a", "online_rmia_score": 0.3},
        {"candidate_id": "negative-01-b", "online_rmia_score": 0.4},
    ]
    mapping = [
        {
            "candidate_id": "member",
            "private_group": "gold_squad_target_train",
            "record_membership_label": 1,
        },
        *[
            {
                "candidate_id": f"negative-{group:02d}-{suffix}",
                "private_group": f"gold_squad_validation_{group:02d}",
                "record_membership_label": 0,
            }
            for group in range(2)
            for suffix in ("a", "b")
        ],
    ]

    metrics = attack_metrics(
        scores,
        mapping,
        calibration_scores=[0.0, 0.05],
        fpr_thresholds=[0.01, 0.05],
        bootstrap_samples=10,
        seed=42,
        include_point_tpr=True,
    )

    assert metrics["auc_gold_train_vs_gold_squad_validation_00"] == 1.0
    assert metrics["auc_gold_train_vs_gold_squad_validation_01"] == 1.0
    assert metrics["auc_gold_train_vs_gold_squad_validation_pooled"] == 1.0
    assert metrics["tpr_at_fpr_0.01_gold_train_vs_gold_squad_validation_pooled"] == 1.0
    assert (
        "tpr_at_fpr_0.05_ci_lower_gold_train_vs_gold_squad_validation_pooled" in metrics
    )


def test_iid_plot_supports_all_dynamic_groups(tmp_path: Path) -> None:
    group_order = tuple(IID_GROUP_COLORS)
    rows = [
        {
            "candidate_id": f"candidate-{index}",
            "private_group": group,
            "relative_log_likelihood": float(index),
        }
        for index, group in enumerate(group_order)
    ]
    output = tmp_path / "iid-ecdf.png"

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


def test_iid_config_and_runners_preserve_scientific_settings() -> None:
    config = OmegaConf.load("conf/squad_lora_rmia_iid_ablation.yaml")
    assert config.candidate_suite.mode == "squad_validation_iid"
    assert config.candidate_suite.group_size == 1024
    assert config.candidate_suite.nonmember_group_count == 5
    assert config.reuse.mode == "target_eval_only"
    assert config.shadow.count == 5
    assert config.shadow.train_size == 43799
    assert config.train.epochs == 1
    assert config.train.per_device_train_batch_size == 16
    assert config.train.gradient_accumulation_steps == 2
    assert config.precision.bf16 is True
    assert config.attack.method == "online_rmia"
    assert config.attack.gamma == 0.5
    assert config.attack.bootstrap_samples == 2000

    runner = Path("scripts/run_squad_lora_rmia_iid_ablation_formal.sh").read_text(
        encoding="utf-8"
    )
    assert "attack_control" in runner
    assert "plot_rmia_feature" in runner
    assert "validate" in runner
    assert 'if [[ "$stage" == "attack" || "$stage" == "attack_control" ]]' in runner

    submitter = Path("scripts/submit_squad_lora_rmia_iid_ablation.sh").read_text(
        encoding="utf-8"
    )
    assert 'partition="xe8545"' in submitter
    assert 'partition="tmp"' in submitter
    assert (
        "exec bash ./scripts/run_squad_lora_rmia_iid_ablation_formal.sh $model"
        in submitter
    )
    assert "--parsable" in submitter
    assert 'if [[ -n "${EXCLUDE_NODES:-}" ]]' in submitter
    assert 'exclude_args+=(--exclude="$EXCLUDE_NODES")' in submitter
