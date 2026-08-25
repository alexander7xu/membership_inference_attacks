import pytest

from src.llm_mia.analysis import (
    bootstrap_binary_metrics,
    candidate_relative_loglikelihood,
    online_rmia_score,
    population_relative_loglikelihood,
)
from src.llm_mia.workflow import attack_metrics


def test_candidate_relative_loglikelihood_requires_in_and_out_shadows():
    with pytest.raises(ValueError):
        candidate_relative_loglikelihood(-1.0, [-2.0, -3.0], [1, 1])


def test_online_rmia_uses_population_calibration():
    candidate_relative, in_score, out_score = candidate_relative_loglikelihood(
        -1.0,
        [-1.5, -2.0, -2.5, -3.0, -1.8],
        [1, 0, 1, 0, 1],
    )
    population = [
        population_relative_loglikelihood(-2.0, [-2.2, -2.1, -2.3, -2.4, -2.2]),
        population_relative_loglikelihood(-2.4, [-2.3, -2.5, -2.4, -2.6, -2.5]),
    ]

    assert in_score > out_score
    assert online_rmia_score(candidate_relative, population, gamma=1.0) == 1.0


def test_bootstrap_metrics_are_seeded_and_include_intervals():
    kwargs = {
        "positive_scores": [0.8, 0.9, 1.0],
        "negative_scores": [0.1, 0.2, 0.3],
        "fpr_thresholds": [0.01, 0.05],
        "samples": 50,
        "seed": 42,
    }

    first = bootstrap_binary_metrics(**kwargs)
    second = bootstrap_binary_metrics(**kwargs)

    assert first == second
    assert first["auc_ci_lower"] == pytest.approx(1.0)
    assert first["auc_ci_upper"] == pytest.approx(1.0)


def test_attack_metrics_expand_one_score_into_multiple_evaluator_groups():
    scores = [
        {"candidate_id": "member", "online_rmia_score": 0.9},
        {"candidate_id": "nonmember", "online_rmia_score": 0.1},
    ]
    mapping = [
        {
            "candidate_id": "member",
            "private_group": "gold_squad_target_train",
            "record_membership_label": 1,
        },
        {
            "candidate_id": "nonmember",
            "private_group": "gold_squad_validation",
            "record_membership_label": 0,
        },
        {
            "candidate_id": "nonmember",
            "private_group": "gold_trivia_validation",
            "record_membership_label": 0,
        },
    ]

    metrics = attack_metrics(
        scores,
        mapping,
        calibration_scores=[0.0, 0.2],
        fpr_thresholds=[0.05],
        bootstrap_samples=10,
        seed=42,
    )

    assert metrics["auc_gold_train_vs_gold_squad_validation"] == 1.0
    assert metrics["auc_gold_train_vs_gold_trivia_validation"] == 1.0
    assert metrics["num_candidates"] == 2.0


def test_attack_metrics_reject_conflicting_canonical_labels():
    scores = [{"candidate_id": "same", "online_rmia_score": 0.5}]
    mapping = [
        {
            "candidate_id": "same",
            "private_group": "a",
            "record_membership_label": 0,
        },
        {
            "candidate_id": "same",
            "private_group": "b",
            "record_membership_label": 1,
        },
    ]

    with pytest.raises(ValueError, match="conflicting membership labels"):
        attack_metrics(
            scores,
            mapping,
            calibration_scores=[0.0],
            fpr_thresholds=[0.05],
            bootstrap_samples=10,
            seed=42,
        )
