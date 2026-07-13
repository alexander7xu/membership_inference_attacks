import pytest

from src.llm_mia.analysis import (
    bootstrap_binary_metrics,
    candidate_relative_loglikelihood,
    online_rmia_score,
    population_relative_loglikelihood,
)


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
