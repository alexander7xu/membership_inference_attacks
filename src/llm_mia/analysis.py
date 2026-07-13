from __future__ import annotations

import math
import random
from collections.abc import Sequence

from sklearn.metrics import roc_auc_score, roc_curve


def logmeanexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("logmeanexp requires at least one value.")
    maximum = max(values)
    return maximum + math.log(
        sum(math.exp(value - maximum) for value in values) / len(values)
    )


def candidate_relative_loglikelihood(
    target_logprob: float,
    shadow_logprobs: Sequence[float],
    inclusion_mask: Sequence[int],
) -> tuple[float, float, float]:
    if len(shadow_logprobs) != len(inclusion_mask):
        raise ValueError("Shadow scores and inclusion mask must have equal length.")
    in_scores = [
        score
        for score, include in zip(shadow_logprobs, inclusion_mask, strict=True)
        if include == 1
    ]
    out_scores = [
        score
        for score, include in zip(shadow_logprobs, inclusion_mask, strict=True)
        if include == 0
    ]
    if not in_scores or not out_scores:
        raise ValueError("Every candidate requires at least one IN and one OUT shadow.")

    in_logmean = logmeanexp(in_scores)
    out_logmean = logmeanexp(out_scores)
    reference_logprob = logmeanexp([in_logmean, out_logmean])
    return target_logprob - reference_logprob, in_logmean, out_logmean


def population_relative_loglikelihood(
    target_logprob: float,
    shadow_logprobs: Sequence[float],
) -> float:
    return target_logprob - logmeanexp(shadow_logprobs)


def online_rmia_score(
    candidate_relative_loglikelihood_value: float,
    population_relative_loglikelihoods: Sequence[float],
    *,
    gamma: float,
) -> float:
    if not population_relative_loglikelihoods:
        raise ValueError("RMIA requires population calibration scores.")
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    log_gamma = math.log(gamma)
    return sum(
        candidate_relative_loglikelihood_value - population_value > log_gamma
        for population_value in population_relative_loglikelihoods
    ) / len(population_relative_loglikelihoods)


def tpr_at_fpr(
    scores: Sequence[float], labels: Sequence[int], target_fpr: float
) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    values = [value for value, rate in zip(tpr, fpr, strict=True) if rate <= target_fpr]
    return float(max(values) if values else 0.0)


def bootstrap_binary_metrics(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    *,
    fpr_thresholds: Sequence[float],
    samples: int,
    seed: int,
) -> dict[str, float]:
    if not positive_scores or not negative_scores:
        return {}
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive.")

    rng = random.Random(seed)
    auc_values: list[float] = []
    tpr_values = {threshold: [] for threshold in fpr_thresholds}
    for _ in range(samples):
        positives = [rng.choice(positive_scores) for _ in positive_scores]
        negatives = [rng.choice(negative_scores) for _ in negative_scores]
        labels = [1] * len(positives) + [0] * len(negatives)
        scores = positives + negatives
        auc_values.append(float(roc_auc_score(labels, scores)))
        for threshold in fpr_thresholds:
            tpr_values[threshold].append(tpr_at_fpr(scores, labels, threshold))

    def interval(values: Sequence[float]) -> tuple[float, float]:
        ordered = sorted(values)
        lower = ordered[int(0.025 * (len(ordered) - 1))]
        upper = ordered[int(0.975 * (len(ordered) - 1))]
        return float(lower), float(upper)

    result = {}
    auc_lower, auc_upper = interval(auc_values)
    result["auc_ci_lower"] = auc_lower
    result["auc_ci_upper"] = auc_upper
    for threshold, values in tpr_values.items():
        lower, upper = interval(values)
        result[f"tpr_at_fpr_{threshold}_ci_lower"] = lower
        result[f"tpr_at_fpr_{threshold}_ci_upper"] = upper
    return result
