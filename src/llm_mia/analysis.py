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


def fixed_variance_lira_parameters(
    shadow_scores_by_candidate: Sequence[Sequence[float]],
    inclusion_masks: Sequence[Sequence[int]],
) -> tuple[list[float], list[float], float, float, int, int]:
    if len(shadow_scores_by_candidate) != len(inclusion_masks):
        raise ValueError("LiRA scores and inclusion masks must have equal length.")
    if not shadow_scores_by_candidate:
        raise ValueError("LiRA requires at least one candidate.")

    in_means: list[float] = []
    out_means: list[float] = []
    in_residual_sum = 0.0
    out_residual_sum = 0.0
    in_degrees = 0
    out_degrees = 0
    for scores, mask in zip(shadow_scores_by_candidate, inclusion_masks, strict=True):
        if len(scores) != len(mask) or not scores:
            raise ValueError(
                "Every LiRA candidate requires one mask bit per shadow score."
            )
        if any(bit not in (0, 1) for bit in mask):
            raise ValueError("LiRA inclusion masks must contain only zero or one.")
        if any(not math.isfinite(float(score)) for score in scores):
            raise ValueError("LiRA shadow scores must be finite.")

        in_scores = [
            float(score) for score, bit in zip(scores, mask, strict=True) if bit == 1
        ]
        out_scores = [
            float(score) for score, bit in zip(scores, mask, strict=True) if bit == 0
        ]
        if not in_scores or not out_scores:
            raise ValueError(
                "Every candidate requires at least one IN and one OUT shadow."
            )

        in_mean = sum(in_scores) / len(in_scores)
        out_mean = sum(out_scores) / len(out_scores)
        in_means.append(in_mean)
        out_means.append(out_mean)
        in_residual_sum += sum((score - in_mean) ** 2 for score in in_scores)
        out_residual_sum += sum((score - out_mean) ** 2 for score in out_scores)
        in_degrees += len(in_scores) - 1
        out_degrees += len(out_scores) - 1

    if in_degrees <= 0 or out_degrees <= 0:
        raise ValueError("LiRA pooled variance requires positive IN and OUT freedom.")
    in_variance = in_residual_sum / in_degrees
    out_variance = out_residual_sum / out_degrees
    if (
        not math.isfinite(in_variance)
        or not math.isfinite(out_variance)
        or in_variance <= 0
        or out_variance <= 0
    ):
        raise ValueError(
            "LiRA pooled IN and OUT variances must be finite and positive."
        )
    return (
        in_means,
        out_means,
        in_variance,
        out_variance,
        in_degrees,
        out_degrees,
    )


def online_lira_fixed_variance_score(
    target_score: float,
    in_mean: float,
    out_mean: float,
    in_variance: float,
    out_variance: float,
) -> float:
    values = (target_score, in_mean, out_mean, in_variance, out_variance)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("LiRA score inputs must be finite.")
    if in_variance <= 0 or out_variance <= 0:
        raise ValueError("LiRA variances must be positive.")
    in_log_likelihood = -0.5 * (
        math.log(2.0 * math.pi * in_variance)
        + (target_score - in_mean) ** 2 / in_variance
    )
    out_log_likelihood = -0.5 * (
        math.log(2.0 * math.pi * out_variance)
        + (target_score - out_mean) ** 2 / out_variance
    )
    score = in_log_likelihood - out_log_likelihood
    if not math.isfinite(score):
        raise ValueError("LiRA produced a non-finite log-likelihood ratio.")
    return score


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


def paired_bootstrap_binary_metric_differences(
    treatment_positive_scores: Sequence[float],
    treatment_negative_scores: Sequence[float],
    control_positive_scores: Sequence[float],
    control_negative_scores: Sequence[float],
    *,
    fpr_thresholds: Sequence[float],
    samples: int,
    seed: int,
) -> dict[str, float]:
    score_sets = (
        treatment_positive_scores,
        treatment_negative_scores,
        control_positive_scores,
        control_negative_scores,
    )
    if any(not scores for scores in score_sets):
        raise ValueError("Paired bootstrap score sets must be non-empty.")
    if len(treatment_positive_scores) != len(control_positive_scores) or len(
        treatment_negative_scores
    ) != len(control_negative_scores):
        raise ValueError("Paired bootstrap treatment/control lengths must match.")
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive.")
    if any(
        not math.isfinite(float(score)) for scores in score_sets for score in scores
    ):
        raise ValueError("Paired bootstrap scores must be finite.")

    positive_count = len(treatment_positive_scores)
    negative_count = len(treatment_negative_scores)
    labels = [1] * positive_count + [0] * negative_count

    def metrics(
        positive_scores: Sequence[float], negative_scores: Sequence[float]
    ) -> tuple[float, dict[float, float]]:
        scores = [*positive_scores, *negative_scores]
        return (
            float(roc_auc_score(labels, scores)),
            {
                threshold: tpr_at_fpr(scores, labels, threshold)
                for threshold in fpr_thresholds
            },
        )

    treatment_auc, treatment_tpr = metrics(
        treatment_positive_scores, treatment_negative_scores
    )
    control_auc, control_tpr = metrics(control_positive_scores, control_negative_scores)
    result = {
        "treatment_auc": treatment_auc,
        "control_auc": control_auc,
        "auc_difference": treatment_auc - control_auc,
    }
    for threshold in fpr_thresholds:
        prefix = f"tpr_at_fpr_{threshold}"
        result[f"treatment_{prefix}"] = treatment_tpr[threshold]
        result[f"control_{prefix}"] = control_tpr[threshold]
        result[f"{prefix}_difference"] = (
            treatment_tpr[threshold] - control_tpr[threshold]
        )

    rng = random.Random(seed)
    auc_differences: list[float] = []
    tpr_differences = {threshold: [] for threshold in fpr_thresholds}
    for _ in range(samples):
        positive_indices = [
            rng.randrange(positive_count) for _ in range(positive_count)
        ]
        negative_indices = [
            rng.randrange(negative_count) for _ in range(negative_count)
        ]
        treatment_auc_sample, treatment_tpr_sample = metrics(
            [treatment_positive_scores[index] for index in positive_indices],
            [treatment_negative_scores[index] for index in negative_indices],
        )
        control_auc_sample, control_tpr_sample = metrics(
            [control_positive_scores[index] for index in positive_indices],
            [control_negative_scores[index] for index in negative_indices],
        )
        auc_differences.append(treatment_auc_sample - control_auc_sample)
        for threshold in fpr_thresholds:
            tpr_differences[threshold].append(
                treatment_tpr_sample[threshold] - control_tpr_sample[threshold]
            )

    def interval(values: Sequence[float]) -> tuple[float, float]:
        ordered = sorted(values)
        lower = ordered[int(0.025 * (len(ordered) - 1))]
        upper = ordered[int(0.975 * (len(ordered) - 1))]
        return float(lower), float(upper)

    auc_lower, auc_upper = interval(auc_differences)
    result["auc_difference_ci_lower"] = auc_lower
    result["auc_difference_ci_upper"] = auc_upper
    for threshold, values in tpr_differences.items():
        lower, upper = interval(values)
        prefix = f"tpr_at_fpr_{threshold}_difference"
        result[f"{prefix}_ci_lower"] = lower
        result[f"{prefix}_ci_upper"] = upper
    return result
