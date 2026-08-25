from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.llm_mia.plotting import (
    CANDIDATE_GROUPS,
    GROUP_SOURCE,
    GROUP_VARIANT,
    SOURCE_COLORS,
    SOURCE_LABELS,
    VARIANT_LABELS,
)

FEATURE_NAME = "population_centered_relative_log_likelihood"
SOURCE_ORDER = ("squad_train", "squad_validation", "trivia_validation")
VARIANT_ORDER = ("gold", "base_generated", "lora_generated")


def population_center(values: Sequence[float]) -> tuple[float, list[float]]:
    normalized = [float(value) for value in values]
    if not normalized or any(not math.isfinite(value) for value in normalized):
        raise ValueError("Population relative scores must be non-empty and finite.")
    center = float(statistics.median(normalized))
    return center, [value - center for value in normalized]


def feature_cache_key(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def common_feature_limits(
    rows_by_setting: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    feature: str = FEATURE_NAME,
) -> tuple[float, float]:
    values = [float(row[feature]) for rows in rows_by_setting.values() for row in rows]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Matrix feature values must be non-empty and finite.")
    lower = min(values)
    upper = max(values)
    margin = max(1.0, abs(lower) * 0.05) if lower == upper else (upper - lower) * 0.03
    return lower - margin, upper + margin


def plot_feature_matrix_ecdf(
    sampled_rows: Sequence[Mapping[str, object]],
    *,
    model_display_name: str,
    setting_label: str,
    output_path: Path,
    dpi: int,
    x_limits: tuple[float, float],
    feature: str = FEATURE_NAME,
) -> None:
    if dpi <= 0:
        raise ValueError("Figure DPI must be positive.")
    if not x_limits[0] < x_limits[1] or any(
        not math.isfinite(value) for value in x_limits
    ):
        raise ValueError("Shared x-axis limits must be finite and increasing.")

    rows_by_group: dict[str, list[float]] = {group: [] for group in CANDIDATE_GROUPS}
    for row in sampled_rows:
        group = str(row.get("private_group", ""))
        if group not in rows_by_group:
            raise ValueError(f"Unexpected matrix candidate group: {group}")
        value = float(row[feature])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite matrix feature for group {group}.")
        rows_by_group[group].append(value)
    counts = {len(values) for values in rows_by_group.values()}
    if len(counts) != 1 or not counts or next(iter(counts)) <= 0:
        raise ValueError("Every matrix cell must contain the same positive count.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(13.2, 10.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    group_for_cell = {
        (GROUP_SOURCE[group], GROUP_VARIANT[group]): group for group in CANDIDATE_GROUPS
    }
    sample_size = next(iter(counts))
    for row_index, source in enumerate(SOURCE_ORDER):
        for column_index, variant in enumerate(VARIANT_ORDER):
            axis = axes[row_index][column_index]
            group = group_for_cell[(source, variant)]
            values = sorted(rows_by_group[group])
            cumulative = [(index + 1) / len(values) for index in range(len(values))]
            axis.step(
                values,
                cumulative,
                where="post",
                linewidth=2.0,
                color=SOURCE_COLORS[source],
            )
            axis.axvline(0.0, color="#555555", linewidth=1.0, linestyle=(0, (1, 2)))
            axis.set_xlim(*x_limits)
            axis.set_ylim(0.0, 1.0)
            axis.grid(color="#D7D7D7", linewidth=0.7, alpha=0.7)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if row_index == 0:
                axis.set_title(VARIANT_LABELS[variant], pad=8)
            if column_index == 0:
                axis.annotate(
                    SOURCE_LABELS[source],
                    xy=(-0.22, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )
    figure.suptitle(
        f"{model_display_name}: {setting_label}\n"
        f"Population-centered target-versus-five-shadow feature, n={sample_size} per cell",
        fontsize=15,
    )
    figure.supxlabel("Population-centered relative mean log-likelihood")
    figure.supylabel("Empirical cumulative probability")
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
