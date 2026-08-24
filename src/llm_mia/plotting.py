from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Sequence
from pathlib import Path

_SOURCE_ROWS = (
    ("squad_train", "gold_squad_target_train", "squad_train"),
    ("squad_validation", "gold_squad_validation", "squad_validation"),
    ("trivia_validation", "gold_trivia_validation", "trivia_validation"),
)


def candidate_groups_for_variant(variant: str) -> tuple[str, ...]:
    if variant not in {"lora", "fullft"}:
        raise ValueError(f"Unsupported generated-text variant: {variant}")
    groups: list[str] = []
    for _, gold_group, source_suffix in _SOURCE_ROWS:
        groups.extend(
            (
                gold_group,
                f"base_gen_from_{source_suffix}",
                f"{variant}_gen_from_{source_suffix}",
            )
        )
    return tuple(groups)


CANDIDATE_GROUPS = candidate_groups_for_variant("lora")
FULLFT_CANDIDATE_GROUPS = candidate_groups_for_variant("fullft")

GROUP_SOURCE: dict[str, str] = {}
for source, gold_group, source_suffix in _SOURCE_ROWS:
    GROUP_SOURCE[gold_group] = source
    GROUP_SOURCE[f"gen_from_{source_suffix}"] = source
    GROUP_SOURCE[f"base_gen_from_{source_suffix}"] = source
    GROUP_SOURCE[f"lora_gen_from_{source_suffix}"] = source
    GROUP_SOURCE[f"fullft_gen_from_{source_suffix}"] = source
for index in range(5):
    GROUP_SOURCE[f"gold_squad_validation_{index:02d}"] = "squad_validation"
    GROUP_SOURCE[f"target_gen_squad_validation_{index:02d}"] = "squad_validation"

GROUP_VARIANT = {
    group: (
        "gold"
        if group.startswith("gold_")
        else "base_generated"
        if group.startswith("base_gen_")
        else "lora_generated"
        if group.startswith(("gen_from_", "lora_gen_"))
        else "fullft_generated"
    )
    for group in GROUP_SOURCE
}

SOURCE_LABELS = {
    "squad_train": "SQuAD train",
    "squad_validation": "SQuAD validation",
    "trivia_validation": "TriviaQA validation",
}

SOURCE_COLORS = {
    "squad_train": "#C62828",
    "squad_validation": "#1565C0",
    "trivia_validation": "#2E7D32",
}

VARIANT_LABELS = {
    "gold": "Gold answer",
    "base_generated": "Base generated",
    "lora_generated": "LoRA generated",
    "fullft_generated": "Full-FT generated",
}

VARIANT_LINESTYLES = {
    "gold": "-",
    "base_generated": "--",
    "lora_generated": ":",
    "fullft_generated": ":",
}

IID_GROUP_COLORS = {
    "gold_squad_target_train": "#C62828",
    "gold_squad_validation_00": "#1565C0",
    "gold_squad_validation_01": "#6A1B9A",
    "gold_squad_validation_02": "#2E7D32",
    "gold_squad_validation_03": "#EF6C00",
    "gold_squad_validation_04": "#00838F",
}
TARGET_GENERATED_GROUP_COLORS = {
    "gold_squad_target_train": "#C62828",
    "target_gen_squad_validation_00": "#1565C0",
    "target_gen_squad_validation_01": "#6A1B9A",
    "target_gen_squad_validation_02": "#2E7D32",
    "target_gen_squad_validation_03": "#EF6C00",
    "target_gen_squad_validation_04": "#00838F",
}


def _sample_key(seed: int, group: str, candidate_id: str) -> str:
    payload = f"{seed}:rmia-feature:{group}:{candidate_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rows_by_candidate_id(
    rows: list[dict[str, object]], *, row_kind: str
) -> dict[str, dict[str, object]]:
    rows_by_id: dict[str, dict[str, object]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id:
            raise ValueError(f"Every {row_kind} row requires a candidate_id.")
        if candidate_id in rows_by_id:
            raise ValueError(f"Duplicate {row_kind} candidate_id: {candidate_id}")
        rows_by_id[candidate_id] = row
    return rows_by_id


def sample_group_candidates(
    public_rows: list[dict[str, object]],
    private_rows: list[dict[str, object]],
    *,
    sample_size: int,
    seed: int,
    group_order: Sequence[str] = CANDIDATE_GROUPS,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")

    public_by_id = _rows_by_candidate_id(public_rows, row_kind="public")
    private_by_id = _rows_by_candidate_id(private_rows, row_kind="private")
    if set(public_by_id) != set(private_by_id):
        raise ValueError("Public and private candidate IDs must match exactly.")

    grouped: dict[str, list[str]] = {group: [] for group in group_order}
    for candidate_id, private_row in private_by_id.items():
        group = str(private_row.get("private_group", ""))
        if group not in grouped:
            raise ValueError(f"Unexpected candidate group: {group}")
        grouped[group].append(candidate_id)

    sampled_public: list[dict[str, object]] = []
    sampled_private: list[dict[str, object]] = []
    for group in group_order:
        candidate_ids = grouped[group]
        if len(candidate_ids) < sample_size:
            raise ValueError(
                f"Group {group} has {len(candidate_ids)} rows; "
                f"{sample_size} are required."
            )
        ranked_ids = sorted(
            candidate_ids,
            key=lambda candidate_id: (
                _sample_key(seed, group, candidate_id),
                candidate_id,
            ),
        )
        for rank, candidate_id in enumerate(ranked_ids[:sample_size]):
            sampled_public.append(public_by_id[candidate_id])
            sampled_private.append({**private_by_id[candidate_id], "sample_rank": rank})
    return sampled_public, sampled_private


def sample_group_feature_rows(
    score_rows: list[dict[str, object]],
    private_rows: list[dict[str, object]],
    *,
    feature: str,
    sample_size: int,
    seed: int,
    group_order: Sequence[str] = CANDIDATE_GROUPS,
) -> list[dict[str, object]]:
    score_by_id = _rows_by_candidate_id(score_rows, row_kind="score")
    private_by_id = _rows_by_candidate_id(private_rows, row_kind="private")
    if set(score_by_id) != set(private_by_id):
        raise ValueError("Score and private-label candidate IDs must match exactly.")

    public_rows = [{"candidate_id": candidate_id} for candidate_id in score_by_id]
    sampled_public, sampled_private = sample_group_candidates(
        public_rows,
        private_rows,
        sample_size=sample_size,
        seed=seed,
        group_order=group_order,
    )
    sampled: list[dict[str, object]] = []
    for public_row, private_row in zip(sampled_public, sampled_private, strict=True):
        candidate_id = str(public_row["candidate_id"])
        score_row = score_by_id[candidate_id]
        value = score_row.get(feature)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                f"Candidate {candidate_id} has invalid {feature}: {value!r}."
            )
        sampled_row: dict[str, object] = {
            "candidate_id": candidate_id,
            "private_group": private_row["private_group"],
            "sample_rank": private_row["sample_rank"],
            feature: float(value),
        }
        for name in (
            "target_mean_logprob",
            "shadow_reference_mean_logprob",
            "num_reference_shadows",
        ):
            if name in score_row:
                sampled_row[name] = score_row[name]
        sampled.append(sampled_row)
    return sampled


def _linear_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_group_features(
    sampled_rows: list[dict[str, object]],
    *,
    feature: str,
    group_order: Sequence[str] = CANDIDATE_GROUPS,
) -> dict[str, dict[str, float]]:
    summaries: dict[str, dict[str, float]] = {}
    for group in group_order:
        values = [
            float(row[feature]) for row in sampled_rows if row["private_group"] == group
        ]
        if not values:
            raise ValueError(f"Group {group} has no sampled values.")
        summaries[group] = {
            "count": float(len(values)),
            "mean": float(statistics.fmean(values)),
            "std": float(statistics.pstdev(values)),
            "min": float(min(values)),
            "p05": _linear_quantile(values, 0.05),
            "p25": _linear_quantile(values, 0.25),
            "median": float(statistics.median(values)),
            "p75": _linear_quantile(values, 0.75),
            "p95": _linear_quantile(values, 0.95),
            "max": float(max(values)),
        }
    return summaries


def plot_group_feature_ecdf(
    sampled_rows: list[dict[str, object]],
    *,
    feature: str,
    model_display_name: str,
    output_path: Path,
    dpi: int,
    group_order: Sequence[str] = CANDIDATE_GROUPS,
    title: str | None = None,
    x_label: str | None = None,
) -> None:
    if dpi <= 0:
        raise ValueError("Figure DPI must be positive.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axis = plt.subplots(figsize=(12.5, 6.4))
    figure.subplots_adjust(right=0.76)
    distribution_colors = (
        IID_GROUP_COLORS
        if set(group_order) == set(IID_GROUP_COLORS)
        else TARGET_GENERATED_GROUP_COLORS
        if set(group_order) == set(TARGET_GENERATED_GROUP_COLORS)
        else None
    )
    group_counts: set[int] = set()
    for group in group_order:
        values = sorted(
            float(row[feature]) for row in sampled_rows if row["private_group"] == group
        )
        if not values:
            raise ValueError(f"Group {group} has no sampled values.")
        group_counts.add(len(values))
        cumulative = [(index + 1) / len(values) for index in range(len(values))]
        source = GROUP_SOURCE[group]
        variant = GROUP_VARIANT[group]
        axis.step(
            values,
            cumulative,
            where="post",
            linewidth=2.0,
            linestyle="-" if distribution_colors else VARIANT_LINESTYLES[variant],
            color=distribution_colors[group]
            if distribution_colors
            else SOURCE_COLORS[source],
        )

    if len(group_counts) != 1:
        raise ValueError("Every plotted group must have the same sample count.")
    sample_size = next(iter(group_counts))
    axis.axvline(0.0, color="#555555", linewidth=1.0, linestyle=(0, (1, 2)))
    axis.set_title(
        f"{model_display_name}: {title or 'RMIA feature distributions'}\n"
        f"{len(group_order)} candidate groups, n={sample_size} per group",
        pad=12,
    )
    axis.set_xlabel(
        x_label or "Relative log-likelihood (target vs. five-shadow reference)"
    )
    axis.set_ylabel("Empirical cumulative probability")
    axis.set_ylim(0.0, 1.0)
    axis.grid(color="#D7D7D7", linewidth=0.7, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    if distribution_colors:
        generated_suite = distribution_colors is TARGET_GENERATED_GROUP_COLORS
        prefix = (
            "target_gen_squad_validation_"
            if generated_suite
            else "gold_squad_validation_"
        )
        labels = {
            "gold_squad_target_train": "Target members",
            **{
                f"{prefix}{index:02d}": (
                    f"Target-generated slice {index:02d}"
                    if generated_suite
                    else f"Validation slice {index:02d}"
                )
                for index in range(5)
            },
        }
        axis.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=distribution_colors[group],
                    linewidth=2.4,
                    label=labels[group],
                )
                for group in group_order
            ],
            title=("Generated SQuAD group" if generated_suite else "Gold SQuAD group"),
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
            fontsize=9,
            title_fontsize=9,
        )
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        return

    source_handles = [
        Line2D(
            [0],
            [0],
            color=SOURCE_COLORS[source],
            linewidth=2.4,
            label=SOURCE_LABELS[source],
        )
        for source in SOURCE_LABELS
    ]
    plotted_variants = tuple(
        dict.fromkeys(GROUP_VARIANT[group] for group in group_order)
    )
    variant_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=2.4,
            linestyle=VARIANT_LINESTYLES[variant],
            label=VARIANT_LABELS[variant],
        )
        for variant in plotted_variants
    ]
    source_legend = axis.legend(
        handles=source_handles,
        title="Data source (color)",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=9,
        title_fontsize=9,
    )
    axis.add_artist(source_legend)
    axis.legend(
        handles=variant_handles,
        title="Text origin (line)",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.70),
        frameon=False,
        fontsize=9,
        title_fontsize=9,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
