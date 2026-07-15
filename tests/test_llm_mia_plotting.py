from collections import Counter
from pathlib import Path

import pytest

from src.llm_mia.plotting import (
    CANDIDATE_GROUPS,
    FULLFT_CANDIDATE_GROUPS,
    GROUP_SOURCE,
    GROUP_VARIANT,
    SOURCE_COLORS,
    VARIANT_LINESTYLES,
    plot_group_feature_ecdf,
    sample_group_candidates,
    sample_group_feature_rows,
    summarize_group_features,
)

FEATURE = "relative_log_likelihood"


def _candidate_rows(rows_per_group: int = 4):
    public = []
    private = []
    scores = []
    for group_index, group in enumerate(CANDIDATE_GROUPS):
        for row_index in range(rows_per_group):
            candidate_id = f"{group}-{row_index}"
            public.append({"candidate_id": candidate_id, "prompt": "p"})
            private.append({"candidate_id": candidate_id, "private_group": group})
            scores.append(
                {
                    "candidate_id": candidate_id,
                    FEATURE: group_index + row_index / 10,
                    "num_reference_shadows": 5,
                }
            )
    return public, private, scores


def test_group_matrix_has_three_sources_and_three_variants() -> None:
    combinations = {
        (GROUP_SOURCE[group], GROUP_VARIANT[group]) for group in CANDIDATE_GROUPS
    }

    assert len(CANDIDATE_GROUPS) == 9
    assert len(set(GROUP_SOURCE.values())) == 3
    assert len({GROUP_VARIANT[group] for group in CANDIDATE_GROUPS}) == 3
    assert len(combinations) == 9
    assert len(set(SOURCE_COLORS.values())) == 3
    assert len(set(VARIANT_LINESTYLES.values())) == 3


def test_fullft_group_matrix_has_three_sources_and_three_variants() -> None:
    combinations = {
        (GROUP_SOURCE[group], GROUP_VARIANT[group]) for group in FULLFT_CANDIDATE_GROUPS
    }

    assert len(FULLFT_CANDIDATE_GROUPS) == 9
    assert len({GROUP_VARIANT[group] for group in FULLFT_CANDIDATE_GROUPS}) == 3
    assert len(combinations) == 9


def test_group_candidate_sampling_is_deterministic_and_order_independent() -> None:
    public, private, _ = _candidate_rows()

    first = sample_group_candidates(public, private, sample_size=2, seed=42)
    second = sample_group_candidates(
        list(reversed(public)),
        list(reversed(private)),
        sample_size=2,
        seed=42,
    )

    assert first == second
    assert Counter(row["private_group"] for row in first[1]) == {
        group: 2 for group in CANDIDATE_GROUPS
    }
    assert all(row["sample_rank"] in {0, 1} for row in first[1])


def test_group_sampling_rejects_mismatched_candidate_ids() -> None:
    public, private, _ = _candidate_rows()
    private.pop()

    with pytest.raises(ValueError, match="must match exactly"):
        sample_group_candidates(public, private, sample_size=2, seed=42)


def test_group_summary_and_ecdf_figure(tmp_path: Path) -> None:
    _, private, scores = _candidate_rows()
    sampled = sample_group_feature_rows(
        scores, private, feature=FEATURE, sample_size=3, seed=42
    )

    summaries = summarize_group_features(sampled, feature=FEATURE)
    figure_path = tmp_path / "feature.png"
    plot_group_feature_ecdf(
        sampled,
        feature=FEATURE,
        model_display_name="Test model",
        output_path=figure_path,
        dpi=72,
    )

    assert len(sampled) == 9 * 3
    assert all(summary["count"] == 3.0 for summary in summaries.values())
    assert figure_path.stat().st_size > 1_000
