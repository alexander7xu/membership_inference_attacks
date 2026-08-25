from pathlib import Path

import pytest

from src.llm_mia.feature_matrix import (
    FEATURE_NAME,
    common_feature_limits,
    feature_cache_key,
    plot_feature_matrix_ecdf,
    population_center,
)
from src.llm_mia.plotting import CANDIDATE_GROUPS


def _matrix_rows(count: int = 3) -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"{group}-{index}",
            "private_group": group,
            FEATURE_NAME: float(group_index + index / 10),
        }
        for group_index, group in enumerate(CANDIDATE_GROUPS)
        for index in range(count)
    ]


def test_population_center_uses_exact_median() -> None:
    center, centered = population_center([1.0, 2.0, 4.0, 8.0])

    assert center == 3.0
    assert centered == [-2.0, -1.0, 1.0, 5.0]


@pytest.mark.parametrize("values", [[], [1.0, float("nan")], [float("inf")]])
def test_population_center_rejects_invalid_values(values: list[float]) -> None:
    with pytest.raises(ValueError, match="non-empty and finite"):
        population_center(values)


def test_cache_key_is_order_independent_and_content_sensitive() -> None:
    first = feature_cache_key({"checkpoint": "abc", "input": [1, 2]})
    reordered = feature_cache_key({"input": [1, 2], "checkpoint": "abc"})
    changed = feature_cache_key({"checkpoint": "def", "input": [1, 2]})

    assert first == reordered
    assert first != changed


def test_common_limits_cover_all_settings_with_margin() -> None:
    first = _matrix_rows()
    second = _matrix_rows()
    for row in second:
        row[FEATURE_NAME] = float(row[FEATURE_NAME]) - 20.0

    lower, upper = common_feature_limits({"first": first, "second": second})

    all_values = [float(row[FEATURE_NAME]) for row in [*first, *second]]
    assert lower < min(all_values)
    assert upper > max(all_values)


def test_matrix_plot_has_nine_equal_cells(tmp_path: Path) -> None:
    output = tmp_path / "matrix.png"

    plot_feature_matrix_ecdf(
        _matrix_rows(),
        model_display_name="Test model",
        setting_label="Test setting",
        output_path=output,
        dpi=72,
        x_limits=(-2.0, 12.0),
    )

    assert output.stat().st_size > 10_000


def test_matrix_plot_rejects_missing_cell(tmp_path: Path) -> None:
    rows = [
        row for row in _matrix_rows() if row["private_group"] != CANDIDATE_GROUPS[-1]
    ]

    with pytest.raises(ValueError, match="same positive count"):
        plot_feature_matrix_ecdf(
            rows,
            model_display_name="Test model",
            setting_label="Test setting",
            output_path=tmp_path / "matrix.png",
            dpi=72,
            x_limits=(-2.0, 12.0),
        )
