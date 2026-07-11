import pytest

from src.evaluator.attack import roc


def test_roc_uses_member_positive_scores():
    references = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]

    fig, auc_score = roc(references, scores)

    try:
        assert auc_score == pytest.approx(1.0)
    finally:
        fig.clear()
