from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve, auc

from src.utils.annotation import typechecked


@typechecked
def roc(references: Iterable[int], scores: Iterable[float]) -> tuple[Figure, float]:
    tpr, fpr, _ = roc_curve(references, scores)
    auc_score = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (area = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend()
    fig.tight_layout()
    return fig, auc_score
