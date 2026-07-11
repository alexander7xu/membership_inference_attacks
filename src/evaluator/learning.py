import matplotlib.pyplot as plt
from beartype.typing import Iterable
from matplotlib.figure import Figure

from src.utils.annotation import typechecked


@typechecked
def learning_curve(history: dict[str, Iterable[float]]) -> Figure:
    fig, ax = plt.subplots()
    ax.plot(history["train_epoch"], history["train_loss"], label="train")
    if "eval_epoch" in history:
        ax.plot(history["eval_epoch"], history["eval_loss"], label="eval")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Learning Curve")
    ax.legend()
    fig.tight_layout()
    return fig
