from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typeguard import typechecked


@typechecked
def learning_curve(train_epoch: Iterable[float], train_loss: Iterable[float]) -> Figure:
    fig, ax = plt.subplots()
    ax.plot(list(train_epoch), list(train_loss))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Learning Curve")
    fig.tight_layout()
    return fig
