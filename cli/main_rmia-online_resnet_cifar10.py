from copy import deepcopy
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch

from attacker import RmiaOnlineAttacker
from dataset import Cifar10Dataset
from evaluator import learning_curve, roc
from model import ResNetModel


logging.basicConfig(level=logging.INFO)


DATASET_CONFIG = {
    "seed": 42,
    "loader": {
        "batch_size": 1024,
        "num_workers": 8,
        "pin_memory": True,
        "shuffle": None,
    },
    "dataset_kwargs": {
        "root": "./data/dataset",
        "download": True,
        "train": None,
    },
}

MODEL_CONFIG = {
    "device": "cuda",
    "dtype": "float",
    "optimizer_name": "Adam",
    "resnet_version": "resnet18",
    "model_kwargs": {
        "num_classes": 10,
    },
    "optimizer_kwargs": {
        "lr": 1e-3,
    },
}

ATTACKER_CONFIG = {
    "seed": 42,
    "population_subset_size": 100,
    "num_shadow_models": 4,
    "shadow_model_training_epochs": 20,
    "gamma": 0.5,
    "scale_a": 0.3,
}

NUM_MEASUREMENT_SAMPLES = 10
MODEL_PATH = Path("./data/training/resnet18_cifar10_10epochs.pth")


def load_or_train_target_model():
    target_model = ResNetModel(MODEL_CONFIG)
    if MODEL_PATH.exists():
        target_model._model.load_state_dict(
            torch.load("./data/training/resnet18_cifar10_10epochs.pth")
        )
        return target_model

    conf = deepcopy(DATASET_CONFIG)
    conf["loader"]["shuffle"] = conf["dataset_kwargs"]["train"] = True
    train_set = Cifar10Dataset(conf)
    history = target_model.train(train_set.make_loader(), num_epochs=100)
    fig = learning_curve(history["train_epoch"], history["train_loss"])
    fig.savefig(MODEL_PATH.parent / f"{MODEL_PATH.stem}.png")
    torch.save(target_model, MODEL_PATH)
    return target_model


def main():
    conf = deepcopy(DATASET_CONFIG)
    conf["loader"]["shuffle"] = conf["dataset_kwargs"]["train"] = True
    train_set = Cifar10Dataset(conf)
    conf["loader"]["shuffle"] = conf["dataset_kwargs"]["train"] = False
    val_set = Cifar10Dataset(conf)

    target_model = load_or_train_target_model()
    attacker = RmiaOnlineAttacker(
        ATTACKER_CONFIG,
        target_model,
        shadow_dataset=val_set,
    )

    generator = torch.Generator("cpu")
    generator.manual_seed(42)
    measurement_train = train_set.select(
        torch.randperm(len(train_set), generator=generator)[:NUM_MEASUREMENT_SAMPLES]
    )
    measurement_val = val_set.select(
        torch.randperm(len(val_set), generator=generator)[:NUM_MEASUREMENT_SAMPLES]
    )

    scores = list[float]()
    # Note that it also works with batch_size > 1
    # In that case, multiple data samples share one shadow model
    for query in measurement_train.make_loader(batch_size=1):
        scores += attacker.score(query)["scores"].tolist()
    for query in measurement_val.make_loader(batch_size=1):
        scores += attacker.score(query)["scores"].tolist()

    with open(
        f"./data/attack/rmia-online_{MODEL_PATH.stem}.txt", "w", encoding="utf-8"
    ) as file:
        file.writelines(f"{x}\n" for x in scores)
    fig, auc = roc([0] * len(measurement_train) + [1] * len(measurement_val), scores)
    fig.savefig(f"./data/attack/rmia-online_{MODEL_PATH.stem}.png")
    print(f"{auc=:.4f}")


if __name__ == "__main__":
    main()
