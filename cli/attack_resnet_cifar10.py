import argparse
from copy import deepcopy
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch

from attacker import load_attacker, AttackerInterface
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

ATTACKER_CONFIGS = {
    "rmia_offline": {
        "seed": 42,
        "population_subset_size": 100,
        "num_shadow_models": 4,
        "shadow_model_training_epochs": 10,
        "gamma": 0.5,
        "scale_a": 0.3,
    },
    "rmia_online": {
        "seed": 42,
        "population_subset_size": 100,
        "num_shadow_models": 4,
        "shadow_model_training_epochs": 10,
        "gamma": 0.5,
        "scale_a": 0.3,
    },
    "lira_offline": {
        "seed": 42,
        "num_shadow_models": 5,
        "shadow_model_training_epochs": 10,
    },
    "lira_online": {
        "seed": 42,
        "num_shadow_models": 5,
        "shadow_model_training_epochs": 10,
    },
}

NUM_MEASUREMENT_SAMPLES = 100
NUM_AVAILABLE_SHADOW_SAMPLES = None
MODEL_PATH = Path("./data/training/resnet18_cifar10_10epochs.pth")


def load_or_train_target_model() -> ResNetModel:
    if MODEL_PATH.exists():
        target_model = ResNetModel(MODEL_CONFIG, str(MODEL_PATH))
        return target_model

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    target_model = ResNetModel(MODEL_CONFIG, None)
    conf = deepcopy(DATASET_CONFIG)
    conf["loader"]["shuffle"] = conf["dataset_kwargs"]["train"] = True
    train_set = Cifar10Dataset(conf)
    conf = deepcopy(DATASET_CONFIG)
    conf["loader"]["shuffle"] = conf["dataset_kwargs"]["train"] = False
    val_set = Cifar10Dataset(conf)

    history = target_model.train(10, train_set.make_loader(), val_set.make_loader())
    fig = learning_curve(history)
    fig.savefig(MODEL_PATH.parent / f"{MODEL_PATH.stem}.png")
    target_model.save_model(str(MODEL_PATH))
    return target_model


def perform_attack(
    attacker: AttackerInterface,
    measurement_train: Cifar10Dataset,
    measurement_val: Cifar10Dataset,
) -> tuple[list[float], list[int]]:
    scores = list[float]()
    for query in measurement_train.make_loader(batch_size=1, num_workers=1):
        scores += attacker.score(query)["scores"].tolist()
    for query in measurement_val.make_loader(batch_size=1, num_workers=1):
        scores += attacker.score(query)["scores"].tolist()
    return scores, [0] * len(measurement_train) + [1] * len(measurement_val)


def main(args: argparse.Namespace):
    conf = deepcopy(DATASET_CONFIG)
    conf["loader"]["shuffle"] = conf["dataset_kwargs"]["train"] = True
    train_set = Cifar10Dataset(conf)
    conf["loader"]["shuffle"] = conf["dataset_kwargs"]["train"] = False
    val_set = Cifar10Dataset(conf)
    target_model = load_or_train_target_model()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    generator = torch.Generator("cpu")
    generator.manual_seed(42)
    measurement_train = train_set.select(
        torch.randperm(len(train_set), generator=generator)[:NUM_MEASUREMENT_SAMPLES]
    )
    indices = torch.randperm(len(val_set), generator=generator)
    measurement_val = val_set.select(indices[:NUM_MEASUREMENT_SAMPLES])

    if NUM_AVAILABLE_SHADOW_SAMPLES is None:
        shadow_set = val_set.select(indices[NUM_MEASUREMENT_SAMPLES:])
    else:
        end = max(len(val_set), NUM_AVAILABLE_SHADOW_SAMPLES + NUM_MEASUREMENT_SAMPLES)
        shadow_set = val_set.select(indices[NUM_MEASUREMENT_SAMPLES:end])
        del end
    print("Number of shadow samples:", len(shadow_set))

    attacker = load_attacker(
        args.attacker,
        ATTACKER_CONFIGS[args.attacker],
        target_model,
        shadow_dataset=shadow_set,
    )
    scores, references = perform_attack(attacker, measurement_train, measurement_val)
    with open(
        f"./data/attack/{args.attacker}_{MODEL_PATH.stem}.txt", "w", encoding="utf-8"
    ) as file:
        file.writelines(f"{x}\n" for x in scores)
    fig, auc = roc(references, scores)
    fig.savefig(f"./data/attack/{args.attacker}_{MODEL_PATH.stem}.png")

    print(f"Results saved into ./data/attack/{args.attacker}_{MODEL_PATH.stem}")
    print(f"{auc=:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("attacker", type=str, choices=ATTACKER_CONFIGS.keys())
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
