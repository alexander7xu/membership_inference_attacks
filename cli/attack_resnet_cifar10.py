import argparse
from copy import deepcopy
import logging
from pathlib import Path

import torch

from src.attacker import AttackerInterface, load_attacker
from src.dataset import DatasetInterface, load_dataset
from src.evaluator import learning_curve, roc
from src.model import ModelInterface, load_model
from src.utils import make_config_from_dict


logging.basicConfig(level=logging.INFO)


DATASET_CONFIG = {
    "type": "TorchvisionDataset",
    "seed": 42,
    "dataset_class": "CIFAR10",
    "dataloader_kwargs": {
        "batch_size": 1024,
        "num_workers": 8,
        "persistent_workers": True,
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
    "type": "TorchvisionModel",
    "device": "cuda",
    "dtype": "float",
    "optimizer_name": "Adam",
    "model_version": "resnet18",
    "model_kwargs": {
        "num_classes": 10,
    },
    "optimizer_kwargs": {
        "lr": 1e-3,
    },
}

ATTACKER_CONFIGS = {
    "RmiaOffline": {
        "type": "RmiaOfflineAttacker",
        "seed": 42,
        "population_subset_size": 100,
        "num_shadow_models": 4,
        "shadow_model_training_epochs": 10,
        "gamma": 0.5,
        "scale_a": 0.3,
    },
    "RmiaOnline": {
        "type": "RmiaOnlineAttacker",
        "seed": 42,
        "population_subset_size": 100,
        "num_shadow_models": 4,
        "shadow_model_training_epochs": 10,
        "gamma": 0.5,
        "scale_a": 0.3,
    },
    "LiraOffline": {
        "type": "LiraOfflineAttacker",
        "seed": 42,
        "num_shadow_models": 5,
        "shadow_model_training_epochs": 10,
    },
    "LiraOnline": {
        "type": "LiraOnlineAttacker",
        "seed": 42,
        "num_shadow_models": 5,
        "shadow_model_training_epochs": 10,
    },
}

NUM_MEASUREMENT_SAMPLES = 100
NUM_AVAILABLE_SHADOW_SAMPLES = None
ATTACK_BATCH_SIZE = 1
MODEL_PATH = Path("./data/training/resnet18_cifar10_10epochs.pth")


def load_or_train_target_model() -> ModelInterface:
    if MODEL_PATH.exists():
        target_model = load_model(make_config_from_dict(MODEL_CONFIG), str(MODEL_PATH))
        return target_model

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    target_model = load_model(make_config_from_dict(MODEL_CONFIG), None)
    conf = deepcopy(DATASET_CONFIG)
    conf["dataloader_kwargs"]["shuffle"] = conf["dataset_kwargs"]["train"] = True
    train_set = load_dataset(make_config_from_dict(conf))
    conf = deepcopy(DATASET_CONFIG)
    conf["dataloader_kwargs"]["shuffle"] = conf["dataset_kwargs"]["train"] = False
    val_set = load_dataset(make_config_from_dict(conf))

    history = target_model.train(10, train_set.make_loader(), val_set.make_loader())
    fig = learning_curve(history)
    fig.savefig(MODEL_PATH.parent / f"{MODEL_PATH.stem}.png")
    target_model.save_model(str(MODEL_PATH))
    return target_model


def perform_attack(
    attacker: AttackerInterface,
    measurement_train: DatasetInterface,
    measurement_val: DatasetInterface,
) -> tuple[list[float], list[int]]:
    scores = list[float]()
    for query in measurement_train.make_loader(
        batch_size=ATTACK_BATCH_SIZE, num_workers=1
    ):
        scores += attacker.score(query)["scores"].tolist()
    for query in measurement_val.make_loader(
        batch_size=ATTACK_BATCH_SIZE, num_workers=1
    ):
        scores += attacker.score(query)["scores"].tolist()
    return scores, [0] * len(measurement_train) + [1] * len(measurement_val)


def main(args: argparse.Namespace):
    conf = deepcopy(DATASET_CONFIG)
    conf["dataloader_kwargs"]["shuffle"] = conf["dataset_kwargs"]["train"] = True
    train_set = load_dataset(make_config_from_dict(conf))
    conf["dataloader_kwargs"]["shuffle"] = conf["dataset_kwargs"]["train"] = False
    val_set = load_dataset(make_config_from_dict(conf))
    target_model = load_or_train_target_model()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    generator = torch.Generator("cpu").manual_seed(42)
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
        make_config_from_dict(ATTACKER_CONFIGS[args.attacker]),
        target_model,
        shadow_dataset=shadow_set,
    )
    scores, references = perform_attack(attacker, measurement_train, measurement_val)

    save_path_stem = Path(
        f"./data/attack/{args.attacker.replace('_', '-')}_{MODEL_PATH.stem.replace('_', '-')}"
    )
    save_path_stem.parent.mkdir(parents=True, exist_ok=True)

    with open(f"{save_path_stem}.txt", "w", encoding="utf-8") as file:
        file.writelines(f"{x}\n" for x in scores)
    fig, auc = roc(references, scores)
    fig.savefig(f"{save_path_stem}.png")

    print(f"Results saved into {save_path_stem}")
    print(f"{auc=:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("attacker", type=str, choices=ATTACKER_CONFIGS.keys())
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
