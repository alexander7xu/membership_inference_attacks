import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch

from attacker import RmiaOfflineAttacker
from hack_aear.hack_aear import HackAearDataset, HackAearModel
from evaluator import roc


logging.basicConfig(level=logging.INFO)


POPULATION_SUBSET_SIZE = 1000
MODEL_NAME = "rar_xxl"
SPLITS = [
    "M1_train",
    "M1_val",
    "M2_train",
    "M2_val",
    "M2_gen",
]


MODEL_CONFIG = {
    "model_name": MODEL_NAME,
    "features_dir": "/storage/juanguixu/proj_save/iar_mia/features/",
    "metric": "icas",
    "size": POPULATION_SUBSET_SIZE,
    "device": "cpu",
    "dtype": "float",
}

ATTACKER_CONFIG = {
    "seed": 42,
    "population_subset_size": POPULATION_SUBSET_SIZE,
    "num_shadow_models": 5,
    "shadow_model_training_epochs": 5,
    "gamma": 0.75,
    "scale_a": 0.3,
}


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    target_model = HackAearModel(MODEL_CONFIG, None)
    population_set = HackAearDataset({"split": "M1_gen_population"})

    attacker = RmiaOfflineAttacker(
        ATTACKER_CONFIG, target_model, shadow_dataset=population_set
    )

    ratios = dict[str, list[list[float]]]()
    for split in SPLITS:
        ratios[split] = list[list[float]]()
        subset = HackAearDataset({"split": split})
        for query in subset.make_loader(batch_size=1):
            ratios[split] += attacker.score(query)["ratio"].tolist()

    import matplotlib.pyplot as plt

    plt.figure()
    for k, v in ratios.items():
        plt.hist([sum(x) / len(x) for x in v], bins=100, label=k, alpha=0.8)
    plt.legend()
    save_dir = Path(f"./data/strong_mia/{MODEL_NAME}_M2/orig_enc/")
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"RMIA_ratios.png")

    subsets = {
        "M1_train": "imagenet_train",
        "M1_val": "imagenet_val",
        "M2_train": f"{MODEL_NAME.replace('_', '')}train_generated",
        "M2_val": f"{MODEL_NAME.replace('_', '')}val_generated",
        "M2_gen": f"{MODEL_NAME.replace('_', '')}M2test_generated",
    }

    for k, v in ratios.items():
        torch.save({"strong_mia": {
            # "_configs_rmia": {"model": MODEL_CONFIG, "attacker": ATTACKER_CONFIG},
            "rmia": v,
        }}, save_dir / f"{subsets[k]}.pt") # fmt:skip

    """
    with open(
        f"./data/attack/rmia-offline_{MODEL_NAME}.txt", "w", encoding="utf-8"
    ) as file:
        file.writelines(f"{x}\n" for x in scores)
    fig, auc = roc([0] * len(measurement_train) + [1] * len(measurement_val), scores)
    fig.savefig(f"./data/attack/rmia-offline_{MODEL_NAME}.png")
    print(f"{auc=:.4f}")
    """


if __name__ == "__main__":
    main()
