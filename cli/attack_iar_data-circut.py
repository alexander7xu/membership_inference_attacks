import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import torch

from attacker import RmiaOfflineAttacker, LiraOfflineAttacker
from hack_aear.hack_aear import HackAearDataset, HackAearModel


logging.basicConfig(level=logging.INFO)


POPULATION_SUBSET_SIZE = 1000
MODEL_NAME = "var_30"
SUBSETS = {
    "M1_train": "imagenet_train",
    "M1_val": "imagenet_val",
    "M2_train": f"{MODEL_NAME.replace('_', '')}train_generated",
    "M2_val": f"{MODEL_NAME.replace('_', '')}val_generated",
    "M2_gen": f"{MODEL_NAME.replace('_', '')}M2test_generated",
}


MODEL_CONFIG = {
    "model_name": MODEL_NAME,
    "features_dir": "/storage/juanguixu/proj_save/iar_mia/features/",
    "metric": "icas",
    "size": POPULATION_SUBSET_SIZE,
    "device": "cpu",
    "dtype": "float",
}

ATTACKER_CONFIG_RMIA = {
    "seed": 42,
    "population_subset_size": POPULATION_SUBSET_SIZE,
    "num_shadow_models": 5,
    "shadow_model_training_epochs": 5,
    "gamma": 0.75,
    "scale_a": 0.3,
}

ATTACKER_CONFIG_LIRA = {
    "seed": 42,
    "num_shadow_models": 5,
    "shadow_model_training_epochs": 5,
}


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    target_model = HackAearModel(MODEL_CONFIG, None)
    population_set = HackAearDataset({"split": "M1_gen_population"})

    rmia_ratios = dict[str, list[list[float]]]()
    rmia_scores = dict[str, list[float]]()
    attacker = RmiaOfflineAttacker(
        ATTACKER_CONFIG_RMIA, target_model, shadow_dataset=population_set
    )
    for split in SUBSETS.keys():
        rmia_ratios[split] = list[list[float]]()
        rmia_scores[split] = list[float]()
        subset = HackAearDataset({"split": split})
        for query in subset.make_loader(batch_size=1):
            rmia_ratios[split] += attacker.score(query)["ratios"].tolist()
            rmia_scores[split] += attacker.score(query)["ratios"].mean(-1).tolist()

    # population_set has no effect here
    HackAearModel.NUM_SHADOW_MODELS = 0
    lira_scores = dict[str, list[float]]()
    attacker = LiraOfflineAttacker(
        ATTACKER_CONFIG_LIRA, target_model, shadow_dataset=population_set
    )
    for split in SUBSETS.keys():
        lira_scores[split] = list[float]()
        subset = HackAearDataset({"split": split})
        for query in subset.make_loader(batch_size=1):
            lira_scores[split] += attacker.score(query)["scores"].tolist()

    save_dir = Path(f"./data/strong_mia/{MODEL_NAME}_M2/orig_enc/")
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for k, v in rmia_scores.items():
        plt.hist(v, bins=100, label=k, alpha=0.8)
    plt.legend()
    plt.savefig(save_dir / f"RMIA_ratios.png")

    plt.figure()
    for k, v in lira_scores.items():
        plt.hist(v, bins=100, label=k, alpha=0.8)
    plt.legend()
    plt.savefig(save_dir / f"LIRA_scores.png")

    for k in SUBSETS.keys():
        torch.save({"strong_mia": {
            # "_configs_rmia": {"model": MODEL_CONFIG, "attacker": ATTACKER_CONFIG},
            "rmia_ratios": rmia_ratios[k],
            "rmia": rmia_scores[k],
            "lira": lira_scores[k],
        }}, save_dir / f"{SUBSETS[k]}.pt") # fmt:skip


if __name__ == "__main__":
    main()
