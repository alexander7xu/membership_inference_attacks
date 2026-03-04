from pathlib import Path
from typing import override, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from .notebook_utils2 import icas_score, uncond_prob, cond_prob, delta
from dataset._dataset_interface import DatasetInterface
from model._model_interface import ModelInterface


class HackAearDataset(DatasetInterface):
    class FakeDataset:
        def __init__(self, split: str):
            self._split = split

        def __len__(self):
            return 1000

        def __getitem__(self, i):
            return {"inputs": i, "labels": 0, "_split": self._split}

    def __init__(self, config: dict, *, _indices: Iterable[int] | None = None):
        super().__init__(config)
        self._split: str = config["split"]

    @override
    def __len__(self) -> int:
        return 1000

    @override
    def select(self, indices: Iterable[int]) -> "HackAearDataset":
        return self

    @override
    def make_loader(self, **overwrite_config) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.FakeDataset(self._split))


class HackAearModel(ModelInterface):
    NUM_SHADOW_MODELS = 0

    def __init__(self, config: dict, trained_model_path: str | None):
        super().__init__(config=config, trained_model_path=trained_model_path)
        self._size = self.config["size"]
        model_name = self.config["model_name"]
        features_root = Path(self.config["features_dir"])
        subsets = {
            "M1_train": "imagenet_train",
            "M1_val": "imagenet_val",
            "M1_gen_population": f"{model_name.replace('_', '')}_population",
            "M2_train": f"{model_name.replace('_', '')}train_generated",
            "M2_val": f"{model_name.replace('_', '')}val_generated",
            "M2_gen": f"{model_name.replace('_', '')}M2test_generated",
        }

        self._probs = dict()
        shadow_idx = type(self).NUM_SHADOW_MODELS
        type(self).NUM_SHADOW_MODELS += 1
        for k, filename in subsets.items():
            model_id = "M2" if shadow_idx == 0 else f"rmia-shadow@{shadow_idx}"
            prefix = f"{model_name}_{model_id}/orig_enc"
            path = features_root / f"{prefix}/{filename}.pt"
            assert path.exists(), path
            self._probs[k] = self._prepare(path, self.config["metric"], self._size)

    @staticmethod
    def _prepare(path: str, metric: str, size: int):
        feature_list = torch.load(path, weights_only=False)

        uncond, cond = (
            np.stack(feature_list["unconditional"]["log_probs"])[:size],
            np.stack(feature_list["conditional"]["log_probs"])[:size],
        )
        if metric == "uncond":
            probs = torch.exp(torch.from_numpy(uncond_prob(uncond)))
        elif metric == "cond":
            probs = torch.exp(torch.from_numpy(cond_prob(cond)))
        elif metric == "delta":
            probs = torch.sigmoid(torch.from_numpy(delta(cond - uncond)))
        elif metric == "icas":
            probs = torch.sigmoid(torch.from_numpy(icas_score(cond, uncond)[0]))
        else:
            raise NotImplementedError(metric)
        return probs

    @override
    def inference(self, query: dict) -> dict:
        split: str = query["_split"][0]
        probs = torch.stack([self._probs[split][i] for i in query["inputs"]])
        return {"probs": probs}

    @override
    def train(
        self,
        num_epochs: int,
        train_loader: Iterable[dict],
        eval_loader: Iterable[dict] | None,
    ) -> dict:
        assert len(train_loader) == self._size
        return dict()

    @override
    def save_model(self, model_path: str) -> None:
        pass
