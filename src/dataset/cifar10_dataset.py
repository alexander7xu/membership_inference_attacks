"""
@misc{krizhevsky2009learning,
  title={Learning multiple layers of features from tiny images.(2009)},
  author={Krizhevsky, Alex and Hinton, Geoffrey and others},
  year={2009}
}
"""


from typing import override, Iterable

import torch
import torchvision

from ._dataset_interface import DatasetInterface


class _WrapTorchvisionDatasetCIFAR10:
    def __init__(self, indices: Iterable[int] | None, **dataset_kwargs):
        self._dataset = torchvision.datasets.CIFAR10(**dataset_kwargs)
        if indices is None:
            indices = list(range(len(self._dataset)))
        self._indices = list(indices)
        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]) # fmt:skip

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self._indices[idx]
        img, target = self._dataset[real_idx]
        inputs = self._transform(img)
        return dict(inputs=inputs, labels=target)


class Cifar10Dataset(DatasetInterface):
    def __init__(self, config: dict, *, _indices: Iterable[int] | None = None):
        super().__init__(config)
        self._dataset = _WrapTorchvisionDatasetCIFAR10(
            _indices, **self.config["dataset_kwargs"]
        )

    @override
    def __len__(self) -> int:
        return len(self._dataset)

    @override
    def select(self, indices: Iterable[int]) -> "Cifar10Dataset":
        return Cifar10Dataset(self.config, _indices=indices)

    @override
    def make_loader(self, **overwrite_config) -> torch.utils.data.DataLoader:
        generator = torch.Generator("cpu")
        generator.manual_seed(self.config["seed"])
        config = dict(self.config["loader"])
        for k, v in overwrite_config.items():
            config[k] = v

        return torch.utils.data.DataLoader(
            self._dataset,
            generator=generator,
            **config,
        )
