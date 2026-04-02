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

from src.dataset.interface import DatasetInterface, _DatasetConfigBase


class TorchvisionDatasetConfig(_DatasetConfigBase):
    dataset_class: str
    dataset_kwargs: dict
    dataloader_kwargs: dict
    pass


class _WrapTorchvisionDataset:
    def __init__(
        self, dataset_class: type, indices: Iterable[int] | None, **dataset_kwargs
    ):
        self._dataset = dataset_class(**dataset_kwargs)
        if indices is None:
            indices = range(len(self._dataset))
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


class TorchvisionDataset(DatasetInterface):
    config: TorchvisionDatasetConfig

    def __init__(self, config: dict, *, _indices: Iterable[int] | None = None, **_):
        super().__init__(config)
        dataset_class = getattr(torchvision.datasets, self.config.dataset_class)
        self._dataset = _WrapTorchvisionDataset(
            dataset_class, _indices, **self.config.dataset_kwargs
        )

    @override
    def __len__(self) -> int:
        return len(self._dataset)

    @override
    def select(self, indices: Iterable[int]) -> "TorchvisionDataset":
        return TorchvisionDataset(self.config, _indices=indices)

    @override
    def make_loader(self, **overwrite_config) -> torch.utils.data.DataLoader:
        generator = torch.Generator("cpu").manual_seed(self.config.seed)
        config = self.config.dataloader_kwargs
        for k, v in overwrite_config.items():
            config[k] = v

        return torch.utils.data.DataLoader(
            self._dataset,
            generator=generator,
            **config,
        )
