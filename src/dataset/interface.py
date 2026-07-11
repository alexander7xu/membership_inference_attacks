import abc

import torch
from beartype.typing import Iterable

from src.utils import ConfigBase
from src.utils.annotation import typechecked


class _DatasetConfigBase(ConfigBase):
    seed: int


_registered_dataset_classes = dict[str, type["DatasetInterface"]]()


@typechecked
class DatasetInterface(abc.ABC):
    config: _DatasetConfigBase

    def __init_subclass__(cls):
        super().__init_subclass__()
        if cls.__name__.startswith("_"):
            return
        assert cls.__name__ not in _registered_dataset_classes
        _registered_dataset_classes[cls.__name__] = cls

    def __init__(self, config: _DatasetConfigBase):
        self.config = config

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def select(self, indices: Iterable[int]) -> "DatasetInterface":
        pass

    @abc.abstractmethod
    def make_loader(self, **overwrite_config) -> torch.utils.data.DataLoader:
        pass


def load_dataset(
    config: _DatasetConfigBase,
    **dataset_kwargs,
) -> DatasetInterface:
    return _registered_dataset_classes[config.type](config, **dataset_kwargs)
