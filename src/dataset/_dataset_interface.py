import abc
from copy import deepcopy
from typing import Iterable

from typeguard import typechecked
import torch


@typechecked
class DatasetInterface(abc.ABC):
    def __init__(self, config: dict):
        self.__config = deepcopy(config)

    @property
    def config(self):
        return self.__config

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def select(self, indices: Iterable[int]) -> "DatasetInterface":
        pass

    @abc.abstractmethod
    def make_loader(self, shuffle: bool) -> torch.utils.data.DataLoader:
        pass
