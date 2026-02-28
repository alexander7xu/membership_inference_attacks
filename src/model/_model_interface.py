import abc
from copy import deepcopy
import logging

import torch
from typeguard import typechecked


LOGGER = logging.getLogger("model")


@typechecked
class ModelInterface(abc.ABC):
    def __init__(self, config: dict):
        self.__config = deepcopy(config)
        self.__device = torch.device(self.config["device"])
        dtype = getattr(torch, self.config["dtype"])
        assert isinstance(dtype, torch.dtype)
        self.__dtype: torch.dtype = dtype

    @property
    def config(self):
        return self.__config

    @property
    def device(self):
        return self.__device

    @property
    def dtype(self):
        return self.__dtype

    @abc.abstractmethod
    def inference(self, query: dict) -> dict:
        pass

    @abc.abstractmethod
    def train(
        self,
        data_loader: torch.utils.data.DataLoader,
        num_epochs: int,
    ) -> dict:
        pass
