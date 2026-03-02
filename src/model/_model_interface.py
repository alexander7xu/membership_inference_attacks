import abc
from copy import deepcopy
import logging
from typing import Iterable

import torch
from typeguard import typechecked


LOGGER = logging.getLogger("model")


@typechecked
class ModelInterface(abc.ABC):
    def __init__(self, config: dict, trained_model_path: str | None):
        self.__config = deepcopy(config)
        self.__device = torch.device(self.config["device"])
        dtype = getattr(torch, self.config["dtype"])
        assert isinstance(dtype, torch.dtype)
        self.__dtype: torch.dtype = dtype
        self._trained_model_path = trained_model_path

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
        num_epochs: int,
        train_loader: Iterable[dict],
        eval_loader: Iterable[dict] | None,
    ) -> dict:
        pass

    @abc.abstractmethod
    def save_model(self, model_path: str) -> None:
        pass
