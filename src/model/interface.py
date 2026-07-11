import abc
import logging

import torch
from beartype.typing import Iterable

from src.utils import ConfigBase
from src.utils.annotation import typechecked

LOGGER = logging.getLogger("model")


class _ModelConfigBase(ConfigBase):
    device: str
    dtype: str


_registered_model_classes = dict[str, type["ModelInterface"]]()


@typechecked
class ModelInterface(abc.ABC):
    config: _ModelConfigBase

    def __init_subclass__(cls):
        super().__init_subclass__()
        if cls.__name__.startswith("_"):
            return
        assert cls.__name__ not in _registered_model_classes
        _registered_model_classes[cls.__name__] = cls

    def __init__(self, config: _ModelConfigBase, trained_model_path: str | None):
        self.config = config
        self.__device = torch.device(self.config.device)
        dtype = getattr(torch, self.config.dtype)
        assert isinstance(dtype, torch.dtype)
        self.__dtype: torch.dtype = dtype
        self._trained_model_path = trained_model_path

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

    def make_shadow(self) -> "ModelInterface":
        return type(self)(self.config, None)


def load_model(
    config: _ModelConfigBase,
    trained_model_path: str | None,
    **model_kwargs,
) -> ModelInterface:
    return _registered_model_classes[config.type](
        config, trained_model_path, **model_kwargs
    )
