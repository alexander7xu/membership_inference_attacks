import abc
import logging

from src.model import ModelInterface
from src.utils import ConfigBase

LOGGER = logging.getLogger("attacker")
LOGGER.setLevel(logging.INFO)


class _AttackerConfigBase(ConfigBase):
    seed: int


_registered_attacker_classes = dict[str, type["AttackerInterface"]]()


class AttackerInterface(abc.ABC):
    config: _AttackerConfigBase

    def __init_subclass__(cls):
        super().__init_subclass__()
        if cls.__name__.startswith("_"):
            return
        assert cls.__name__ not in _registered_attacker_classes
        _registered_attacker_classes[cls.__name__] = cls

    def __init__(
        self,
        config: _AttackerConfigBase,
        target_model: ModelInterface,
    ):
        self.config = config
        self._target_model = target_model

    @abc.abstractmethod
    def score(self, query: dict) -> dict:
        pass


def load_attacker(
    config: _AttackerConfigBase,
    target_model: ModelInterface,
    **attacker_kwargs,
) -> AttackerInterface:
    return _registered_attacker_classes[config.type](
        config, target_model, **attacker_kwargs
    )
