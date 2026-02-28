import abc
from copy import deepcopy
import logging

from model import ModelInterface


LOGGER = logging.getLogger("attacker")
LOGGER.setLevel(logging.INFO)


class AttackerInterface(abc.ABC):
    def __init__(
        self,
        config: dict,
        target_model: ModelInterface,
    ):
        self.__config = deepcopy(config)
        self._target_model = target_model

    @property
    def config(self):
        return self.__config

    @abc.abstractmethod
    def score(self, query: dict) -> dict:
        pass
