import dataclasses
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from beartype.typing import Iterable

from src.utils.annotation import typechecked

type _ConfigLegalValueType = Any

_registered_config_classes = dict[str, type]()


@typechecked
@dataclasses.dataclass(init=True, order=True, frozen=True, kw_only=True)
class ConfigBase:
    type: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dataclasses.dataclass(cls, init=True, order=True, frozen=True, kw_only=True)
        if not cls.__name__.startswith("_"):
            assert cls.__name__.endswith("Config")
            _registered_config_classes[cls.__name__[:-6]] = cls

    def to_dict(self) -> dict[str, _ConfigLegalValueType]:
        return dataclasses.asdict(self)

    def to_yaml(self, with_root_key: str = "") -> str:
        dict_ = self.to_dict()
        if with_root_key:
            assert with_root_key.isidentifier()
            dict_ = {with_root_key: dict_}
        return yaml.safe_dump(dict_, sort_keys=True)

    def fingerprint(self) -> int:
        text = self.to_yaml()
        bytes_data = text.encode("utf-8")
        sha256_value = int.from_bytes(hashlib.sha256(bytes_data).digest(), "big")
        return sha256_value


@typechecked
def make_config_from_dict(data: dict[str, _ConfigLegalValueType]) -> ConfigBase:
    def _recur(node):
        cls = _registered_config_classes[node["type"]]
        node = node.copy()
        assert issubclass(cls, ConfigBase)
        for name, field in cls.__dataclass_fields__.items():
            if type(field.type) is type and issubclass(field.type, ConfigBase):
                node[name] = make_config_from_dict(node[name])
        return typechecked(cls)(**node)

    return _recur(deepcopy(data))


@typechecked
def load_configs_from_yaml_files(paths: Iterable[Path | str]) -> dict[str, ConfigBase]:
    merged_data = dict[str, dict[str, _ConfigLegalValueType]]()
    for path in paths:
        with open(path, encoding="utf-8") as file:
            data = yaml.safe_load(file)
        assert len(merged_data.keys() & data.keys()) == 0
        merged_data.update(data)
    results = dict[str, ConfigBase]()
    for key, value in merged_data.items():
        results[key] = make_config_from_dict(value)
    return results
