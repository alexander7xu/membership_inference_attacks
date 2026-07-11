import pytest

from src.dataset.torchvision_dataset import TorchvisionDatasetConfig
from src.utils.config_class import make_config_from_dict


def test_make_config_from_dict_builds_registered_config():
    config = make_config_from_dict(
        {
            "type": "TorchvisionDataset",
            "seed": 42,
            "dataset_class": "CIFAR10",
            "dataset_kwargs": {
                "root": "data/dataset",
                "download": False,
                "train": True,
            },
            "dataloader_kwargs": {"batch_size": 2, "shuffle": False},
        }
    )

    assert isinstance(config, TorchvisionDatasetConfig)
    assert config.seed == 42


def test_make_config_from_dict_rejects_unknown_type():
    with pytest.raises(KeyError):
        make_config_from_dict({"type": "MissingConfig"})
