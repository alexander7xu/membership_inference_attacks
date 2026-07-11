import torch
import torchvision

from src.dataset.torchvision_dataset import TorchvisionDataset, TorchvisionDatasetConfig


class _TinyVisionDataset:
    def __init__(self, **_):
        self.samples = [
            (torch.zeros(3, 4, 4), 0),
            (torch.ones(3, 4, 4), 1),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_make_loader_override_does_not_mutate_config(monkeypatch):
    monkeypatch.setattr(torchvision.datasets, "CIFAR10", _TinyVisionDataset)
    monkeypatch.setattr(torchvision.transforms, "ToTensor", lambda: lambda value: value)

    config = TorchvisionDatasetConfig(
        type="TorchvisionDataset",
        seed=123,
        dataset_class="CIFAR10",
        dataset_kwargs={"root": "unused", "download": False, "train": True},
        dataloader_kwargs={
            "batch_size": 2,
            "num_workers": 0,
            "persistent_workers": True,
            "shuffle": False,
        },
    )
    dataset = TorchvisionDataset(config)

    loader = dataset.make_loader(batch_size=1)
    batch = next(iter(loader))

    assert batch["inputs"].shape == (1, 3, 4, 4)
    assert config.dataloader_kwargs["batch_size"] == 2
    assert config.dataloader_kwargs["persistent_workers"] is True
