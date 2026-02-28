from typing import override

import torch
from torch import Tensor as T
import torchvision
from jaxtyping import jaxtyped, Int
from jaxtyping import Float as FP
from typeguard import typechecked

from ._model_interface import LOGGER, ModelInterface


@typechecked
class ResNetModel(ModelInterface):
    def __init__(self, config: dict):
        super().__init__(config=config)
        resnet_version: str = self.config["resnet_version"]
        assert resnet_version.startswith("resnet")
        backbone = torchvision.models.get_model(
            resnet_version, **self.config["model_kwargs"]
        )
        self._model: torchvision.models.ResNet = backbone.eval().to(
            device=self.device, dtype=self.dtype
        )

        optimizer_name: dict = self.config["optimizer_name"]
        optimizer_cls = getattr(torch.optim, optimizer_name)
        assert issubclass(optimizer_cls, torch.optim.Optimizer)
        self._optimizer = optimizer_cls(
            self._model.parameters(), **self.config["optimizer_kwargs"]
        )

    @jaxtyped(typechecker=typechecked)
    @override
    def inference(self, query: dict) -> dict:
        inputs: FP[T, "b 3 h w"] = query["inputs"].to(
            device=self.device, dtype=self.dtype
        )
        labels: Int[T, "b"] | None = query["labels"].to(device=self.device)
        logits: FP[T, "b c"] = self._model.forward(inputs)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return dict(logits=logits, loss=loss)

    @override
    def train(
        self,
        data_loader: torch.utils.data.DataLoader[dict],
        num_epochs: int,
    ) -> dict:
        self._model.train()
        train_epoch = list[float]()
        train_loss = list[float]()

        for epoch in range(num_epochs):
            for step, data in enumerate(data_loader, 1):
                outputs = self.inference(data)
                loss: FP[T, ""] = outputs["loss"]

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                loss = loss.item()
                epoch_step = step / len(data_loader) + epoch
                LOGGER.info(f"training epoch={epoch_step:.3f}/{num_epochs} {loss=:.4e}")
                train_loss.append(loss)
                train_epoch.append(epoch_step)

        self._model.eval()
        return dict(train_epoch=train_epoch, train_loss=train_loss)
