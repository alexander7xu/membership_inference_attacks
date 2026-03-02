"""
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
"""


from typing import override, Iterable

import torch
from torch import Tensor as T
import torchvision
from jaxtyping import jaxtyped, Int
from jaxtyping import Float as FP
from typeguard import typechecked

from ._model_interface import LOGGER, ModelInterface


@typechecked
class ResNetModel(ModelInterface):
    def __init__(self, config: dict, trained_model_path: str | None):
        super().__init__(config=config, trained_model_path=trained_model_path)
        resnet_version: str = self.config["resnet_version"]
        assert resnet_version.startswith("resnet")
        backbone = torchvision.models.get_model(
            resnet_version, **self.config["model_kwargs"]
        )
        self._model: torchvision.models.ResNet = backbone.eval().to(
            device=self.device, dtype=self.dtype
        )
        if self._trained_model_path is not None:
            self._model.load_state_dict(torch.load(self._trained_model_path))

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
        num_epochs: int,
        train_loader: Iterable[dict],
        eval_loader: Iterable[dict] | None,
    ) -> dict:
        self._model.train()
        train_epoch, train_loss = (list[float]() for _ in range(2))
        eval_epoch, eval_loss = (list[float]() for _ in range(2))
        for epoch in range(num_epochs):
            for step, data in enumerate(train_loader, 1):
                outputs = self.inference(data)
                loss: FP[T, ""] = outputs["loss"]

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                loss = loss.item()
                epoch_step = step / len(train_loader) + epoch
                LOGGER.info(f"training epoch={epoch_step:.3f}/{num_epochs} {loss=:.4e}")
                train_epoch.append(epoch_step)
                train_loss.append(loss)

            if eval_loader is None:
                continue
            self._model.eval()
            sum_eval_loss = 0.0
            for step, data in enumerate(eval_loader, 1):
                sum_eval_loss += self.inference(data)["loss"].item()
            loss = sum_eval_loss / len(eval_loader)
            eval_epoch.append(float(epoch+1))
            eval_loss.append(loss)
            self._model.train()
            LOGGER.info(f"evaluation epoch={epoch+1}/{num_epochs} {loss=:.4e}")

        self._model.eval()
        result = dict(train_epoch=train_epoch, train_loss=train_loss)
        if eval_loader is not None:
            result |= dict(eval_epoch=eval_epoch, eval_loss=eval_loss)
        return result

    @override
    def save_model(self, model_path: str) -> None:
        torch.save(self._model.state_dict(), model_path)
