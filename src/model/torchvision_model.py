"""
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
"""

from collections.abc import Iterable
from pathlib import Path
from typing import override

import torch
import torchvision

from src.model.interface import LOGGER, ModelInterface, _ModelConfigBase
from src.utils.annotation import FP, Int, T, tensor_typechecked, typechecked


class TorchvisionModelConfig(_ModelConfigBase):
    model_version: str
    model_kwargs: dict
    optimizer_name: str
    optimizer_kwargs: dict


@typechecked
class TorchvisionModel(ModelInterface):
    config: TorchvisionModelConfig

    def __init__(
        self, config: TorchvisionModelConfig, trained_model_path: str | None, **_
    ):
        super().__init__(config=config, trained_model_path=trained_model_path)
        backbone = torchvision.models.get_model(
            self.config.model_version, **self.config.model_kwargs
        )
        self._model = backbone.eval().to(device=self.device, dtype=self.dtype)
        if self._trained_model_path is not None:
            state = torch.load(self._trained_model_path, map_location=self.device)
            self._model.load_state_dict(state)

        optimizer_name = self.config.optimizer_name
        optimizer_cls = getattr(torch.optim, optimizer_name)
        assert issubclass(optimizer_cls, torch.optim.Optimizer)
        self._optimizer = optimizer_cls(
            self._model.parameters(), **self.config.optimizer_kwargs
        )

    @tensor_typechecked
    @override
    def inference(self, query: dict) -> dict:
        inputs: FP[T, "b 3 h w"] = query["inputs"].to(
            device=self.device, dtype=self.dtype
        )
        labels: Int[T, "b"] | None = query.get("labels")
        if labels is not None:
            labels = labels.to(device=self.device)
        logits: FP[T, "b c"] = self._model(inputs)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return {"logits": logits, "loss": loss}

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

                loss_item = loss.item()
                epoch_step = step / len(train_loader) + epoch
                LOGGER.info(
                    "training epoch=%.3f/%s loss=%.4e",
                    epoch_step,
                    num_epochs,
                    loss_item,
                )
                train_epoch.append(epoch_step)
                train_loss.append(loss_item)

            if eval_loader is None:
                continue
            self._model.eval()
            sum_eval_loss = 0.0
            with torch.no_grad():
                for data in eval_loader:
                    sum_eval_loss += self.inference(data)["loss"].item()
            loss_item = sum_eval_loss / len(eval_loader)
            eval_epoch.append(float(epoch + 1))
            eval_loss.append(loss_item)
            self._model.train()
            LOGGER.info(
                "evaluation epoch=%s/%s loss=%.4e",
                epoch + 1,
                num_epochs,
                loss_item,
            )

        self._model.eval()
        result = {"train_epoch": train_epoch, "train_loss": train_loss}
        if eval_loader is not None:
            result.update({"eval_epoch": eval_epoch, "eval_loss": eval_loss})
        return result

    @override
    def save_model(self, model_path: str) -> None:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)
