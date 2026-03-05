"""
@inproceedings{carlini2022membership,
  title={Membership inference attacks from first principles},
  author={Carlini, Nicholas and Chien, Steve and Nasr, Milad and Song, Shuang and Terzis, Andreas and Tramer, Florian},
  booktitle={2022 IEEE symposium on security and privacy (SP)},
  pages={1897--1914},
  year={2022},
  organization={IEEE}
}
"""

from typing import override

import torch
from torch import Tensor as T
from jaxtyping import jaxtyped, Int
from jaxtyping import Float as FP
from typeguard import typechecked

from ._attacker_interface import LOGGER, AttackerInterface, ModelInterface
from dataset import DatasetInterface


@jaxtyped(typechecker=typechecked)
def phi_stable(model: ModelInterface, query: dict) -> FP[T, "batch"]:
    outputs = model.inference(query)
    if "probs" in outputs:
        probs = outputs["probs"]
    else:
        probs: FP[T, "batch class"] = torch.softmax(outputs["logits"], -1)
    assert probs.isfinite().all()

    labels: Int[T, "batch"] = query["labels"].to(probs.device)
    probs_true: FP[T, "batch"] = torch.gather(probs, -1, labels[..., None]).squeeze(-1)
    probs_false: FP[T, "batch"] = probs.sum(-1) - probs_true
    phi = torch.log(probs_true) - torch.log(probs_false + 1e-9)
    return phi


@typechecked
class LiraOfflineAttacker(AttackerInterface):
    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        config: dict,
        target_model: ModelInterface,
        *,
        shadow_dataset: DatasetInterface,
        **_,
    ):
        super().__init__(config, target_model)
        self._shadow_dataset = shadow_dataset

        self._generator = torch.Generator("cpu")
        self._generator.manual_seed(self.config["seed"])
        self._shadow_models = list[ModelInterface]()
        for i in range(1, self.config["num_shadow_models"] + 1):
            LOGGER.info(f"training shadow model {i}/{self.config['num_shadow_models']}")
            model = self._train_shadow_models()
            self._shadow_models.append(model)

    @jaxtyped(typechecker=typechecked)
    def _train_shadow_models(self):
        # randomly select half of the shadow dataset to train a shadow model
        indices = torch.randperm(len(self._shadow_dataset), generator=self._generator)
        data_loader = self._shadow_dataset.select(
            indices[: len(self._shadow_dataset) // 2]
        ).make_loader(shuffle=True)
        model = type(self._target_model)(self._target_model.config, None)
        model.train(self.config["shadow_model_training_epochs"], data_loader, None)
        return model

    @jaxtyped(typechecker=typechecked)
    @override
    def score(self, query: dict) -> dict:
        loss_out = list[float]()
        for shadow_model in self._shadow_models:
            loss_out.append(phi_stable(shadow_model, query))
        loss_out = torch.stack(loss_out, -1)
        # FP[T, "b"]
        mean_out, std_out = loss_out.mean(-1), loss_out.std(-1)
        loss_target: FP[T, "b"] = phi_stable(self._target_model, query)
        scores = torch.distributions.Normal(loc=mean_out, scale=std_out).cdf(
            loss_target
        )
        return dict(scores=scores)


@typechecked
class LiraOnlineAttacker(AttackerInterface):
    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        config: dict,
        target_model: ModelInterface,
        *,
        shadow_dataset: DatasetInterface,
        **_,
    ):
        super().__init__(config, target_model)
        self._shadow_dataset = shadow_dataset

        self._generator = torch.Generator("cpu")
        self._generator.manual_seed(self.config["seed"])

    @jaxtyped(typechecker=typechecked)
    def _train_shadow_models(self, query: dict):
        # randomly select half of the shadow dataset to train a shadow model
        indices = torch.randperm(len(self._shadow_dataset), generator=self._generator)
        data_in_loader = self._shadow_dataset.select(
            indices[: len(self._shadow_dataset) // 2]
        ).make_loader(shuffle=True)

        batch_size = query["labels"].shape[0]
        used_samples = torch.randperm(batch_size, generator=self._generator)[
            : batch_size // 2
        ]

        raise NotImplementedError("!!!BUG HERE!!!")
        # Include the target example in the dataset
        # Hacky way to solve BUG https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
        # fmt:off
        class _HackDataInLoader:
            def __len__(self): return len(data_in_loader)
            def __iter__(self):
                it = iter(data_in_loader)
                last_data = next(it)
                for data in it:
                    yield last_data
                    last_data = data
                last_data = [dict() for _ in range(used_samples)]
                for k, v in data.items():
                    for i, vv in enumerate(v):
                        last_data[i][k] = vv
                yield torch.utils.data.default_collate(last_data)
        # fmt:on
        model = type(self._target_model)(self._target_model.config, None)
        model.train(
            self.config["shadow_model_training_epochs"], _HackDataInLoader(), None
        )
        loss_in: FP[T, "b"] = phi_stable(model, query)

        # Exclude the target example from the dataset
        model = type(self._target_model)(self._target_model.config, None)
        model.train(self.config["shadow_model_training_epochs"], data_out_loader, None)
        loss_out: FP[T, "b"] = phi_stable(model, query)

        return loss_in, loss_out

    @jaxtyped(typechecker=typechecked)
    @override
    def score(self, query: dict) -> dict:
        loss_in, loss_out = list[float](), list[float]()
        for i in range(1, self.config["num_shadow_models"] + 1):
            LOGGER.info(f"training shadow model {i}/{self.config['num_shadow_models']}")
            l_in, l_out = self._train_shadow_models(query)
            loss_in.append(l_in)
            loss_out.append(l_out)
        loss_in, loss_out = torch.stack(loss_in, -1), torch.stack(loss_out, -1)
        # FP[T, "b"]
        mean_in, std_in = loss_in.mean(-1), loss_in.std(-1)
        mean_out, std_out = loss_out.mean(-1), loss_out.std(-1)

        loss_target: FP[T, "b"] = phi_stable(self._target_model, query)
        logp_in = torch.distributions.Normal(loc=mean_in, scale=std_in).log_prob(
            loss_target
        )
        logp_out = torch.distributions.Normal(loc=mean_out, scale=std_out).log_prob(
            loss_target
        )
        log_ratios = logp_in - logp_out

        # scores = torch.exp(log_ratios)
        return dict(scores=log_ratios)
