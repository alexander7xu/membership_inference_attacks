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

import torch
from typing_extensions import override

from src.attacker.interface import (
    LOGGER,
    AttackerInterface,
    ModelInterface,
    _AttackerConfigBase,
)
from src.dataset import DatasetInterface
from src.utils.annotation import FP, Int, T, tensor_typechecked, typechecked


@torch.no_grad
@tensor_typechecked
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


class LiraOfflineAttackerConfig(_AttackerConfigBase):
    num_shadow_models: int
    shadow_model_training_epochs: int


@typechecked
class LiraOfflineAttacker(AttackerInterface):
    config: LiraOfflineAttackerConfig

    @tensor_typechecked
    def __init__(
        self,
        config: LiraOfflineAttackerConfig,
        target_model: ModelInterface,
        *,
        shadow_dataset: DatasetInterface,
        **_,
    ):
        super().__init__(config, target_model)
        self._shadow_dataset = shadow_dataset

        self._generator = torch.Generator("cpu").manual_seed(self.config.seed)
        self._shadow_models = list[ModelInterface]()
        for i in range(1, self.config.num_shadow_models + 1):
            LOGGER.info(f"training shadow model {i}/{self.config.num_shadow_models}")
            model = self._train_shadow_model()
            self._shadow_models.append(model)

    @tensor_typechecked
    def _train_shadow_model(self):
        # randomly select half of the shadow dataset to train a shadow model
        indices = torch.randperm(len(self._shadow_dataset), generator=self._generator)
        data_loader = self._shadow_dataset.select(
            indices[: len(self._shadow_dataset) // 2]
        ).make_loader(shuffle=True)
        model = self._target_model.make_shadow()
        model.train(self.config.shadow_model_training_epochs, data_loader, None)
        return model

    @tensor_typechecked
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


class LiraOnlineAttackerConfig(_AttackerConfigBase):
    num_shadow_models: int
    shadow_model_training_epochs: int


@typechecked
class LiraOnlineAttacker(AttackerInterface):
    config: LiraOnlineAttackerConfig

    @tensor_typechecked
    def __init__(
        self,
        config: LiraOnlineAttackerConfig,
        target_model: ModelInterface,
        *,
        shadow_dataset: DatasetInterface,
        **_,
    ):
        super().__init__(config, target_model)
        self._shadow_dataset = shadow_dataset

        self._generator = torch.Generator("cpu").manual_seed(self.config.seed)

    @tensor_typechecked
    def _train_shadow_models(self, query: dict):
        # randomly select half of the shadow dataset to train a shadow model
        indices = torch.randperm(len(self._shadow_dataset), generator=self._generator)
        data_in_loader = self._shadow_dataset.select(
            indices[: len(self._shadow_dataset) // 2]
        ).make_loader(shuffle=True)
        data_out_loader = self._shadow_dataset.select(
            indices[len(self._shadow_dataset) // 2 :]
        ).make_loader(shuffle=True)

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
                last_data = dict()
                for k, v in data.items():
                    last_data[k] = torch.cat([v, query[k]], 0)
                yield last_data
        # fmt:on
        model = self._target_model.make_shadow()
        model.train(self.config.shadow_model_training_epochs, _HackDataInLoader(), None)
        loss_in: FP[T, "b"] = phi_stable(model, query)

        # Exclude the target example from the dataset
        model = self._target_model.make_shadow()
        model.train(self.config.shadow_model_training_epochs, data_out_loader, None)
        loss_out: FP[T, "b"] = phi_stable(model, query)

        return loss_in, loss_out

    @tensor_typechecked
    @override
    def score(self, query: dict) -> dict:
        loss_in, loss_out = list[float](), list[float]()
        for i in range(1, self.config.num_shadow_models + 1):
            LOGGER.info(f"training shadow model {i}/{self.config.num_shadow_models}")
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


class LiraOnlineAttackerV2Config(_AttackerConfigBase):
    num_shadow_models: int
    shadow_model_training_epochs: int


@typechecked
class LiraOnlineAttackerV2(AttackerInterface):
    config: LiraOnlineAttackerV2Config

    @tensor_typechecked
    def __init__(
        self,
        config: LiraOnlineAttackerV2Config,
        target_model: ModelInterface,
        *,
        shadow_dataset: DatasetInterface,
        **_,
    ):
        super().__init__(config, target_model)
        self._shadow_dataset = shadow_dataset
        assert self.config.num_shadow_models >= 4

        self._generator = torch.Generator("cpu").manual_seed(self.config.seed)

    @tensor_typechecked
    def _train_shadow_models(self, query: dict, query_permutation: int):
        # randomly select half of the shadow dataset
        indices = torch.randperm(len(self._shadow_dataset), generator=self._generator)
        data_loader = self._shadow_dataset.select(
            indices[: len(self._shadow_dataset) // 2]
        ).make_loader(shuffle=True)

        # randomly select half of the query
        batch_size = query["labels"].shape[0]
        assert batch_size > 1
        assert query_permutation >= 0
        if query_permutation > 3:
            query_indices = torch.randperm(batch_size, generator=self._generator)
            indices_used = query_indices[: batch_size // 2].tolist()
            indices_unused = query_indices[batch_size // 2 :].tolist()
        else:
            query_indices = torch.arange(batch_size).tolist()
            quarters = [query_indices[i::4] for i in range(4)]
            # 0: [[0, 2], [1, 3]]; 1: [[1, 2], [1,3]]; 2: [[0, 3], [1, 2]]; 3: [[1, 3], [0, 2]]
            indices_used = (
                quarters[query_permutation & 1] + quarters[query_permutation >> 1 | 2]
            )
            indices_unused = (
                quarters[1 - (query_permutation & 1)]
                + quarters[3 - (query_permutation >> 1)]
            )

        used_query = {
            k: torch.stack([v[i] for i in indices_used], 0) for k, v in query.items()
        }
        unused_query = {
            k: torch.stack([v[i] for i in indices_unused], 0) for k, v in query.items()
        }

        # Include the target example in the dataset
        # fmt:off
        class _HackDataInLoader:
            def __len__(self): return len(data_loader) + 1
            def __iter__(self):
                yield from iter(data_loader)
                data = list[dict]()
                for i in range(len(indices_used)):
                    data.append(dict())
                    for k, v in used_query.items():
                        data[-1][k] = v[i]
                yield torch.utils.data.default_collate(data)
        # fmt:on

        model = self._target_model.make_shadow()
        model.train(self.config.shadow_model_training_epochs, _HackDataInLoader(), None)
        loss_used: FP[T, "b"] = phi_stable(model, used_query)
        loss_unused: FP[T, "b"] = phi_stable(model, unused_query)

        return loss_used, loss_unused, indices_used, indices_unused

    @tensor_typechecked
    @override
    def score(self, query: dict) -> dict:
        sums = torch.zeros(
            query["labels"].shape[0], 4, device=self._target_model.device
        )
        cnts_in = torch.zeros(sums.shape[0], dtype=int, device=sums.device)
        for i in range(1, self.config.num_shadow_models + 1):
            LOGGER.info(f"training shadow model {i}/{self.config.num_shadow_models}")
            l_in, l_out, idx_in, idx_out = self._train_shadow_models(query, i - 1)
            cnts_in[idx_in] += 1
            sums[idx_in, 0] += l_in
            sums[idx_in, 1] += l_in**2
            sums[idx_out, 2] += l_out
            sums[idx_out, 3] += l_out**2
        # FP[T, "b"]
        cnts_out = self.config.num_shadow_models - cnts_in
        assert torch.all(cnts_in > 0) and torch.all(cnts_out > 0)
        mean_in = sums[:, 0] / cnts_in
        mean_out = sums[:, 2] / cnts_out
        std_in = torch.sqrt(sums[:, 1] / cnts_in - mean_in**2)
        std_out = torch.sqrt(sums[:, 3] / cnts_out - mean_out**2)

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
