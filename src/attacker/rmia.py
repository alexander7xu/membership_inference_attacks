"""
@article{zarifzadeh2023low,
  title={Low-cost high-power membership inference attacks},
  author={Zarifzadeh, Sajjad and Liu, Philippe and Shokri, Reza},
  journal={arXiv preprint arXiv:2312.03262},
  year={2023}
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
def _get_prob(model: ModelInterface, query: dict) -> FP[T, "batch"]:
    outputs = model.inference(query)
    if "probs" in outputs:
        probs: FP[T, "batch class"] = outputs["probs"]
    else:
        probs: FP[T, "batch class"] = torch.softmax(outputs["logits"], -1)

    labels: Int[T, "batch"] = query["labels"].to(probs.device)
    probs: FP[T, "batch"] = torch.gather(probs, -1, labels[..., None]).squeeze(-1)
    return probs


@typechecked
class RmiaOfflineAttacker(AttackerInterface):
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
        indices = torch.randperm(len(self._shadow_dataset), generator=self._generator)
        self._population_data_loader = self._shadow_dataset.select(
            indices[: self.config["population_subset_size"]]
        ).make_loader(shuffle=False)

        self._shadow_models = list[ModelInterface]()
        pr_z = 0
        for i in range(1, self.config["num_shadow_models"] + 1):
            LOGGER.info(f"training shadow model {i}/{self.config['num_shadow_models']}")
            model, prob = self._train_shadow_models()
            self._shadow_models.append(model)
            pr_z = pr_z + prob
        population_prob_shadow = pr_z / len(self._shadow_models)

        population_prob_target = list[FP[T, "batch=_"]]()
        for query in self._population_data_loader:
            prob = _get_prob(self._target_model, query)
            population_prob_target.append(prob)
        population_prob_target = torch.cat(population_prob_target, 0)
        self._lr_population: FP[T, "S"] = population_prob_target / (
            population_prob_shadow + 1e-15
        )

    @jaxtyped(typechecker=typechecked)
    def _train_shadow_models(self):
        # randomly select half of the shadow dataset to train a shadow model
        indices = torch.randperm(len(self._shadow_dataset), generator=self._generator)[
            : len(self._shadow_dataset) // 2
        ]
        data_loader = self._shadow_dataset.select(indices).make_loader(shuffle=True)
        model = type(self._target_model)(self._target_model.config, None)
        model.train(self.config["shadow_model_training_epochs"], data_loader, None)

        # calculate probability of true class on population dataset using shadow model
        prob = list[FP[T, "b=_"]]()
        for query in self._population_data_loader:
            prob.append(_get_prob(model, query))
        prob = torch.cat(prob, 0)
        return model, prob

    @jaxtyped(typechecker=typechecked)
    def _calc_query_prob_distribution(self, query: dict) -> FP[T, "b"]:
        # calculate probability of true class using shadow models
        pr_out_x = 0
        for model in self._shadow_models:
            prob = _get_prob(model, query)
            pr_out_x = pr_out_x + prob
        pr_out_x = pr_out_x / len(self._shadow_models)
        scale_a = self.config["scale_a"]
        prob_target = 0.5 * ((1 + scale_a) * pr_out_x + (1 - scale_a))
        return prob_target

    @jaxtyped(typechecker=typechecked)
    @override
    def score(self, query: dict) -> dict:
        query_prob_shadow = self._calc_query_prob_distribution(query)
        query_prob_target = _get_prob(self._target_model, query)
        lr_target: FP[T, "b"] = query_prob_target / (query_prob_shadow + 1e-15)

        ratio: FP[T, "b S"] = lr_target[:, None] / self._lr_population[None]
        scores: FP[T, "b"] = (ratio > self.config["gamma"]).to(ratio).mean(-1)
        return dict(scores=scores)


@typechecked
class RmiaOnlineAttacker(AttackerInterface):
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
        indices = torch.randperm(len(self._shadow_dataset), generator=self._generator)
        self._population_data_loader = self._shadow_dataset.select(
            indices[: self.config["population_subset_size"]]
        ).make_loader(shuffle=False)

        population_prob_target = list[FP[T, "batch=_"]]()
        for query in self._population_data_loader:
            prob = _get_prob(self._target_model, query)
            population_prob_target.append(prob)
        self._population_prob_target: FP[T, "S"] = torch.cat(population_prob_target, 0)

    @jaxtyped(typechecker=typechecked)
    def _train_shadow_models(self, query: dict):
        # randomly select half of the shadow dataset to train a shadow model
        indices = torch.randperm(len(self._shadow_dataset), generator=self._generator)
        data_in_loader = self._shadow_dataset.select(
            indices[: len(self._shadow_dataset) // 2]
        ).make_loader(shuffle=True)
        data_out_loader = self._shadow_dataset.select(
            indices[len(self._shadow_dataset) // 2 :]
        ).make_loader(shuffle=True)

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
                last_data = [dict() for _ in range(data["labels"].shape[0])]
                for k, v in data.items():
                    for i, vv in enumerate(v):
                        last_data[i][k] = torch.cat([vv, query[k]], 0)
                yield torch.utils.data.default_collate(last_data)
        # fmt:on
        model = type(self._target_model)(self._target_model.config, None)
        model.train(
            self.config["shadow_model_training_epochs"], _HackDataInLoader(), None
        )
        prob_in: FP[T, "b"] = _get_prob(model, query)

        # Exclude the target example from the dataset
        model = type(self._target_model)(self._target_model.config, None)
        model.train(self.config["shadow_model_training_epochs"], data_out_loader, None)
        prob_out: FP[T, "b"] = _get_prob(model, query)

        # calculate probability of true class on population dataset using shadow model
        prob_population = list()
        for sample in self._population_data_loader:
            prob_population.append(_get_prob(model, sample))
        prob_population: FP[T, "b S"] = torch.cat(prob_population, 0)
        return prob_in, prob_out, prob_population

    @jaxtyped(typechecker=typechecked)
    @override
    def score(self, query: dict) -> dict:
        prob_in, prob_out, population_prob_shadow = 0, 0, 0
        for i in range(1, self.config["num_shadow_models"] + 1):
            LOGGER.info(f"training shadow model {i}/{self.config['num_shadow_models']}")
            p_in, p_out, p_population = self._train_shadow_models(query)
            prob_in = prob_in + p_in
            prob_out = prob_out + p_out
            population_prob_shadow = population_prob_shadow + p_population

        prob_in, prob_out, population_prob_shadow = (
            x / self.config["num_shadow_models"]
            for x in (prob_in, prob_out, population_prob_shadow)
        )

        query_prob_shadow = (prob_in + prob_out) / 2
        query_prob_target = _get_prob(self._target_model, query)
        lr_target: FP[T, "b"] = query_prob_target / (query_prob_shadow + 1e-15)
        lr_population: FP[T, "b S"] = self._population_prob_target[None] / (
            population_prob_shadow + 1e-15
        )

        ratios: FP[T, "b S"] = lr_target[:, None] / lr_population
        scores: FP[T, "b"] = (ratios > self.config["gamma"]).to(ratios).mean(-1)
        return dict(scores=scores, ratios=ratios)
