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
        ).make_loader(False)

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
            prob = self._get_prob(self._target_model, query)
            population_prob_target.append(prob)
        population_prob_target = torch.cat(population_prob_target, 0)
        self._lr_population: FP[T, "ShadowData"] = population_prob_target / (
            population_prob_shadow + 1e-15
        )

    @jaxtyped(typechecker=typechecked)
    @torch.no_grad
    def _get_prob(self, model: ModelInterface, query: dict) -> FP[T, "batch"]:
        outputs = model.inference(query)
        logits: FP[T, "batch class"] = outputs["logits"]
        labels: Int[T, "batch"] = query["labels"].to(logits.device)
        prob = torch.softmax(logits, -1)
        prob = torch.gather(prob, -1, labels[..., None]).squeeze(-1)
        return prob

    @jaxtyped(typechecker=typechecked)
    def _train_shadow_models(self):
        # randomly select half of the shadow dataset to train a shadow model
        indices = torch.randperm(len(self._shadow_dataset), generator=self._generator)[
            : len(self._shadow_dataset) // 2
        ]
        data_loader = self._shadow_dataset.select(indices).make_loader(True)
        model = type(self._target_model)(self._target_model.config)
        model.train(data_loader, self.config["shadow_model_training_epochs"])

        # calculate probability of true class on population dataset using shadow model
        prob = list[FP[T, "b=_"]]()
        for query in self._population_data_loader:
            prob.append(self._get_prob(model, query))
        prob = torch.cat(prob, 0)
        return model, prob

    @jaxtyped(typechecker=typechecked)
    def _calc_query_prob_distribution(self, query: dict) -> FP[T, "b"]:
        # calculate probability of true class using shadow models
        pr_out_x = 0
        for model in self._shadow_models:
            prob = self._get_prob(model, query)
            pr_out_x = pr_out_x + prob
        pr_out_x = pr_out_x / len(self._shadow_models)
        scale_a = self.config["scale_a"]
        prob_target = 0.5 * ((1 + scale_a) * pr_out_x + (1 - scale_a))
        return prob_target

    @jaxtyped(typechecker=typechecked)
    @override
    def score(self, query: dict) -> dict:
        query_prob_shadow = self._calc_query_prob_distribution(query)
        query_prob_target = self._get_prob(self._target_model, query)
        lr_target: FP[T, "b"] = query_prob_target / (query_prob_shadow + 1e-15)

        ratio: FP[T, "b S"] = lr_target[:, None] / self._lr_population[None]
        scores: FP[T, "b"] = (ratio > self.config["gamma"]).to(ratio).mean(-1)
        return dict(scores=scores)
