import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.attacker import AttackerInterface, load_attacker
from src.dataset import DatasetInterface, load_dataset
from src.evaluator import learning_curve, roc
from src.experiment import (
    collect_environment,
    collect_git_state,
    config_fingerprint,
    seed_everything,
    write_experiment_markdown,
    write_yaml,
)
from src.model import ModelInterface, load_model
from src.utils import make_config_from_dict

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _plain_dict(node: Any) -> dict:
    return OmegaConf.to_container(node, resolve=True)


def _project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _dataset_config(cfg: DictConfig, *, train: bool, shuffle: bool) -> dict:
    data = deepcopy(_plain_dict(cfg.data.dataset_config))
    data["seed"] = int(cfg.runtime.seed)
    data["dataset_kwargs"] = dict(data["dataset_kwargs"])
    data["dataloader_kwargs"] = dict(data["dataloader_kwargs"])
    data["dataset_kwargs"]["train"] = train
    data["dataloader_kwargs"]["shuffle"] = shuffle
    return data


def _load_dataset(cfg: DictConfig, *, train: bool, shuffle: bool) -> DatasetInterface:
    return load_dataset(
        make_config_from_dict(_dataset_config(cfg, train=train, shuffle=shuffle))
    )


def _model_config(cfg: DictConfig) -> dict:
    data = deepcopy(_plain_dict(cfg.model.config))
    data["optimizer_name"] = str(cfg.optimizer.name)
    data["optimizer_kwargs"] = _plain_dict(cfg.optimizer.kwargs)
    return data


def _load_model(cfg: DictConfig, trained_model_path: Path | None) -> ModelInterface:
    path = str(trained_model_path) if trained_model_path is not None else None
    return load_model(make_config_from_dict(_model_config(cfg)), path)


def _attacker_config(cfg: DictConfig) -> dict:
    data = deepcopy(_plain_dict(cfg.attack.config))
    data["seed"] = int(cfg.runtime.seed)
    if "shadow_model_training_epochs" in data:
        data["shadow_model_training_epochs"] = int(cfg.train.shadow_epochs)
    return data


def load_or_train_target_model(
    cfg: DictConfig, *, project_root: Path, run_dir: Path
) -> tuple[ModelInterface, dict, dict[str, str]]:
    model_path = _project_path(project_root, cfg.checkpoint.target_model_path)
    artifacts: dict[str, str] = {}
    if model_path.exists() and not bool(cfg.checkpoint.force_retrain):
        target_model = _load_model(cfg, model_path)
        artifacts["target_checkpoint"] = str(model_path)
        return target_model, {}, artifacts

    seed_everything(
        int(cfg.runtime.seed), deterministic=bool(cfg.runtime.deterministic)
    )
    target_model = _load_model(cfg, None)
    train_set = _load_dataset(cfg, train=True, shuffle=True)
    val_set = _load_dataset(cfg, train=False, shuffle=False)

    history = target_model.train(
        int(cfg.train.target_epochs),
        train_set.make_loader(),
        val_set.make_loader(),
    )
    figure_path = run_dir / "figures" / str(cfg.report.learning_curve_filename)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig = learning_curve(history)
    fig.savefig(figure_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    target_model.save_model(str(model_path))
    artifacts["learning_curve"] = str(figure_path)
    artifacts["target_checkpoint"] = str(model_path)
    return target_model, history, artifacts


def make_attack_sets(
    cfg: DictConfig, train_set: DatasetInterface, val_set: DatasetInterface
) -> tuple[DatasetInterface, DatasetInterface, DatasetInterface]:
    generator = torch.Generator("cpu").manual_seed(int(cfg.runtime.seed))
    num_measurement = int(cfg.data.measurement_samples)
    train_indices = torch.randperm(len(train_set), generator=generator)[
        :num_measurement
    ].tolist()
    val_indices = torch.randperm(len(val_set), generator=generator)
    measurement_train = train_set.select(train_indices)
    measurement_val = val_set.select(val_indices[:num_measurement].tolist())

    available_shadow_samples = cfg.data.available_shadow_samples
    if available_shadow_samples is None:
        end = len(val_set)
    else:
        end = min(len(val_set), num_measurement + int(available_shadow_samples))
    shadow_set = val_set.select(val_indices[num_measurement:end].tolist())
    return measurement_train, measurement_val, shadow_set


def perform_attack(
    cfg: DictConfig,
    attacker: AttackerInterface,
    measurement_train: DatasetInterface,
    measurement_val: DatasetInterface,
) -> tuple[list[float], list[int]]:
    scores = list[float]()
    attack_batch_size = int(cfg.eval.attack_batch_size)
    for query in measurement_train.make_loader(
        batch_size=attack_batch_size, num_workers=1
    ):
        scores += attacker.score(query)["scores"].detach().cpu().tolist()
    for query in measurement_val.make_loader(
        batch_size=attack_batch_size, num_workers=1
    ):
        scores += attacker.score(query)["scores"].detach().cpu().tolist()

    member_label = int(cfg.data.member_positive_label)
    nonmember_label = 1 - member_label
    references = [member_label] * len(measurement_train) + [nonmember_label] * len(
        measurement_val
    )
    return scores, references


def _maybe_start_wandb(cfg: DictConfig, run_dir: Path):
    if not bool(cfg.wandb.enabled):
        return None
    try:
        import wandb

        return wandb.init(
            project=str(cfg.wandb.project),
            entity=cfg.wandb.entity,
            mode=str(cfg.wandb.mode),
            tags=list(cfg.wandb.tags),
            dir=str(run_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    except Exception as exc:
        LOGGER.warning("W&B run was not created: %r", exc)
        return None


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="attack_resnet_cifar10",
)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    project_root = Path.cwd()
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    resolved_config_path = run_dir / str(cfg.report.resolved_config_filename)
    write_yaml(resolved_config_path, resolved_config)
    fingerprint = config_fingerprint(resolved_config)

    seed_everything(
        int(cfg.runtime.seed), deterministic=bool(cfg.runtime.deterministic)
    )
    wandb_run = _maybe_start_wandb(cfg, run_dir)

    train_set = _load_dataset(cfg, train=True, shuffle=True)
    val_set = _load_dataset(cfg, train=False, shuffle=False)
    target_model, history, artifacts = load_or_train_target_model(
        cfg, project_root=project_root, run_dir=run_dir
    )
    measurement_train, measurement_val, shadow_set = make_attack_sets(
        cfg, train_set, val_set
    )
    LOGGER.info("Number of shadow samples: %s", len(shadow_set))

    attacker = load_attacker(
        make_config_from_dict(_attacker_config(cfg)),
        target_model,
        shadow_dataset=shadow_set,
    )
    scores, references = perform_attack(
        cfg, attacker, measurement_train, measurement_val
    )

    scores_path = run_dir / "scores" / str(cfg.report.scores_filename)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_path, "w", encoding="utf-8") as file:
        file.write("score,reference\n")
        for score, reference in zip(scores, references, strict=True):
            file.write(f"{score},{reference}\n")

    fig, auc_score = roc(references, scores)
    roc_path = run_dir / "figures" / str(cfg.report.roc_filename)
    roc_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(roc_path)
    metrics = {"auc": float(auc_score)}
    artifacts.update(
        {
            "resolved_config": str(resolved_config_path),
            "attack_scores": str(scores_path),
            "roc_figure": str(roc_path),
        }
    )

    if wandb_run is not None:
        wandb_run.log(metrics)
        for name, path in artifacts.items():
            wandb_run.log({f"artifact_path/{name}": path})

    record_path = run_dir / str(cfg.report.experiment_filename)
    command = " ".join([sys.executable, *sys.argv])
    overrides = HydraConfig.get().overrides.task
    write_experiment_markdown(
        record_path,
        purpose=str(cfg.experiment.purpose),
        hypothesis=str(cfg.experiment.hypothesis),
        command=command,
        overrides=list(overrides),
        config_fingerprint_value=fingerprint,
        git_state=collect_git_state(project_root),
        environment=collect_environment(project_root),
        artifacts=artifacts,
        metrics=metrics,
        conclusion=(
            f"{cfg.attack.name} completed with member-positive AUC {auc_score:.4f}."
        ),
        achieved_purpose=True,
        next_action=str(cfg.experiment.next_action),
        wandb_run_id=getattr(wandb_run, "id", None),
    )
    if wandb_run is not None:
        wandb_run.finish()

    LOGGER.info("Results saved into %s", run_dir)
    LOGGER.info("auc=%.4f", auc_score)


if __name__ == "__main__":
    main()
