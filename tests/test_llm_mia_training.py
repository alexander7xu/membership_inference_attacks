from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf

from src.llm_mia import hf, workflow


def _training_cfg() -> Any:
    return OmegaConf.create(
        {
            "runtime": {"seed": 42},
            "train": {
                "epochs": 10,
                "per_device_train_batch_size": 16,
                "per_device_eval_batch_size": 16,
                "gradient_accumulation_steps": 2,
                "learning_rate": 1.0e-4,
                "warmup_ratio": 0.03,
                "weight_decay": 0.0,
                "logging_steps": 10,
                "dataloader_num_workers": 0,
                "dataloader_pin_memory": True,
            },
            "precision": {"bf16": False},
            "checkpoint": {"save_strategy": "final_adapter_only"},
            "finetuning": {"gradient_checkpointing": False},
            "tokenizer": {"max_length": 1024, "preprocessing_workers": 1},
        }
    )


def test_training_arguments_disable_evaluation_and_checkpoint_selection() -> None:
    args = hf.training_arguments(
        "trainer",
        _training_cfg(),
        max_steps=None,
    )

    assert args.eval_strategy.value == "no"
    assert args.save_strategy.value == "no"
    assert args.load_best_model_at_end is False


def test_training_arguments_enable_epoch_evaluation_without_checkpoints() -> None:
    args = hf.training_arguments(
        "trainer",
        _training_cfg(),
        max_steps=None,
        evaluate_each_epoch=True,
    )

    assert args.eval_strategy.value == "epoch"
    assert args.save_strategy.value == "no"
    assert args.load_best_model_at_end is False


def test_make_trainer_builds_named_completion_only_eval_datasets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_calls: list[list[object]] = []
    arguments_calls: list[bool] = []

    class FakeDataset:
        def __init__(
            self,
            records: list[object],
            tokenizer: object,
            max_length: int,
            preprocessing_workers: int,
        ) -> None:
            del tokenizer, max_length, preprocessing_workers
            self.records = records
            dataset_calls.append(records)

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    def fake_training_arguments(
        output_dir: str,
        cfg: object,
        *,
        max_steps: int | None,
        seed: int | None = None,
        evaluate_each_epoch: bool = False,
    ) -> object:
        del output_dir, cfg, max_steps, seed
        arguments_calls.append(evaluate_each_epoch)
        return object()

    monkeypatch.setattr(hf, "CompletionOnlyDataset", FakeDataset)
    monkeypatch.setattr(hf, "CompletionOnlyCollator", lambda tokenizer: tokenizer)
    monkeypatch.setattr(hf, "Trainer", FakeTrainer)
    monkeypatch.setattr(hf, "training_arguments", fake_training_arguments)

    train_records = [object()]
    squad_records = [object(), object()]
    trivia_records = [object()]
    trainer = hf.make_trainer(
        model=object(),
        tokenizer=object(),
        train_records=train_records,
        eval_records={
            "squad_validation": squad_records,
            "trivia_validation": trivia_records,
        },
        cfg=_training_cfg(),
        output_dir="trainer",
        max_steps=None,
    )

    assert arguments_calls == [True]
    assert dataset_calls == [train_records, squad_records, trivia_records]
    assert set(trainer.kwargs["eval_dataset"]) == {
        "squad_validation",
        "trivia_validation",
    }


def test_make_trainer_without_eval_records_keeps_evaluation_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments_calls: list[bool] = []

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    def fake_training_arguments(
        output_dir: str,
        cfg: object,
        *,
        max_steps: int | None,
        seed: int | None = None,
        evaluate_each_epoch: bool = False,
    ) -> object:
        del output_dir, cfg, max_steps, seed
        arguments_calls.append(evaluate_each_epoch)
        return object()

    monkeypatch.setattr(hf, "CompletionOnlyDataset", lambda *args, **kwargs: object())
    monkeypatch.setattr(hf, "CompletionOnlyCollator", lambda tokenizer: tokenizer)
    monkeypatch.setattr(hf, "Trainer", FakeTrainer)
    monkeypatch.setattr(hf, "training_arguments", fake_training_arguments)

    trainer = hf.make_trainer(
        model=object(),
        tokenizer=object(),
        train_records=[object()],
        cfg=_training_cfg(),
        output_dir="trainer",
        max_steps=None,
    )

    assert arguments_calls == [False]
    assert trainer.kwargs["eval_dataset"] is None


def test_train_model_rejects_shadow_evaluation_records(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Only target training"):
        workflow.train_model(
            OmegaConf.create({}),
            project_root=tmp_path,
            train_records=[],
            run_dir=tmp_path / "shadow",
            role="shadow",
            shadow_index=0,
            command="test",
            max_steps_value=None,
            model_seed=42,
            eval_records={"squad_validation": []},
        )


def test_training_completion_can_ignore_target_only_declaration_and_mask_metadata(
    tmp_path: Path,
) -> None:
    prior_cfg = OmegaConf.create(
        {
            "report": {
                "resolved_config_filename": "resolved_config.yaml",
                "experiment_filename": "experiment.md",
            },
            "workflow": {"stage": "train_target", "force": False},
            "train": {"epochs": 10},
            "experiment": {"purpose": "original declaration"},
        }
    )
    current_cfg = OmegaConf.create(OmegaConf.to_container(prior_cfg, resolve=True))
    current_cfg.workflow.stage = "train_shadows"
    current_cfg.experiment.purpose = "updated declaration"
    current_cfg.mask_reuse = {
        "enabled": True,
        "source_output_root": "outputs/squad_lora_rmia/formal",
        "expected_source_shadow_count": 5,
    }
    run_dir = tmp_path / "target"
    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    for name in ("train_manifest.jsonl", "metrics.json", "experiment.md"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    OmegaConf.save(prior_cfg, run_dir / "resolved_config.yaml")
    strategy = workflow.LoraFineTuningStrategy()

    assert not workflow._training_run_is_complete(current_cfg, run_dir, strategy)
    assert workflow._training_run_is_complete(
        current_cfg,
        run_dir,
        strategy,
        ignored_config_sections=("experiment", "mask_reuse"),
    )

    current_cfg.train.epochs = 11
    assert not workflow._training_run_is_complete(
        current_cfg,
        run_dir,
        strategy,
        ignored_config_sections=("experiment", "mask_reuse"),
    )


def test_train_target_ignores_mask_reuse_for_completion_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = OmegaConf.create(
        {
            "runtime": {"seed": 42},
            "train": {"target_eval_each_epoch": True},
            "workflow": {"force": False},
        }
    )
    captured: dict[str, object] = {}

    def fake_complete(
        cfg: object,
        run_dir: Path,
        strategy: object,
        *,
        extra_required_files: tuple[Path, ...] = (),
        ignored_config_sections: tuple[str, ...] = (),
    ) -> bool:
        del cfg, run_dir, strategy
        captured["extra_required_files"] = extra_required_files
        captured["ignored_config_sections"] = ignored_config_sections
        return True

    monkeypatch.setattr(workflow, "model_root", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(workflow, "_training_run_is_complete", fake_complete)

    checkpoint = workflow.train_target(
        cfg,
        project_root=tmp_path,
        command="test",
        smoke=False,
    )

    target_dir = tmp_path / "target" / "seed_42"
    assert checkpoint == target_dir / "adapter"
    assert captured == {
        "extra_required_files": (target_dir / "epoch_validation_metrics.json",),
        "ignored_config_sections": ("experiment", "mask_reuse"),
    }


def test_evaluation_records_share_formal_and_smoke_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    squad_records = [SimpleNamespace(record_id="squad")]
    trivia_records = [SimpleNamespace(record_id="trivia")]
    calls: list[tuple[str, int | None, str]] = []

    monkeypatch.setattr(
        workflow,
        "load_squad_records",
        lambda cfg, split: squad_records,
    )
    monkeypatch.setattr(
        workflow,
        "load_trivia_records",
        lambda cfg, split, *, limit: trivia_records,
    )

    def fake_subset(
        records: list[object],
        *,
        seed: int,
        limit: int | None,
        namespace: str,
    ) -> list[object]:
        del seed
        calls.append((namespace, limit, records[0].record_id))
        return records

    monkeypatch.setattr(workflow, "deterministic_subset", fake_subset)
    cfg = OmegaConf.create(
        {
            "runtime": {"seed": 42},
            "data": {"eval_limit": None, "smoke_eval_limit": 32},
        }
    )

    formal = workflow._evaluation_records(cfg, smoke=False)
    smoke = workflow._evaluation_records(cfg, smoke=True)

    assert list(formal) == ["squad_validation", "trivia_validation"]
    assert list(smoke) == ["squad_validation", "trivia_validation"]
    assert calls == [
        ("eval_squad_validation", None, "squad"),
        ("eval_trivia_validation", None, "trivia"),
        ("eval_squad_validation", 32, "squad"),
        ("eval_trivia_validation", 32, "trivia"),
    ]


def test_epoch_validation_metrics_merge_losses_and_perplexities() -> None:
    metrics = workflow._epoch_validation_metrics(
        [
            {"epoch": 1.0, "step": 10, "eval_squad_validation_loss": 1.0},
            {"epoch": 1.0, "step": 10, "eval_trivia_validation_loss": 2.0},
            {"epoch": 2.0, "step": 20, "eval_squad_validation_loss": 0.5},
            {"epoch": 2.0, "step": 20, "eval_trivia_validation_loss": 1.5},
        ],
        {
            "squad_validation": [object(), object()],
            "trivia_validation": [object()],
        },
        expected_epochs=2,
    )

    assert metrics["datasets"] == {
        "squad_validation": {"records": 2},
        "trivia_validation": {"records": 1},
    }
    assert [row["epoch"] for row in metrics["epochs"]] == [1.0, 2.0]
    assert metrics["epochs"][0]["global_step"] == 10
    assert math.isclose(
        metrics["epochs"][0]["squad_validation"]["perplexity"],
        math.e,
    )


def test_epoch_validation_metrics_reject_incomplete_or_duplicate_history() -> None:
    eval_records = {
        "squad_validation": [object()],
        "trivia_validation": [object()],
    }
    with pytest.raises(ValueError, match="missing validation losses"):
        workflow._epoch_validation_metrics(
            [{"epoch": 1.0, "step": 10, "eval_squad_validation_loss": 1.0}],
            eval_records,
            expected_epochs=1,
        )

    with pytest.raises(ValueError, match="Duplicate squad_validation"):
        workflow._epoch_validation_metrics(
            [
                {"epoch": 1.0, "step": 10, "eval_squad_validation_loss": 1.0},
                {"epoch": 1.0, "step": 10, "eval_squad_validation_loss": 1.1},
            ],
            eval_records,
            expected_epochs=1,
        )


def test_more_epochs_runner_has_required_model_gate_and_stage_order() -> None:
    project_root = Path(__file__).resolve().parents[1]
    runner = (
        project_root / "scripts" / "run_squad_lora_rmia_more_epochs_formal.sh"
    ).read_text(encoding="utf-8")

    assert '--config-name="$CONFIG_NAME"' in runner
    assert "pythia_410m | olmo_1b_hf" in runner
    assert "unsupported model" in runner
    positions = [
        runner.index(stage)
        for stage in (
            "prepare \\",
            "train_target \\",
            "evaluate \\",
            "generate_base \\",
            "generate \\",
            "build_candidates \\",
            "train_shadows;",
        )
    ]
    assert positions == sorted(positions)
    assert "attack" not in runner.split("for stage in", maxsplit=1)[1]
