from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, GPT2Config

from src.llm_mia.finetuning import FullFineTuningStrategy
from src.llm_mia.workflow import _resume_checkpoint, fine_tuning_strategy


class TinyTrainableModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(2, 2)
        self.second = torch.nn.Linear(2, 2)
        self.second.weight.requires_grad_(False)
        self.config = SimpleNamespace(use_cache=True)
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True


class MinimalTokenizer:
    def save_pretrained(self, output_dir: Path) -> None:
        (Path(output_dir) / "tokenizer_config.json").write_text(
            "{}\n", encoding="utf-8"
        )


def _fullft_config():
    config_dir = str(Path("conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(
            config_name="squad_fullft_rmia",
            overrides=["runtime.seed=42"],
        )


def test_fullft_hydra_config_has_strategy_and_no_lora_block() -> None:
    cfg = _fullft_config()
    OmegaConf.resolve(cfg)

    strategy = fine_tuning_strategy(cfg)

    assert strategy.name == "full"
    assert strategy.checkpoint_dirname == "model"
    assert cfg.paths.output_root == "outputs/squad_fullft_rmia"
    assert "lora" not in cfg
    assert cfg.train.learning_rate == 2e-5
    assert cfg.train.epochs == 1


def test_fullft_strategy_makes_every_parameter_trainable() -> None:
    model = TinyTrainableModel()
    strategy = FullFineTuningStrategy()

    prepared = strategy.prepare_model(model, cfg=None)
    trainable, total = strategy.validate_trainable(prepared)

    assert prepared is model
    assert trainable == total
    assert model.gradient_checkpointing_enabled is True
    assert model.config.use_cache is False


def test_full_checkpoint_is_atomic_reloadable_and_hash_checked(tmp_path: Path) -> None:
    config = GPT2Config(
        vocab_size=17,
        n_positions=8,
        n_ctx=8,
        n_embd=8,
        n_layer=1,
        n_head=1,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = AutoModelForCausalLM.from_config(config).eval()
    input_ids = torch.tensor([[0, 2, 3]], dtype=torch.long)
    with torch.inference_mode():
        expected = model(input_ids=input_ids).logits

    strategy = FullFineTuningStrategy(gradient_checkpointing=False)
    checkpoint = strategy.save_final(
        model,
        MinimalTokenizer(),
        tmp_path,
        config_sha256="config-hash",
    )
    reloaded = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True
    ).eval()
    with torch.inference_mode():
        actual = reloaded(input_ids=input_ids).logits

    torch.testing.assert_close(actual, expected)
    assert checkpoint == tmp_path / "model"
    assert (checkpoint / "_SUCCESS").is_file()
    assert strategy.checkpoint_is_complete(checkpoint, config_sha256="config-hash")
    assert not list(tmp_path.glob(".model-publish-*"))

    config_path = checkpoint / "config.json"
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n")
    assert not strategy.checkpoint_is_complete(checkpoint, config_sha256="config-hash")


def test_auto_resume_chooses_latest_numeric_checkpoint(tmp_path: Path) -> None:
    cfg = _fullft_config()
    trainer_dir = tmp_path / "trainer"
    (trainer_dir / "checkpoint-9").mkdir(parents=True)
    (trainer_dir / "checkpoint-101").mkdir()
    (trainer_dir / "checkpoint-invalid").mkdir()

    selected = _resume_checkpoint(
        cfg,
        project_root=tmp_path,
        trainer_dir=trainer_dir,
    )

    assert selected == str(trainer_dir / "checkpoint-101")
