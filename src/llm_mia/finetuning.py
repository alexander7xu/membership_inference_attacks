from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from src.llm_mia.data import file_sha256, write_json
from src.llm_mia.hf import (
    attach_lora,
    ensure_token_embeddings,
    load_causal_lm,
    load_model_for_inference,
    load_tokenizer,
)


class FineTuningStrategy(Protocol):
    name: str
    checkpoint_dirname: str
    target_eval_name: str
    generated_variant: str
    generated_group_prefix: str
    artifact_name: str
    report_label: str
    gradient_checkpointing: bool

    def prepare_model(self, model: Any, cfg: Any) -> Any: ...

    def validate_trainable(self, model: Any) -> tuple[int, int]: ...

    def checkpoint_is_complete(
        self, checkpoint_path: Path, *, config_sha256: str | None = None
    ) -> bool: ...

    def save_final(
        self,
        model: Any,
        tokenizer: Any,
        run_dir: Path,
        *,
        config_sha256: str,
    ) -> Path: ...

    def load_for_inference(
        self, cfg: Any, checkpoint_path: Path | None
    ) -> tuple[Any, Any]: ...


def parameter_counts(model: Any) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


@dataclass(frozen=True)
class LoraFineTuningStrategy:
    name: str = "lora"
    checkpoint_dirname: str = "adapter"
    target_eval_name: str = "target_lora"
    generated_variant: str = "lora"
    generated_group_prefix: str = "gen"
    artifact_name: str = "adapter"
    report_label: str = "LoRA"
    gradient_checkpointing: bool = False

    def prepare_model(self, model: Any, cfg: Any) -> Any:
        return attach_lora(model, cfg)

    def validate_trainable(self, model: Any) -> tuple[int, int]:
        trainable, total = parameter_counts(model)
        if not isinstance(model, PeftModel):
            raise TypeError("LoRA training requires a PeftModel.")
        if not 0 < trainable < total:
            raise ValueError(
                "LoRA must train a non-empty strict subset of model parameters."
            )
        return trainable, total

    def checkpoint_is_complete(
        self, checkpoint_path: Path, *, config_sha256: str | None = None
    ) -> bool:
        del config_sha256
        return (checkpoint_path / "adapter_config.json").is_file()

    def save_final(
        self,
        model: Any,
        tokenizer: Any,
        run_dir: Path,
        *,
        config_sha256: str,
    ) -> Path:
        del config_sha256
        checkpoint_path = run_dir / self.checkpoint_dirname
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        return checkpoint_path

    def load_for_inference(
        self, cfg: Any, checkpoint_path: Path | None
    ) -> tuple[Any, Any]:
        return load_model_for_inference(
            cfg,
            adapter_path=(None if checkpoint_path is None else str(checkpoint_path)),
        )


@dataclass(frozen=True)
class FullFineTuningStrategy:
    gradient_checkpointing: bool = True
    name: str = "full"
    checkpoint_dirname: str = "model"
    target_eval_name: str = "target_fullft"
    generated_variant: str = "fullft"
    generated_group_prefix: str = "fullft_gen"
    artifact_name: str = "model"
    report_label: str = "full-parameter"

    def prepare_model(self, model: Any, cfg: Any) -> Any:
        del cfg
        if isinstance(model, PeftModel):
            raise TypeError("Full fine-tuning forbids PeftModel wrappers.")
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        if self.gradient_checkpointing:
            if not hasattr(model, "gradient_checkpointing_enable"):
                raise TypeError("Model does not support gradient checkpointing.")
            model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        self.validate_trainable(model)
        return model

    def validate_trainable(self, model: Any) -> tuple[int, int]:
        if isinstance(model, PeftModel):
            raise TypeError("Full fine-tuning forbids PeftModel wrappers.")
        trainable, total = parameter_counts(model)
        if total <= 0 or trainable != total:
            raise ValueError(
                "Full fine-tuning requires every model parameter to be trainable: "
                f"trainable={trainable}, total={total}."
            )
        return trainable, total

    def checkpoint_is_complete(
        self, checkpoint_path: Path, *, config_sha256: str | None = None
    ) -> bool:
        success_path = checkpoint_path / "_SUCCESS"
        manifest_path = checkpoint_path / "manifest.json"
        if not success_path.is_file() or not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if manifest.get("format") != "full_model_safetensors":
            return False
        if config_sha256 is not None and manifest.get("config_sha256") != config_sha256:
            return False
        file_hashes = manifest.get("file_sha256")
        if not isinstance(file_hashes, dict) or not file_hashes:
            return False
        if not any(str(name).endswith(".safetensors") for name in file_hashes):
            return False
        return all(
            (checkpoint_path / str(name)).is_file()
            and file_sha256(checkpoint_path / str(name)) == digest
            for name, digest in file_hashes.items()
        )

    def save_final(
        self,
        model: Any,
        tokenizer: Any,
        run_dir: Path,
        *,
        config_sha256: str,
    ) -> Path:
        self.validate_trainable(model)
        checkpoint_path = run_dir / self.checkpoint_dirname
        if checkpoint_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing final checkpoint: {checkpoint_path}"
            )
        temporary_path = Path(
            tempfile.mkdtemp(prefix=".model-publish-", dir=str(run_dir))
        )
        try:
            model.save_pretrained(
                temporary_path,
                safe_serialization=True,
                max_shard_size="5GB",
            )
            tokenizer.save_pretrained(temporary_path)
            file_hashes = _checkpoint_file_hashes(temporary_path)
            if not any(name.endswith(".safetensors") for name in file_hashes):
                raise ValueError("Full checkpoint did not contain safetensors weights.")
            write_json(
                temporary_path / "manifest.json",
                {
                    "format": "full_model_safetensors",
                    "config_sha256": config_sha256,
                    "file_sha256": file_hashes,
                },
            )
            _verify_full_model_reload(temporary_path)
            if _checkpoint_file_hashes(temporary_path) != file_hashes:
                raise ValueError("Full checkpoint hashes changed during reload check.")
            (temporary_path / "_SUCCESS").write_text("verified\n", encoding="utf-8")
            os.replace(temporary_path, checkpoint_path)
        except Exception:
            shutil.rmtree(temporary_path, ignore_errors=True)
            raise
        if not self.checkpoint_is_complete(
            checkpoint_path, config_sha256=config_sha256
        ):
            raise ValueError("Published full checkpoint failed final validation.")
        return checkpoint_path

    def load_for_inference(
        self, cfg: Any, checkpoint_path: Path | None
    ) -> tuple[Any, Any]:
        if checkpoint_path is None:
            return load_model_for_inference(cfg, adapter_path=None)
        if not self.checkpoint_is_complete(checkpoint_path):
            raise ValueError(f"Full checkpoint is incomplete: {checkpoint_path}")
        tokenizer = load_tokenizer(
            str(checkpoint_path),
            revision=None,
            trust_remote_code=bool(cfg.tokenizer.trust_remote_code),
        )
        model = load_causal_lm(
            str(checkpoint_path),
            bf16=bool(cfg.precision.bf16),
            revision=None,
            trust_remote_code=bool(cfg.tokenizer.trust_remote_code),
        )
        ensure_token_embeddings(model, tokenizer)
        model.config.use_cache = True
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        return model, tokenizer


def _checkpoint_file_hashes(root: Path) -> dict[str, str]:
    excluded = {"_SUCCESS", "manifest.json"}
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def _verify_full_model_reload(checkpoint_path: Path) -> None:
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    input_ids = torch.zeros((1, 1), dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits
    if tuple(logits.shape[:2]) != (1, 1):
        raise ValueError(f"Reloaded checkpoint returned invalid logits: {logits.shape}")
    del model
