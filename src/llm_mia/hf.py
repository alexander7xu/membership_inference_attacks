from __future__ import annotations

import inspect
import math
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
from queue import SimpleQueue
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.llm_mia.data import QARecord, make_prompt
from src.utils.annotation import FP, Int, T, tensor_typechecked


@dataclass
class SequenceStats:
    token_logprob_sum: float
    token_count: int
    mean_logprob: float
    loss: float


def _ordered_thread_map[MapResult](
    function: Callable[[QARecord], MapResult],
    records: list[QARecord],
    *,
    num_workers: int,
) -> list[MapResult]:
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}.")
    if num_workers == 1:
        return [function(record) for record in records]
    with ThreadPoolExecutor(
        max_workers=num_workers, thread_name_prefix="llm-mia-tokenizer"
    ) as executor:
        return list(executor.map(function, records))


def _bind_thread_local_tokenizer[MapResult](
    function: Callable[..., MapResult],
    tokenizer: Any,
    *,
    num_workers: int,
) -> Callable[[QARecord], MapResult]:
    if num_workers == 1:
        return partial(function, tokenizer=tokenizer)
    tokenizers: SimpleQueue[Any] = SimpleQueue()
    for _ in range(num_workers):
        tokenizers.put(deepcopy(tokenizer))
    state = threading.local()

    def apply(record: QARecord) -> MapResult:
        local_tokenizer = getattr(state, "tokenizer", None)
        if local_tokenizer is None:
            local_tokenizer = tokenizers.get()
            state.tokenizer = local_tokenizer
        return function(record, tokenizer=local_tokenizer)

    return apply


class CompletionOnlyDataset(Dataset):
    def __init__(
        self,
        records: list[QARecord],
        tokenizer: Any,
        max_length: int,
        *,
        preprocessing_workers: int = 1,
    ):
        tokenize = _bind_thread_local_tokenizer(
            partial(tokenize_completion_only, max_length=max_length),
            tokenizer,
            num_workers=preprocessing_workers,
        )
        self._items = _ordered_thread_map(
            tokenize,
            records,
            num_workers=preprocessing_workers,
        )
        self._items = [item for item in self._items if item["label_count"] > 0]
        if not self._items:
            raise ValueError("No records with completion tokens after truncation.")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        item = self._items[index]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
        }


class CompletionOnlyCollator:
    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        label_rows = [feature.pop("labels") for feature in features]
        batch = self._tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        max_length = batch["input_ids"].shape[1]
        labels = []
        for row in label_rows:
            labels.append(row + [-100] * (max_length - len(row)))
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def load_tokenizer(
    model_name_or_path: str, *, revision: str | None = None, trust_remote_code: bool
) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_causal_lm(
    model_name_or_path: str,
    *,
    bf16: bool,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> Any:
    dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    model.config.use_cache = False
    return model


def attach_lora(model: Any, cfg: Any) -> Any:
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(cfg.lora.r),
        lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        target_modules=str(cfg.lora.target_modules),
    )
    return get_peft_model(model, lora_config)


def load_model_for_inference(
    cfg: Any,
    *,
    adapter_path: str | None,
) -> tuple[Any, Any]:
    tokenizer = load_tokenizer(
        str(cfg.model.name_or_path),
        revision=str(cfg.model.revision),
        trust_remote_code=bool(cfg.tokenizer.trust_remote_code),
    )
    model = load_causal_lm(
        str(cfg.model.name_or_path),
        bf16=bool(cfg.precision.bf16),
        revision=str(cfg.model.revision),
        trust_remote_code=bool(cfg.tokenizer.trust_remote_code),
    )
    ensure_token_embeddings(model, tokenizer)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    _set_cache(model, True)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model, tokenizer


def ensure_token_embeddings(model: Any, tokenizer: Any) -> None:
    embedding_count = model.get_input_embeddings().num_embeddings
    tokenizer_count = len(tokenizer)
    if tokenizer_count > embedding_count:
        model.resize_token_embeddings(tokenizer_count)


def fit_prompt_and_completion(
    record: QARecord,
    tokenizer: Any,
    max_length: int,
) -> tuple[str, str]:
    prompt = record.prompt
    completion = record.completion
    full = tokenizer(prompt + completion, add_special_tokens=True, truncation=False)
    if len(full["input_ids"]) <= max_length:
        return prompt, completion
    if not record.context:
        return prompt, completion

    low, high = 0, len(record.context)
    best_context = ""
    while low <= high:
        width = (low + high) // 2
        context = _context_window(record.context, record.answer_start, width)
        trial_prompt = make_prompt(context, record.question)
        trial = tokenizer(
            trial_prompt + completion,
            add_special_tokens=True,
            truncation=False,
        )
        if len(trial["input_ids"]) <= max_length:
            best_context = context
            low = width + 1
        else:
            high = width - 1
    return make_prompt(best_context, record.question), completion


def _materialize_record(
    record: QARecord,
    *,
    tokenizer: Any,
    max_length: int,
) -> QARecord:
    prompt, completion = fit_prompt_and_completion(record, tokenizer, max_length)
    return replace(record, prompt=prompt, completion=completion)


def materialize_records(
    records: list[QARecord],
    tokenizer: Any,
    max_length: int,
    *,
    num_workers: int = 1,
) -> list[QARecord]:
    materialize = _bind_thread_local_tokenizer(
        partial(_materialize_record, max_length=max_length),
        tokenizer,
        num_workers=num_workers,
    )
    return _ordered_thread_map(materialize, records, num_workers=num_workers)


def _context_window(context: str, answer_start: int | None, width: int) -> str:
    if width >= len(context):
        return context
    if answer_start is None:
        return context[:width]
    center = min(max(answer_start, 0), len(context))
    start = max(0, center - width // 2)
    end = min(len(context), start + width)
    start = max(0, end - width)
    return context[start:end]


def tokenize_completion_only(
    record: QARecord,
    tokenizer: Any,
    max_length: int,
) -> dict[str, list[int] | int]:
    prompt, completion = fit_prompt_and_completion(record, tokenizer, max_length)
    prompt_tokens = tokenizer(prompt, add_special_tokens=True, truncation=False)
    full_tokens = tokenizer(
        prompt + completion,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = list(full_tokens["input_ids"])
    labels = list(input_ids)
    prompt_length = min(len(prompt_tokens["input_ids"]), len(labels))
    labels[:prompt_length] = [-100] * prompt_length
    label_count = sum(1 for label in labels if label != -100)
    return {
        "input_ids": input_ids,
        "attention_mask": list(full_tokens["attention_mask"]),
        "labels": labels,
        "label_count": label_count,
    }


def _cfg_get(obj: Any, name: str, default: Any) -> Any:
    try:
        return getattr(obj, name)
    except (AttributeError, KeyError):
        return default


def _set_cache(model: Any, enabled: bool) -> None:
    if hasattr(model, "config"):
        model.config.use_cache = enabled
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "config"):
        base_model.config.use_cache = enabled


def training_arguments(
    output_dir: str, cfg: Any, *, max_steps: int | None
) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": float(cfg.train.epochs),
        "per_device_train_batch_size": int(cfg.train.per_device_train_batch_size),
        "per_device_eval_batch_size": int(cfg.train.per_device_eval_batch_size),
        "gradient_accumulation_steps": int(cfg.train.gradient_accumulation_steps),
        "learning_rate": float(cfg.train.learning_rate),
        "warmup_ratio": float(cfg.train.warmup_ratio),
        "weight_decay": float(cfg.train.weight_decay),
        "bf16": bool(cfg.precision.bf16) and torch.cuda.is_available(),
        "logging_steps": int(cfg.train.logging_steps),
        "save_strategy": "no",
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": int(_cfg_get(cfg.train, "dataloader_num_workers", 0)),
        "dataloader_pin_memory": bool(
            _cfg_get(cfg.train, "dataloader_pin_memory", True)
        ),
        "seed": int(cfg.runtime.seed),
        "data_seed": int(cfg.runtime.seed),
    }
    if int(kwargs["dataloader_num_workers"]) > 0:
        kwargs["dataloader_persistent_workers"] = bool(
            _cfg_get(cfg.train, "dataloader_persistent_workers", True)
        )
    if max_steps is not None:
        kwargs["max_steps"] = int(max_steps)
    signature = inspect.signature(TrainingArguments)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"
    return TrainingArguments(**kwargs)


def make_trainer(
    *,
    model: Any,
    tokenizer: Any,
    train_records: list[QARecord],
    cfg: Any,
    output_dir: str,
    max_steps: int | None,
) -> Trainer:
    train_dataset = CompletionOnlyDataset(
        train_records,
        tokenizer,
        int(cfg.tokenizer.max_length),
        preprocessing_workers=int(cfg.tokenizer.preprocessing_workers),
    )
    return Trainer(
        model=model,
        args=training_arguments(output_dir, cfg, max_steps=max_steps),
        train_dataset=train_dataset,
        data_collator=CompletionOnlyCollator(tokenizer),
    )


@tensor_typechecked
def completion_token_losses(
    logits: FP[T, "batch sequence vocab"],
    labels: Int[T, "batch sequence"],
    *,
    chunk_tokens: int,
) -> FP[T, "batch sequence"]:
    """Compute float32 token losses without materializing full float32 logits."""
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}.")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "Logit and label dimensions must match before the vocabulary axis: "
            f"{tuple(logits.shape)} versus {tuple(labels.shape)}."
        )

    token_losses = torch.empty_like(labels, dtype=torch.float32)
    vocab_size = logits.shape[-1]
    for start in range(0, logits.shape[1], chunk_tokens):
        end = min(start + chunk_tokens, logits.shape[1])
        chunk_logits = logits[:, start:end, :].contiguous().float()
        chunk_labels = labels[:, start:end].contiguous()
        chunk_losses = F.cross_entropy(
            chunk_logits.reshape(-1, vocab_size),
            chunk_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(chunk_labels.shape)
        token_losses[:, start:end].copy_(chunk_losses)
    return token_losses


@torch.no_grad()
def score_records(
    model: Any,
    tokenizer: Any,
    records: list[QARecord],
    *,
    batch_size: int,
    max_length: int,
    loss_chunk_tokens: int = 64,
    tokenization_workers: int = 1,
) -> list[SequenceStats]:
    dataset = CompletionOnlyDataset(
        records,
        tokenizer,
        max_length,
        preprocessing_workers=tokenization_workers,
    )
    collator = CompletionOnlyCollator(tokenizer)
    device = next(model.parameters()).device
    stats: list[SequenceStats] = []
    for start in range(0, len(dataset), batch_size):
        features = [
            dataset[index]
            for index in range(start, min(start + batch_size, len(dataset)))
        ]
        batch = collator(features)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        mask = shifted_labels != -100
        token_losses = completion_token_losses(
            logits,
            shifted_labels,
            chunk_tokens=loss_chunk_tokens,
        )
        token_log_probs = (-token_losses).masked_fill(~mask, 0.0)
        sums = token_log_probs.sum(dim=1).detach().cpu().tolist()
        counts = mask.sum(dim=1).detach().cpu().tolist()
        for total, count in zip(sums, counts, strict=True):
            count = int(count)
            mean_logprob = float(total / max(count, 1))
            loss = -mean_logprob if count else math.nan
            stats.append(
                SequenceStats(
                    token_logprob_sum=float(total),
                    token_count=count,
                    mean_logprob=mean_logprob,
                    loss=loss,
                )
            )
    return stats


@torch.no_grad()
def generate_completions(
    model: Any,
    tokenizer: Any,
    records: list[QARecord],
    *,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> list[str]:
    device = next(model.parameters()).device
    completions: list[str] = []
    prompts = [
        fit_prompt_and_completion(record, tokenizer, max_length)[0]
        for record in records
    ]
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            generate_kwargs: dict[str, Any] = {
                "do_sample": do_sample,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
            }
            if do_sample:
                generate_kwargs["temperature"] = temperature
            generated = model.generate(**encoded, **generate_kwargs)
            prompt_length = encoded["input_ids"].shape[1]
            for row in generated:
                new_tokens = row[prompt_length:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                completions.append(text.strip())
    finally:
        tokenizer.padding_side = original_padding_side
    return completions
