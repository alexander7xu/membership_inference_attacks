from __future__ import annotations

import torch
import torch.nn.functional as F

from src.llm_mia import hf


def test_chunked_completion_losses_match_full_float32() -> None:
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(3, 7, 11, generator=generator, dtype=torch.bfloat16)
    labels = torch.randint(0, 11, (3, 7), generator=generator, dtype=torch.long)
    labels[0, :2] = -100

    expected = F.cross_entropy(
        logits.float().transpose(1, 2),
        labels,
        ignore_index=-100,
        reduction="none",
    )
    actual = hf.completion_token_losses(logits, labels, chunk_tokens=3)

    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32


def test_completion_losses_bound_each_float32_chunk(monkeypatch) -> None:
    original = hf.F.cross_entropy
    observed_shapes: list[tuple[int, int]] = []

    def recording_cross_entropy(inputs, targets, **kwargs):
        observed_shapes.append(tuple(inputs.shape))
        return original(inputs, targets, **kwargs)

    monkeypatch.setattr(hf.F, "cross_entropy", recording_cross_entropy)
    logits = torch.zeros(3, 7, 11, dtype=torch.bfloat16)
    labels = torch.zeros(3, 7, dtype=torch.long)

    hf.completion_token_losses(logits, labels, chunk_tokens=2)

    assert observed_shapes == [(6, 11), (6, 11), (6, 11), (3, 11)]


def test_completion_losses_reject_invalid_chunk_size() -> None:
    logits = torch.zeros(1, 2, 3)
    labels = torch.zeros(1, 2, dtype=torch.long)

    try:
        hf.completion_token_losses(logits, labels, chunk_tokens=0)
    except ValueError as exc:
        assert "must be positive" in str(exc)
    else:
        raise AssertionError("Expected invalid chunk size to fail.")
