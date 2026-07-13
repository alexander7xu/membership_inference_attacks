from __future__ import annotations

import threading

from src.llm_mia import hf
from src.llm_mia.data import QARecord


def make_records(count: int) -> list[QARecord]:
    return [
        QARecord(
            record_id=f"row-{index}",
            dataset="fixture",
            split="test",
            prompt=f"prompt-{index}",
            completion=" answer",
            answers=("answer",),
        )
        for index in range(count)
    ]


def test_materialize_records_uses_workers_and_preserves_order(monkeypatch) -> None:
    barrier = threading.Barrier(4)
    thread_ids: set[int] = set()
    tokenizer_ids: set[int] = set()
    lock = threading.Lock()

    class DummyTokenizer:
        pass

    def fake_fit(record, tokenizer, max_length):
        del max_length
        with lock:
            thread_ids.add(threading.get_ident())
            tokenizer_ids.add(id(tokenizer))
        barrier.wait(timeout=5)
        return record.prompt, record.completion

    monkeypatch.setattr(hf, "fit_prompt_and_completion", fake_fit)
    result = hf.materialize_records(
        make_records(4), DummyTokenizer(), 32, num_workers=4
    )

    assert [record.record_id for record in result] == [
        "row-0",
        "row-1",
        "row-2",
        "row-3",
    ]
    assert len(thread_ids) == 4
    assert len(tokenizer_ids) == 4


def test_materialize_records_rejects_nonpositive_workers() -> None:
    try:
        hf.materialize_records(make_records(1), object(), 32, num_workers=0)
    except ValueError as exc:
        assert "must be positive" in str(exc)
    else:
        raise AssertionError("Expected invalid worker count to fail.")
