from __future__ import annotations

import csv
import hashlib
import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QARecord:
    record_id: str
    dataset: str
    split: str
    prompt: str
    completion: str
    answers: tuple[str, ...]
    context: str = ""
    question: str = ""
    title: str = ""
    answer_start: int | None = None

    @property
    def full_text(self) -> str:
        return self.prompt + self.completion

    @property
    def content_sha256(self) -> str:
        return text_sha256(self.full_text)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash_int(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    return int(digest, 16)


def make_prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n"


def first_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            text = first_text(item)
            if text:
                return text
    return ""


def squad_row_to_record(row: dict[str, Any], split: str) -> QARecord:
    answers = row.get("answers") or {}
    answer_texts = tuple(str(x) for x in answers.get("text", []) if str(x))
    answer = answer_texts[0] if answer_texts else ""
    starts = answers.get("answer_start", [])
    answer_start = int(starts[0]) if starts else None
    context = str(row["context"])
    question = str(row["question"])
    return QARecord(
        record_id=str(row["id"]),
        dataset="squad",
        split=split,
        context=context,
        question=question,
        title=str(row.get("title", "")),
        prompt=make_prompt(context, question),
        completion=f" {answer.strip()}",
        answers=answer_texts,
        answer_start=answer_start,
    )


def _first_nonempty_from_mapping(value: Any, key: str) -> str:
    if not isinstance(value, dict):
        return ""
    return first_text(value.get(key))


def trivia_row_to_record(row: dict[str, Any], split: str) -> QARecord | None:
    context = _first_nonempty_from_mapping(row.get("entity_pages"), "wiki_context")
    if not context:
        context = _first_nonempty_from_mapping(
            row.get("search_results"), "search_context"
        )
    if not context:
        return None

    answer_block = row.get("answer") if isinstance(row.get("answer"), dict) else row
    value = first_text(answer_block.get("value")) or first_text(
        answer_block.get("normalized_value")
    )
    aliases = (
        answer_block.get("aliases") or answer_block.get("normalized_aliases") or []
    )
    aliases_tuple = tuple(str(x) for x in aliases if str(x))
    answers = aliases_tuple or ((value,) if value else tuple())
    answer = value or (answers[0] if answers else "")
    question = str(row["question"])
    record_id = str(row.get("question_id", row.get("id", text_sha256(question)[:16])))
    return QARecord(
        record_id=record_id,
        dataset="triviaqa",
        split=split,
        context=context,
        question=question,
        prompt=make_prompt(context, question),
        completion=f" {answer.strip()}",
        answers=answers,
    )


def split_records(
    rows: list[QARecord],
    *,
    seed: int,
    sizes: Mapping[str, int],
    namespace: str,
) -> dict[str, list[QARecord]]:
    if any(size < 0 for size in sizes.values()):
        raise ValueError("Split sizes must be non-negative.")
    requested = sum(sizes.values())
    if requested > len(rows):
        raise ValueError(
            f"Requested split sizes exceed available rows: {requested} > {len(rows)}"
        )

    ordered = sorted(
        rows,
        key=lambda row: stable_hash_int(seed, f"{namespace}:{row.record_id}"),
    )
    result: dict[str, list[QARecord]] = {}
    start = 0
    for name, size in sizes.items():
        end = start + size
        result[name] = ordered[start:end]
        start = end
    return result


def split_squad_train(
    rows: list[QARecord],
    *,
    seed: int,
    target_train_size: int,
    auxiliary_pool_size: int,
) -> dict[str, list[QARecord]]:
    return split_records(
        rows,
        seed=seed,
        namespace="squad_train",
        sizes={
            "squad_target_train": target_train_size,
            "squad_attacker_auxiliary_pool": auxiliary_pool_size,
        },
    )


def deterministic_subset(
    rows: list[QARecord], *, seed: int, limit: int | None, namespace: str
) -> list[QARecord]:
    if limit is None or limit >= len(rows):
        return list(rows)
    ordered = sorted(
        rows,
        key=lambda row: stable_hash_int(seed, f"{namespace}:{row.record_id}"),
    )
    return ordered[:limit]


def deterministic_unique_subset(
    rows: list[QARecord],
    *,
    seed: int,
    limit: int | None,
    namespace: str,
    excluded_content_hashes: set[str] | None = None,
) -> list[QARecord]:
    ordered = sorted(
        rows,
        key=lambda row: stable_hash_int(seed, f"{namespace}:{row.record_id}"),
    )
    seen = set(excluded_content_hashes or ())
    result: list[QARecord] = []
    for row in ordered:
        if row.content_sha256 in seen:
            continue
        seen.add(row.content_sha256)
        result.append(row)
        if limit is not None and len(result) >= limit:
            break
    return result


def record_to_json(record: QARecord) -> dict[str, Any]:
    data = asdict(record)
    data["answers"] = list(record.answers)
    data["content_sha256"] = record.content_sha256
    return data


def record_from_json(data: dict[str, Any]) -> QARecord:
    payload = dict(data)
    payload.pop("content_sha256", None)
    payload["answers"] = tuple(payload.get("answers", []))
    return QARecord(**payload)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def deduplicate_candidate_rows(
    public_rows: list[dict[str, Any]], private_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(public_rows) != len(private_rows):
        raise ValueError("Public candidates and private labels must have equal length.")

    unique_public: list[dict[str, Any]] = []
    unique_private: list[dict[str, Any]] = []
    seen_by_content: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for public, private in zip(public_rows, private_rows, strict=True):
        if public.get("candidate_id") != private.get("candidate_id"):
            raise ValueError("Public candidate and private label IDs must align.")
        content_hash = str(public.get("content_sha256", ""))
        if not content_hash:
            raise ValueError("Public candidates must include content_sha256.")
        previous = seen_by_content.get(content_hash)
        if previous is None:
            seen_by_content[content_hash] = (public, private)
            unique_public.append(public)
            unique_private.append(private)
        elif previous != (public, private):
            raise ValueError(
                "Duplicate candidate content has conflicting public or private metadata."
            )
    return unique_public, unique_private


def canonicalize_candidate_rows(
    public_rows: list[dict[str, Any]], private_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Canonicalize model inputs while preserving all evaluator-only provenance."""
    if len(public_rows) != len(private_rows):
        raise ValueError("Public candidates and private labels must have equal length.")

    canonical_public: list[dict[str, Any]] = []
    evaluator_mapping: list[dict[str, Any]] = []
    canonical_by_content: dict[str, dict[str, Any]] = {}
    content_by_canonical_id: dict[str, str] = {}
    labels_by_content: dict[str, int] = {}
    for raw_index, (public, private) in enumerate(
        zip(public_rows, private_rows, strict=True)
    ):
        source_candidate_id = str(public.get("candidate_id", ""))
        if not source_candidate_id or source_candidate_id != str(
            private.get("candidate_id", "")
        ):
            raise ValueError("Public candidate and private label IDs must align.")
        content_hash = str(public.get("content_sha256", ""))
        if not content_hash:
            raise ValueError("Public candidates must include content_sha256.")
        canonical_id = f"candidate_{content_hash[:20]}"
        previous_hash = content_by_canonical_id.get(canonical_id)
        if previous_hash is not None and previous_hash != content_hash:
            raise ValueError(f"Canonical candidate ID collision: {canonical_id}")
        content_by_canonical_id[canonical_id] = content_hash

        model_row = {
            "candidate_id": canonical_id,
            "prompt": str(public["prompt"]),
            "completion": str(public["completion"]),
            "content_sha256": content_hash,
        }
        previous_public = canonical_by_content.get(content_hash)
        if previous_public is None:
            canonical_by_content[content_hash] = model_row
            canonical_public.append(model_row)
        elif previous_public != model_row:
            raise ValueError("Equal content hashes have different model text.")

        label = int(private.get("record_membership_label", 0))
        previous_label = labels_by_content.get(content_hash)
        if previous_label is not None and previous_label != label:
            raise ValueError("Duplicate candidate content has conflicting labels.")
        labels_by_content[content_hash] = label
        evaluator_mapping.append(
            {
                **private,
                "candidate_id": canonical_id,
                "source_candidate_id": source_candidate_id,
                "raw_row_index": raw_index,
            }
        )
    return canonical_public, evaluator_mapping


def candidate_id(prefix: str, record: QARecord) -> str:
    return f"{prefix}_{record.content_sha256[:20]}"


def public_candidate(record: QARecord, candidate_id_value: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id_value,
        "prompt": record.prompt,
        "completion": record.completion,
        "content_sha256": record.content_sha256,
    }


def record_from_public_candidate(row: dict[str, Any]) -> QARecord:
    return QARecord(
        record_id=str(row["candidate_id"]),
        dataset="candidate",
        split="public",
        prompt=str(row["prompt"]),
        completion=str(row["completion"]),
        answers=(str(row["completion"]).strip(),),
    )


def conditioned_bernoulli_mask(
    candidate_id_value: str,
    *,
    seed: int,
    shadow_count: int,
    inclusion_probability: float,
) -> list[int]:
    if shadow_count < 2:
        raise ValueError("At least two shadow models are required.")
    if not 0.0 < inclusion_probability < 1.0:
        raise ValueError("Inclusion probability must be strictly between zero and one.")

    rng = random.Random(stable_hash_int(seed, candidate_id_value))
    for _ in range(10_000):
        mask = [int(rng.random() < inclusion_probability) for _ in range(shadow_count)]
        if 0 < sum(mask) < shadow_count:
            return mask
    raise RuntimeError("Could not sample a valid conditioned shadow mask.")


def write_shadow_masks(
    path: Path,
    candidate_ids: list[str],
    *,
    seed: int,
    shadow_count: int,
    inclusion_probability: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["candidate_id"] + [f"shadow_{idx:02d}" for idx in range(shadow_count)]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidate_ids:
            mask = conditioned_bernoulli_mask(
                candidate,
                seed=seed,
                shadow_count=shadow_count,
                inclusion_probability=inclusion_probability,
            )
            row: dict[str, str | int] = {"candidate_id": candidate}
            row.update(
                {f"shadow_{idx:02d}": include for idx, include in enumerate(mask)}
            )
            writer.writerow(row)


def write_single_shadow_smoke_mask(
    path: Path,
    candidate_ids: list[str],
    *,
    seed: int,
    max_included: int,
) -> None:
    """Write a bounded one-shadow inclusion mask for training-only smoke tests."""
    if max_included < 1:
        raise ValueError("Smoke mask must include at least one candidate.")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Smoke candidate IDs must be unique.")
    ranked = sorted(
        candidate_ids,
        key=lambda candidate: stable_hash_int(seed, f"smoke-shadow:{candidate}"),
    )
    included = set(ranked[: min(max_included, len(ranked))])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["candidate_id", "shadow_00"])
        writer.writeheader()
        for candidate in candidate_ids:
            writer.writerow(
                {
                    "candidate_id": candidate,
                    "shadow_00": int(candidate in included),
                }
            )


def read_shadow_masks(
    path: Path, *, require_in_out: bool = True
) -> dict[str, list[int]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        result: dict[str, list[int]] = {}
        shadow_fields = [
            name for name in reader.fieldnames or [] if name.startswith("shadow_")
        ]
        for row in reader:
            mask = [int(row[name]) for name in shadow_fields]
            if not mask or any(value not in (0, 1) for value in mask):
                raise ValueError(
                    f"Candidate {row['candidate_id']} has an invalid mask."
                )
            if require_in_out and not 0 < sum(mask) < len(mask):
                raise ValueError(
                    f"Candidate {row['candidate_id']} does not have both IN and OUT shadows."
                )
            result[row["candidate_id"]] = mask
        return result
