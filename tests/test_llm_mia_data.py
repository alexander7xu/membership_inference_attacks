from src.llm_mia.data import (
    QARecord,
    deduplicate_candidate_rows,
    deterministic_unique_subset,
    read_shadow_masks,
    split_records,
    split_squad_train,
    write_shadow_masks,
)


def _records(count: int) -> list[QARecord]:
    return [
        QARecord(
            record_id=f"id-{idx}",
            dataset="squad",
            split="train",
            prompt="Context:\nctx\n\nQuestion:\nq\n\nAnswer:\n",
            completion=" answer",
            answers=("answer",),
        )
        for idx in range(count)
    ]


def test_squad_hash_split_is_deterministic():
    rows = _records(10)

    left = split_squad_train(
        rows,
        seed=42,
        target_train_size=4,
        auxiliary_pool_size=6,
    )
    right = split_squad_train(
        rows,
        seed=42,
        target_train_size=4,
        auxiliary_pool_size=6,
    )

    assert left == right
    assert len(left["squad_target_train"]) == 4
    assert len(left["squad_attacker_auxiliary_pool"]) == 6
    assert {record.record_id for record in left["squad_target_train"]}.isdisjoint(
        record.record_id for record in left["squad_attacker_auxiliary_pool"]
    )


def test_split_records_builds_disjoint_validation_partitions():
    rows = _records(10)

    split = split_records(
        rows,
        seed=42,
        namespace="validation",
        sizes={"candidates": 4, "population": 3},
    )

    assert len(split["candidates"]) == 4
    assert len(split["population"]) == 3
    assert {record.record_id for record in split["candidates"]}.isdisjoint(
        record.record_id for record in split["population"]
    )


def test_shadow_masks_are_deterministic_and_have_in_out_models(tmp_path):
    path = tmp_path / "masks.csv"
    candidate_ids = [f"cand-{idx}" for idx in range(32)]
    write_shadow_masks(
        path,
        candidate_ids,
        seed=123,
        shadow_count=5,
        inclusion_probability=0.5,
    )

    first = read_shadow_masks(path)
    write_shadow_masks(
        path,
        candidate_ids,
        seed=123,
        shadow_count=5,
        inclusion_probability=0.5,
    )

    assert first == read_shadow_masks(path)
    assert set(first) == set(candidate_ids)
    assert all(0 < sum(mask) < 5 for mask in first.values())


def test_candidate_deduplication_preserves_first_exact_copy():
    public = {
        "candidate_id": "candidate-1",
        "prompt": "prompt",
        "completion": " completion",
        "content_sha256": "content-hash",
    }
    private = {
        "candidate_id": "candidate-1",
        "private_group": "gold_trivia_validation",
        "record_membership_label": 0,
    }

    unique_public, unique_private = deduplicate_candidate_rows(
        [public, dict(public)], [private, dict(private)]
    )

    assert unique_public == [public]
    assert unique_private == [private]


def test_candidate_deduplication_rejects_conflicting_labels():
    public = {
        "candidate_id": "candidate-1",
        "prompt": "prompt",
        "completion": " completion",
        "content_sha256": "content-hash",
    }
    member = {"candidate_id": "candidate-1", "record_membership_label": 1}
    nonmember = {"candidate_id": "candidate-1", "record_membership_label": 0}

    try:
        deduplicate_candidate_rows([public, dict(public)], [member, nonmember])
    except ValueError as error:
        assert "conflicting" in str(error)
    else:
        raise AssertionError("Conflicting duplicate labels must be rejected.")


def test_deterministic_unique_subset_skips_duplicates_and_exclusions():
    rows = [
        QARecord(
            record_id=f"id-{index}",
            dataset="squad",
            split="train",
            prompt=f"prompt-{content}",
            completion=" completion",
            answers=("completion",),
        )
        for index, content in enumerate(["a", "a", "b", "c"])
    ]
    excluded = {rows[2].content_sha256}

    first = deterministic_unique_subset(
        rows,
        seed=42,
        limit=2,
        namespace="shadow_fill",
        excluded_content_hashes=excluded,
    )
    second = deterministic_unique_subset(
        rows,
        seed=42,
        limit=2,
        namespace="shadow_fill",
        excluded_content_hashes=excluded,
    )

    assert first == second
    assert len(first) == 2
    assert len({record.content_sha256 for record in first}) == 2
    assert all(record.content_sha256 not in excluded for record in first)
