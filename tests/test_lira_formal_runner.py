from pathlib import Path


def test_lira_formal_runner_uses_separate_safe_attack_batch() -> None:
    script = Path("scripts/run_squad_lora_lira_formal.sh").read_text(encoding="utf-8")

    assert 'INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-96}"' in script
    assert 'ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-32}"' in script
    assert 'eval.batch_size="$INFERENCE_BATCH_SIZE"' in script
    assert 'generation.batch_size="$INFERENCE_BATCH_SIZE"' in script
    assert 'attack.batch_size="$ATTACK_BATCH_SIZE"' in script
    assert 'attack.batch_size="$INFERENCE_BATCH_SIZE"' not in script
