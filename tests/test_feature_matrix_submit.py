from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("mode", "partition", "model"),
    [
        ("pythia-full", 'partition="xe8545"', 'model="pythia_410m"'),
        ("olmo-full", 'partition="tmp"', 'model="olmo_1b_hf"'),
    ],
)
def test_submit_script_fixed_modes(mode: str, partition: str, model: str) -> None:
    script = Path("scripts/submit_squad_lora_mia_feature_matrix.sh").read_text()

    mode_block = script.split(f"  {mode})", 1)[1].split("    ;;", 1)[0]
    assert partition in mode_block
    assert model in mode_block
    assert "--gpus-per-node=1" in script
    assert "exec bash ./scripts/run_squad_lora_mia_feature_matrix.sh" in script
    assert "--time=6-23:30:00" in script


def test_runner_uses_only_analysis_stage_and_propagates_exit_code() -> None:
    script = Path("scripts/run_squad_lora_mia_feature_matrix.sh").read_text()

    assert "set -euo pipefail" in script
    assert "workflow.stage=plot_feature_matrix" in script
    assert "train_shadows" not in script
    assert "attack_control" not in script
    assert "plot_rmia_feature" not in script
    assert "plot_lira_feature" not in script
    assert "_SUCCESS" in script
