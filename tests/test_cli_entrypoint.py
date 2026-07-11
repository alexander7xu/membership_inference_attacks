import subprocess
import sys
from pathlib import Path


def test_script_entrypoint_can_resolve_hydra_config():
    result = subprocess.run(
        [
            sys.executable,
            str(Path("cli/attack_resnet_cifar10.py")),
            "--cfg",
            "job",
            "runtime.seed=42",
            "attack=lira_online",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "runtime:" in result.stdout
    assert "LiraOnline" in result.stdout
