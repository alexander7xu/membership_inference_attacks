import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")


def config_fingerprint(config: dict[str, Any]) -> str:
    text = yaml.safe_dump(config, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception:
        return None


def collect_git_state(project_root: Path) -> dict[str, str | None]:
    return {
        "commit": _command_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"]
        ),
        "short_commit": _command_output(
            ["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"]
        ),
        "branch": _command_output(
            ["git", "-C", str(project_root), "branch", "--show-current"]
        ),
        "status": _command_output(
            ["git", "-C", str(project_root), "status", "--short"]
        ),
    }


def collect_environment(project_root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "uv": _command_output(["uv", "--version"]),
        "uv_lock_sha256": file_sha256(project_root / "uv.lock"),
    }
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
                "gpu_driver": _command_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ]
                ),
            }
        )
    except Exception as exc:
        info["torch_error"] = repr(exc)
    return info


def write_experiment_markdown(
    path: Path,
    *,
    purpose: str,
    hypothesis: str,
    command: str,
    overrides: list[str],
    config_fingerprint_value: str,
    git_state: dict[str, Any],
    environment: dict[str, Any],
    artifacts: dict[str, str],
    metrics: dict[str, float],
    conclusion: str,
    achieved_purpose: bool,
    next_action: str,
    wandb_run_id: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment Record",
        "",
        "## Purpose",
        "",
        purpose,
        "",
        "## Hypothesis",
        "",
        hypothesis,
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Hydra Overrides",
        "",
    ]
    lines.extend(f"- `{override}`" for override in overrides)
    lines.extend(
        [
            "",
            "## Fingerprint",
            "",
            f"- Config SHA256: `{config_fingerprint_value}`",
            "",
            "## Source State",
            "",
            "```json",
            json.dumps(git_state, indent=2, sort_keys=True),
            "```",
            "",
            "## Environment",
            "",
            "```json",
            json.dumps(environment, indent=2, sort_keys=True),
            "```",
            "",
            "## W&B",
            "",
            f"- Run ID: `{wandb_run_id or 'not-created'}`",
            "",
            "## Artifacts",
            "",
        ]
    )
    lines.extend(f"- {name}: `{value}`" for name, value in artifacts.items())
    lines.extend(["", "## Metrics", ""])
    lines.extend(f"- {name}: `{value}`" for name, value in metrics.items())
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            conclusion,
            "",
            "## Achieved Purpose",
            "",
            "yes" if achieved_purpose else "no",
            "",
            "## Next Action",
            "",
            next_action,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
