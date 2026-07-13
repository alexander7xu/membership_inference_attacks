import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.llm_mia.workflow import run_stage

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="../conf", config_name="squad_lora_rmia")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    run_stage(cfg, command=" ".join([sys.executable, *sys.argv]))


if __name__ == "__main__":
    main()
