"""Run the agent loop. Usage: uv run python scripts/run_episode.py configs/episode.yaml"""

import sys
import time

from homeo.agent import RunConfig, run
from homeo.model import LM

cfg = RunConfig.from_yaml(sys.argv[1] if len(sys.argv) > 1 else "configs/episode.yaml")
cfg.out_dir = f"{cfg.out_dir}/{time.strftime('%Y%m%d-%H%M%S')}"
print(run(cfg, LM(cfg.model, cfg.device, cfg.dtype)))
