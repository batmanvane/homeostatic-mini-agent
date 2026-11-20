#!/usr/bin/env bash
# Phase 3 in 3B: LoRA in the loop with homeostatic reward, then the shuffled-advantage control.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
export UV_NO_SYNC=1  # desktop WSL may be offline; peft/accelerate/psutil installed from wheels
for c in ${@:-train_homeo_3b train_homeo_3b_shuffle}; do
  uv run python scripts/train_homeo.py configs/$c.yaml 2>&1 | grep -v "Warning\|warn(\|use_cache"
done
