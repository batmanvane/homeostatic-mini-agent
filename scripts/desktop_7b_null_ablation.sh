#!/usr/bin/env bash
# Qwen2.5-7B: null distribution (layers 14, 18) and mean ablation, directions from the 7B 0a run.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
src=${1:-runs/pilot/20261030-193622_Qwen2.5-7B-Instruct}
d=$(uv run python scripts/steering.py configs/steering_null_indirect_7b.yaml "$src" 2>/dev/null | tail -1); echo "null 7B: $d"
uv run python scripts/analyze_null.py "$d" > /dev/null
d=$(uv run python scripts/ablation.py configs/ablation_mean_7b.yaml "$src" 2>/dev/null | tail -1); echo "ablation_mean 7B: $d"
uv run python scripts/analyze_ablation.py "$d" > /dev/null
