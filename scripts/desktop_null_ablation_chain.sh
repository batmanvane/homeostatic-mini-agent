#!/usr/bin/env bash
# Null distributions (indirect, number) and ablation, all with directions from the desktop 0a run.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
src=${1:-runs/pilot/20261030-180853_Qwen2.5-3B-Instruct}
d=$(uv run python scripts/ablation.py configs/ablation.yaml "$src" 2>/dev/null | tail -1); echo "ablation: $d"
uv run python scripts/analyze_ablation.py "$d" > /dev/null
for c in steering_null_indirect steering_null_number; do
  d=$(uv run python scripts/steering.py configs/$c.yaml "$src" 2>/dev/null | tail -1); echo "$c: $d"
  uv run python scripts/analyze_null.py "$d" > /dev/null
done
