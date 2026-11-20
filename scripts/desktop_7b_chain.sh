#!/usr/bin/env bash
# 1) mean ablation for 3B; meanwhile download Qwen2.5-7B. 2) 7B: 0a cross-section, 0c, fine steering sweep.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=4  # sklearn probes oversubscribe the 20 cores otherwise
src3=${1:-runs/pilot/20261030-180853_Qwen2.5-3B-Instruct}
M7=Qwen/Qwen2.5-7B-Instruct
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$M7')" > runs/download_7b.log 2>&1 &
dl=$!
d=$(uv run python scripts/ablation.py configs/ablation_mean.yaml "$src3" 2>/dev/null | tail -1); echo "ablation_mean 3B: $d"
uv run python scripts/analyze_ablation.py "$d" > /dev/null
wait $dl; echo "7B downloaded"
src=$(uv run python scripts/pilot.py configs/pilot.yaml $M7 2>runs/7b_0a.err | tail -1); echo "0a 7B: $src"
uv run python scripts/analyze_pilot.py "$src" > /dev/null 2>&1
d=$(uv run python scripts/pilot.py configs/pilot_0c.yaml $M7 2>/dev/null | tail -1); echo "0c 7B: $d"
uv run python scripts/analyze_0c.py "$d" > /dev/null
d=$(uv run python scripts/steering.py configs/steering_fine_7b.yaml "$src" 2>/dev/null | tail -1); echo "steering_fine 7B: $d"
uv run python scripts/analyze_steering.py "$d" > /dev/null
