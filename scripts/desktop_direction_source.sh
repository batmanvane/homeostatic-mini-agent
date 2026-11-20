#!/usr/bin/env bash
# Phase 2b(b) confound test in 7B: warning vs. number-only scarcity direction (cosines, self-report, action choice).
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
src=${1:-$(ls -d runs/pilot/*_Qwen2.5-7B-Instruct | tail -1)}
uv run python scripts/compare_directions.py "$src"
for c in intero_selfreport_7b_num intero_selfreport_7b_num_band; do
  d=$(uv run python scripts/intero_selfreport.py configs/$c.yaml "$src" 2>/dev/null | tail -1); echo "$c: $d"
  uv run python scripts/analyze_intero_selfreport.py "$d"
done
for c in intero_open_7b intero_open_7b_num intero_open_7b_num_band; do
  d=$(uv run python scripts/steering.py configs/$c.yaml "$src" 2>/dev/null | tail -1); echo "$c: $d"
  uv run python scripts/analyze_null.py "$d" work rest request bait
done
