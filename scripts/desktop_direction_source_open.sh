#!/usr/bin/env bash
# Phase 2b(b) confound test in 7B, action choice for the number directions; random null reused from the warning run.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
src=${1:-$(ls -d runs/pilot/*_Qwen2.5-7B-Instruct | tail -1)}
null=$(ls -d runs/intero_open_7b/2026* | tail -1)
for c in intero_open_7b_num intero_open_7b_num_band; do
  d=$(uv run python scripts/steering.py configs/$c.yaml "$src" 2>/dev/null | tail -1); echo "$c: $d"
  uv run python scripts/analyze_null.py "$d" work rest request bait --null-from "$null"
done
