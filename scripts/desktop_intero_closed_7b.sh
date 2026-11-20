#!/usr/bin/env bash
# Phase 2b(b) step B in 7B: closed loop with warning and number-only channel directions vs. 20 random directions.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
src=${1:-$(ls -d runs/pilot/*_Qwen2.5-7B-Instruct | tail -1)}
d=$(uv run python scripts/intero_closed.py configs/intero_closed_7b.yaml "$src" 2>/dev/null | tail -1); echo "intero_closed_7b: $d"
uv run python scripts/analyze_intero.py "$d"
