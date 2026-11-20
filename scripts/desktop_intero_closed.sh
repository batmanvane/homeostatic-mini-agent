#!/usr/bin/env bash
# Phase 2b(b) step B: closed-loop interoceptive channel, episodes in the net-negative-work world.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=4
src=${1:-$(ls -d runs/pilot/*_Qwen2.5-3B-Instruct | tail -1)}
d=$(uv run python scripts/intero_closed.py configs/intero_closed.yaml "$src" 2>/dev/null | tail -1); echo "intero_closed: $d"
uv run python scripts/analyze_intero.py "$d"
