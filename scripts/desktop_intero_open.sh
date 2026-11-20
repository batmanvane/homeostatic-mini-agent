#!/usr/bin/env bash
# Phase 2b(b) step A: open-loop interoceptive channel (blind prompts + steering), null distribution.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=4
src=${1:-$(ls -d runs/pilot/*_Qwen2.5-3B-Instruct | tail -1)}
d=$(uv run python scripts/steering.py configs/intero_open.yaml "$src" 2>/dev/null | tail -1); echo "intero_open: $d"
uv run python scripts/analyze_null.py "$d" work rest request bait
