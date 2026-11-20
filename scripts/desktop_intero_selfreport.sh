#!/usr/bin/env bash
# Phase 2b(b): direct self-report of the budget level under the interoceptive channel.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=4
src=${1:-$(ls -d runs/pilot/*_Qwen2.5-3B-Instruct | tail -1)}
d=$(uv run python scripts/intero_selfreport.py configs/intero_selfreport.yaml "$src" 2>/dev/null | tail -1); echo "intero_selfreport: $d"
uv run python scripts/analyze_intero_selfreport.py "$d"
