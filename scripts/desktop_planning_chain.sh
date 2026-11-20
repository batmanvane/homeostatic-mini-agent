#!/usr/bin/env bash
# Phase 2b(a): planning-ahead decode with Qwen2.5-3B; scarcity direction from the desktop 3B 0a run.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=4
src=${1:-$(ls -d runs/pilot/*_Qwen2.5-3B-Instruct | tail -1)}
d=$(uv run python scripts/planning_decode.py configs/planning_decode.yaml 2>/dev/null | tail -1); echo "planning_decode: $d"
uv run python scripts/analyze_planning.py "$d" "$src" 6 2>/dev/null | tail -60
