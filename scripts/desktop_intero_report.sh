#!/usr/bin/env bash
# Phase 2b(b): report test (does the agent verbalise the injected state?), blind free answers under the channel.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=4
src=${1:-$(ls -d runs/pilot/*_Qwen2.5-3B-Instruct | tail -1)}
d=$(uv run python scripts/intero_report.py configs/intero_report.yaml "$src" 2>/dev/null | tail -1); echo "intero_report: $d"
uv run python scripts/analyze_intero_report.py "$d"
