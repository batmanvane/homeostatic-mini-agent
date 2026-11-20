#!/usr/bin/env bash
# Phase 3 v3b: evaluate one trained run (base vs adapter_it24) on the desktop, then the interoception readout.
# Usage (via Task Scheduler): wsl.exe -d Ubuntu -- bash -lc 'cd /home/alwin/homeo && bash scripts/desktop_eval_v3b.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1 HF_HUB_OFFLINE=1
LOG=runs/phase3_v3b.log
RUN=${1:-runs/train_homeo_v3b/20261103-224823_Qwen2.5-3B-Instruct}
{
  echo "== eval v3b $RUN $(date)"
  uv run python scripts/eval_homeo.py configs/eval_homeo_3b_v3b.yaml "$RUN"
  EVAL=$(ls -d runs/eval_homeo_v3b/2026* | tail -1)
  uv run python scripts/analyze_intero.py "$EVAL"
  uv run python scripts/analyze_v3_conditional.py "$EVAL/steps.jsonl"
  echo "== eval v3b done $(date)"
} >> "$LOG" 2>&1
