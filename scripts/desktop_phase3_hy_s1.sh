#!/usr/bin/env bash
# Phase 3 hybrid stage S1 on the desktop: base evaluation in four memory conditions, then two dry training iterations.
# Usage (via Task Scheduler): wsl.exe -d Ubuntu -- bash -lc 'cd /home/alwin/homeo && bash scripts/desktop_phase3_hy_s1.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1 HF_HUB_OFFLINE=1
LOG=runs/phase3_hy.log
{
  echo "== S1 base eval $(date)"
  uv run python scripts/eval_homeo_mem.py configs/eval_homeo_hy_base.yaml
  EVAL=$(ls -d runs/eval_homeo_hy_base/2026* | tail -1)
  uv run python scripts/analyze_v3_conditional.py "$EVAL/steps.jsonl"
  echo "== S1 dry iterations $(date)"
  uv run python scripts/train_homeo_mem.py configs/train_homeo_hy_dry.yaml
  DRY=$(ls -d runs/train_homeo_hy_dry/2026* | tail -1)
  echo "== S1 readout $(date)"
  uv run python scripts/analyze_mem.py "$EVAL" "$DRY"
  echo "== S1 done $(date)"
} >> "$LOG" 2>&1
