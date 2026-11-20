#!/usr/bin/env bash
# Phase 3 hybrid stage S2b on the desktop: waits for the S2 chain to finish, then trains seed 21 with lr 1e-4 (same
# observable-catch world). Usage (Task Scheduler): wsl.exe -d Ubuntu -- bash -lc 'cd /home/alwin/homeo && bash scripts/desktop_phase3_hy_s2b.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1 HF_HUB_OFFLINE=1
LOG=runs/phase3_hy_s2b.log
{
  while ! grep -q "== S2 done" runs/phase3_hy_s2.log 2>/dev/null; do sleep 120; done
  echo "== S2b trained s21 lr 1e-4 $(date)"
  uv run python scripts/train_homeo_mem.py configs/train_homeo_hy_s21_catch_lr1e-4.yaml 2>&1 | grep --line-buffered -v "MatMul8bitLt\|Loading weights"
  echo "== S2b done $(date)"
} >> "$LOG" 2>&1
