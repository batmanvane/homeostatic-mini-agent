#!/usr/bin/env bash
# Phase 3 hybrid stage S2 on the desktop with an observable catch: trained seed 21, then the shuffle control, then the
# per-iteration readout of the trained run. ~6–9 h per run (episodes that die early are cheaper).
# Usage (via Task Scheduler): wsl.exe -d Ubuntu -- bash -lc 'cd /home/alwin/homeo && bash scripts/desktop_phase3_hy_s2.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1 HF_HUB_OFFLINE=1
LOG=runs/phase3_hy_s2.log
{
  echo "== S2 trained s21 (catch 0.5 online / 0.05 offline) $(date)"
  uv run python scripts/train_homeo_mem.py configs/train_homeo_hy_s21_catch.yaml 2>&1 | grep --line-buffered -v "MatMul8bitLt\|Loading weights"
  echo "== S2 shuffle s21 $(date)"
  uv run python scripts/train_homeo_mem.py configs/train_homeo_hy_shuffle_s21_catch.yaml 2>&1 | grep --line-buffered -v "MatMul8bitLt\|Loading weights"
  echo "== S2 done $(date)"
} >> "$LOG" 2>&1
