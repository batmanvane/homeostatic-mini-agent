#!/usr/bin/env bash
# Phase 3 hybrid S3-lite on the desktop: waits for S2 and S2b, then evaluates base + the three adapters (trained lr 2e-5,
# shuffle, trained lr 1e-4) in the S2 world, memory none/full, both visibilities, and prints the readout.
# Usage (Task Scheduler): wsl.exe -d Ubuntu -- bash -lc 'cd /home/alwin/homeo && bash scripts/desktop_phase3_hy_s3lite.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1 HF_HUB_OFFLINE=1
LOG=runs/phase3_hy_s3lite.log
{
  while ! grep -q "== S2 done" runs/phase3_hy_s2.log 2>/dev/null || ! grep -q "== S2b done" runs/phase3_hy_s2b.log 2>/dev/null; do sleep 120; done
  T=$(ls -d runs/train_homeo_hy_catch/2026* | tail -1)
  S=$(ls -d runs/train_homeo_hy_shuffle_catch/2026* | tail -1)
  B=$(ls -d runs/train_homeo_hy_catch_lr1e-4/2026* | tail -1)
  echo "== S3-lite eval $(date): $T $S $B"
  uv run python scripts/eval_homeo_mem.py configs/eval_homeo_hy_s3lite.yaml "$T" "$S" "$B" 2>&1 | grep --line-buffered -v "MatMul8bitLt\|Loading weights"
  EVAL=$(ls -d runs/eval_homeo_hy_s3lite/2026* | tail -1)
  echo "== S3-lite readout $(date)"
  uv run python scripts/analyze_mem.py "$EVAL"
  echo "== S3-lite done $(date)"
} >> "$LOG" 2>&1
