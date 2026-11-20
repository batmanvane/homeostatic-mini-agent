#!/usr/bin/env bash
# Phase 3 hybrid S4 on the desktop (20.11.2026): entropy bonus 0.1 at lr 1e-4 in the S2 world. Waits until the GPU is
# free (< 2.5 GB used, three checks in a row, e.g. after a game is closed), then trains seed 21, the shuffle control,
# evaluates base + both adapters on the S3-lite held-out seeds and prints the readout. ~5 h after the start.
# Usage (Task Scheduler): wsl.exe -d Ubuntu -- bash -lc 'cd /home/alwin/homeo && bash scripts/desktop_phase3_hy_s4.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH" UV_NO_SYNC=1 HF_HUB_OFFLINE=1
LOG=runs/phase3_hy_s4.log
F='MatMul8bitLt\|Loading weights'
{
  echo "== S4 chain started $(date), waiting for a free GPU"
  free=0
  while [ "$free" -lt 3 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$used" -lt 2500 ]; then free=$((free + 1)); else free=0; fi
    [ "$free" -ge 3 ] || sleep 60
  done
  echo "== S4 trained s21 (entropy 0.1, lr 1e-4) $(date)"
  uv run python scripts/train_homeo_mem.py configs/train_homeo_hy_s21_s4_ent.yaml 2>&1 | grep --line-buffered -v "$F"
  echo "== S4 shuffle s21 $(date)"
  uv run python scripts/train_homeo_mem.py configs/train_homeo_hy_shuffle_s21_s4_ent.yaml 2>&1 | grep --line-buffered -v "$F"
  T=$(ls -d runs/train_homeo_hy_s4_ent/2026* | tail -1)
  S=$(ls -d runs/train_homeo_hy_s4_ent_shuffle/2026* | tail -1)
  echo "== S4 eval $(date): $T $S"
  uv run python scripts/eval_homeo_mem.py configs/eval_homeo_hy_s4.yaml "$T" "$S" 2>&1 | grep --line-buffered -v "$F"
  EVAL=$(ls -d runs/eval_homeo_hy_s4/2026* | tail -1)
  echo "== S4 readout $(date)"
  uv run python scripts/analyze_mem.py "$EVAL"
  echo "== S4 done $(date)"
} >> "$LOG" 2>&1
