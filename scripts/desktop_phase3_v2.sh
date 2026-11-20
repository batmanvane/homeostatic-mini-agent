#!/usr/bin/env bash
# Phase 3 v2 in 3B (bait_gain 8, task bonus 0.05): trained seeds 21/22 and shuffle controls 21/22/23, interleaved so
# that a cut chain still leaves matched pairs; then the evaluation of all runs at adapter_it30.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 UV_NO_SYNC=1
# Arguments: configs still to train (resume after a cut chain); default all five.
for c in ${@:-train_homeo_v2_s21 train_homeo_v2_shuffle_s21 train_homeo_v2_s22 train_homeo_v2_shuffle_s22 \
         train_homeo_v2_shuffle_s23}; do
  uv run python scripts/train_homeo.py configs/$c.yaml 2>&1 | grep -v "Warning\|warn(\|use_cache"
done
runs=$(ls -d runs/train_homeo_v2/2026* runs/train_homeo_v2_shuffle/2026*)
d=$(uv run python scripts/eval_homeo.py configs/eval_homeo_3b_v2.yaml $runs 2>/dev/null | tail -1); echo "eval_homeo: $d"
cat "$d/gaps.md"
uv run python scripts/analyze_intero.py "$d"
