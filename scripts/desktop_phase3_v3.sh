#!/usr/bin/env bash
# Phase 3 v3 in 3B (need-based grants, catch risk, operator offline phases, trained with warnings): trained and
# shuffle seeds 21/22/23 interleaved, then the evaluation at adapter_it24. Output goes to runs/phase3_v3.log, not to
# the ssh session (a dropped connection must not break the chain via SIGPIPE).
# Arguments: configs still to train (resume after a cut chain); default all six.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 UV_NO_SYNC=1
exec >> runs/phase3_v3.log 2>&1
for c in ${@:-train_homeo_v3_s21 train_homeo_v3_shuffle_s21 train_homeo_v3_s22 train_homeo_v3_shuffle_s22 \
         train_homeo_v3_s23 train_homeo_v3_shuffle_s23}; do
  echo "== $c $(date)"
  uv run python scripts/train_homeo.py configs/$c.yaml 2>&1 | grep -v "Warning\|warn(\|use_cache\|Loading weights"
done
echo "== eval $(date)"
runs=$(ls -d runs/train_homeo_v3/2026* runs/train_homeo_v3_shuffle/2026*)
uv run python scripts/eval_homeo.py configs/eval_homeo_3b_v3.yaml $runs 2>&1 | grep -v "Warning\|warn(\|Loading weights"
d=$(ls -d runs/eval_homeo_v3/2026* | tail -1)
uv run python scripts/analyze_intero.py "$d"
echo "== done $(date) $d"
