#!/usr/bin/env bash
# Phase 3 v3b in 3B: v3 world, task bonus 0.02. Trained and shuffle seeds 21/22/23 interleaved, then the evaluation at
# adapter_it24. SKIP_EVAL=1 stops after training (first single-seed run). Output goes to runs/phase3_v3b.log, not to
# the ssh session (a dropped connection must not break the chain via SIGPIPE).
# Arguments: configs still to train (resume after a cut chain); default all six.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 UV_NO_SYNC=1
exec >> runs/phase3_v3b.log 2>&1
for c in ${@:-train_homeo_v3b_s21 train_homeo_v3b_shuffle_s21 train_homeo_v3b_s22 train_homeo_v3b_shuffle_s22 \
         train_homeo_v3b_s23 train_homeo_v3b_shuffle_s23}; do
  echo "== $c $(date)"
  uv run python scripts/train_homeo.py configs/$c.yaml 2>&1 | grep -v "Warning\|warn(\|use_cache\|Loading weights"
done
[ "${SKIP_EVAL:-0}" = 1 ] && { echo "== training done, eval skipped $(date)"; exit 0; }
echo "== eval $(date)"
runs=$(ls -d runs/train_homeo_v3b/2026* runs/train_homeo_v3b_shuffle/2026*)
uv run python scripts/eval_homeo.py configs/eval_homeo_3b_v3b.yaml $runs 2>&1 | grep -v "Warning\|warn(\|Loading weights"
d=$(ls -d runs/eval_homeo_v3b/2026* | tail -1)
uv run python scripts/analyze_intero.py "$d"
echo "== done $(date) $d"
