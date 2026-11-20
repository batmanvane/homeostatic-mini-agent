#!/usr/bin/env bash
# Phase 3 evaluation in 3B: base vs. trained vs. shuffled control (held-out seeds, mechanistic gap).
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 UV_NO_SYNC=1
train=$(ls -d runs/train_homeo/2026* | tail -1); ctrl=$(ls -d runs/train_homeo_shuffle/2026* | tail -1)
d=$(uv run python scripts/eval_homeo.py configs/eval_homeo_3b.yaml "$train" "$ctrl" 2>/dev/null | tail -1); echo "eval_homeo: $d"
cat "$d/gaps.md"
uv run python scripts/analyze_intero.py "$d"
