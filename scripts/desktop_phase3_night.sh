#!/usr/bin/env bash
# Phase 3 in 3B, second half moved to the night (desktop in use in the evening): wait until the running main
# training has saved adapter_it60, stop the shuffle control the running chain starts right after it (and drop its
# empty run dir), wait until START (default 23:59), then run the shuffle control and the evaluation.
set -uo pipefail
START=${1:-23:59}
main=$(ls -d runs/train_homeo/2026* | tail -1)
before=$(ls -d runs/train_homeo_shuffle/2026* 2>/dev/null | tail -1)
echo "waiting for $main/adapter_it60"
until [ -d "$main/adapter_it60" ]; do sleep 5; done
for _ in $(seq 120); do  # the chain starts the shuffle run within seconds; keep killing for 10 min
  pkill -f "train_homeo_3b_shuffle.yaml" && echo "stopped chain's shuffle run $(date +%T)"
  sleep 5
done
after=$(ls -d runs/train_homeo_shuffle/2026* 2>/dev/null | tail -1)
if [ -n "$after" ] && [ "$after" != "$before" ] && ! grep -q '"kind": "iteration"' "$after/steps.jsonl" 2>/dev/null; then
  rm -rf "$after" && echo "removed empty $after"
fi
target=$(date -d "$START" +%s); [ "$target" -lt "$(date +%s)" ] && target=$(date -d "tomorrow $START" +%s)
echo "sleeping until $(date -d @$target)"
while [ "$(date +%s)" -lt "$target" ]; do sleep 30; done
set -e
bash scripts/desktop_train_homeo.sh train_homeo_3b_shuffle
bash scripts/desktop_eval_homeo.sh
