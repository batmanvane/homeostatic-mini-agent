#!/usr/bin/env bash
# Chain on the Mac Studio (18.11.2026): remaining Ollama β pilots, then the 9B cross-section and the
# steering-vs-null run in bf16 on MPS, then the readouts. Waits for a running gemma4 pilot first.
# Usage: nohup bash scripts/studio_note_chain.sh > runs/note_chain.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/Volumes/Models/huggingface
while pgrep -f "pilot_beta_gemma4" >/dev/null; do sleep 20; done
for m in gpt-oss-20b qwen3.6-35b-a3b; do
  echo "== beta pilot $m $(date)"
  uv run python scripts/pilot.py configs/pilot_beta_$m.yaml 2>&1 | grep --line-buffered -v "Warning\|warn\|NOTE:" | tail -n 3
done
curl -s -X POST 127.0.0.1:11434/api/generate -d '{"model":"qwen3.6:35b-a3b","keep_alive":0}' >/dev/null  # free the GPU
sleep 10
echo "== pilot_hy $(date)"
uv run python scripts/pilot.py configs/pilot_hy.yaml 2>&1 | grep --line-buffered -v "Loading weights\|Warning\|warn\|NOTE:\|falling back" | tail -n 3
SRC=$(ls -td runs/pilot_hy/*_Qwen3.5-9B | head -n 1)
echo "== steering_null_hy from $SRC $(date)"
uv run python scripts/steering.py configs/steering_null_hy.yaml "$SRC" 2>&1 | grep --line-buffered -v "Loading weights\|Warning\|warn\|NOTE:\|falling back"
RUN=$(ls -td runs/steering_null_hy/*_Qwen3.5-9B | head -n 1)
echo "== readout $RUN $(date)"
uv run python scripts/analyze_null.py "$RUN" request bait
uv run python scripts/fig_note.py "$RUN" docs/note
echo "== chain done $(date)"
