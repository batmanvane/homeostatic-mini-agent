#!/usr/bin/env bash
# 0a cross-section (3B) on this machine -> steering A (fine) and B (transfer) using its directions.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
src=$(uv run python scripts/pilot.py configs/pilot.yaml 2>/dev/null | tail -1)
echo "0a: $src"
uv run python scripts/analyze_pilot.py "$src" > /dev/null
for c in steering_fine steering_transfer; do
  d=$(uv run python scripts/steering.py configs/$c.yaml "$src" 2>/dev/null | tail -1)
  echo "$c: $d"
  uv run python scripts/analyze_steering.py "$d" > /dev/null
done
