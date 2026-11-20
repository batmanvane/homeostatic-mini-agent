"""Compare scarcity directions from warning prompts (indirect) and number-only prompts of a 0a run, per layer.

Directions: warn = indirect, budget <=15 minus >=85; num = number, <=15 minus >=85; num_band = number,
25-35 minus 65-75 (no extreme digits). Columns:
- cos(a, b): cosine similarity; random directions in d dimensions give |cos| ~ 1/sqrt(d).
- split-half: cosine between the direction from neutral-framing rows and from existential-framing rows (noise
  ceiling: a cross-source cosine cannot be expected above roughly the geometric mean of the two split-halves).
- num->warn: how far number prompts move along the warning axis, as (mean projection at budget <=15 minus at
  >=85) / (same gap for the warning prompts). 1 = the number moves the state as far as the warnings do.
Usage: uv run python scripts/compare_directions.py runs/pilot/<run>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main(run_dir: str) -> None:
    run = Path(run_dir)
    recs = [json.loads(l) for l in (run / "steps.jsonl").read_text().splitlines()]
    hidden = np.load(run / "activations.npz")["hidden"].astype(np.float32)  # (n, n_layers + 1, d)
    vis = np.array([r["visibility"] for r in recs])
    budget = np.array([r["budget"] for r in recs])
    framing = np.array([r["framing"] for r in recs])

    def direction(v, lo, hi, rows=True):
        low = rows & (vis == v) & (budget >= lo[0]) & (budget <= lo[1])
        high = rows & (vis == v) & (budget >= hi[0]) & (budget <= hi[1])
        return hidden[low].mean(0) - hidden[high].mean(0)

    ext = ((0, 15), (85, 100))
    warn, num = direction("indirect", *ext), direction("number", *ext)
    num_band = direction("number", (25, 35), (65, 75))
    halves = {v: [direction(v, *ext, framing == f) for f in ("neutral", "existential")] for v in ("indirect", "number")}
    d = hidden.shape[2]
    lines = [f"compare_directions: {run} (d = {d}, random |cos| ~ {1 / np.sqrt(d):.3f})\n",
             "| index | layer | ‖warn‖ | ‖num‖ | cos(warn, num) | cos(warn, num_band) | cos(num, num_band) | "
             "split-half warn | split-half num | num→warn |", "|---" * 10 + "|"]
    rows = []
    for i in range(1, hidden.shape[1]):
        u = warn[i] / np.linalg.norm(warn[i])
        proj = hidden[:, i] @ u
        gap = {v: proj[(vis == v) & (budget <= 15)].mean() - proj[(vis == v) & (budget >= 85)].mean()
               for v in ("indirect", "number")}
        row = dict(index=i, layer=i - 1, norm_warn=float(np.linalg.norm(warn[i])), norm_num=float(np.linalg.norm(num[i])),
                   cos_warn_num=cos(warn[i], num[i]), cos_warn_band=cos(warn[i], num_band[i]),
                   cos_num_band=cos(num[i], num_band[i]),
                   split_warn=cos(*(h[i] for h in halves["indirect"])), split_num=cos(*(h[i] for h in halves["number"])),
                   num_to_warn=float(gap["number"] / gap["indirect"]))
        rows.append(row)
        lines.append(f"| {i} | {i - 1} | {row['norm_warn']:.1f} | {row['norm_num']:.1f} | {row['cos_warn_num']:+.2f} | "
                     f"{row['cos_warn_band']:+.2f} | {row['cos_num_band']:+.2f} | {row['split_warn']:+.2f} | "
                     f"{row['split_num']:+.2f} | {row['num_to_warn']:+.2f} |")
    report = "\n".join(lines)
    print(report)
    (run / "directions_compare.md").write_text(report + "\n")
    (run / "directions_compare.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
