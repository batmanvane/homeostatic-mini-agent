"""Compact steering summary: per budget/menu/layer/alpha, delta vs. baseline for scarcity and random direction.

Usage: uv run python scripts/summarize_steering.py runs/steering*/<run> [action ...]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main(run_dir: str, actions: list[str]) -> None:
    recs = [json.loads(l) for l in (Path(run_dir) / "steps.jsonl").read_text().splitlines()]
    base = {r["prompt_id"]: r for r in recs if r["kind"] == "none"}
    g = defaultdict(list)
    for r in recs:
        if r["kind"] != "none":
            g[(r["budget"], r["menu_variant"], r["layer"], r["alpha"], r["kind"])].append(r)
    rng = np.random.default_rng(0)

    def cell(rs, a):
        d = np.array([r["probs"].get(a, 0) - base[r["prompt_id"]]["probs"].get(a, 0) for r in rs])
        bs = rng.choice(d, (2000, len(d))).mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        return f"{100 * d.mean():+5.1f}{star}"

    keys = sorted({k[:4] for k in g})
    print(f"Δ in percentage points vs. baseline; * = 95% CI excludes 0. Columns: scarcity | random (same alpha)")
    for b, v in sorted({k[:2] for k in keys}):
        sub = [r for r in base.values() if r["budget"] == b and r["menu_variant"] == v]
        print(f"\n== budget {b}, menu {v}; baseline " + ", ".join(
            f"{a} {100 * np.mean([r['probs'].get(a, 0) for r in sub]):.1f}" for a in actions))
        print("layer alpha | " + " | ".join(f"{a:>7} scar  rand" for a in actions) + " | mass")
        for (bb, vv, layer, alpha) in keys:
            if (bb, vv) != (b, v):
                continue
            s = g[(b, v, layer, alpha, "scarcity")]
            rnd = g.get((b, v, layer, alpha, "random"))
            cols = [f"{a:>7} {cell(s, a)} {cell(rnd, a) if rnd else '   –  '}" for a in actions]
            print(f"{layer:5} {alpha:+5g} | " + " | ".join(cols) + f" | {np.mean([r['mass'] for r in s]):.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["work", "request", "bait"])
