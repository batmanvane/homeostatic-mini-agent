"""Ablation: P(action) per budget and condition; the key number is how much of the budget-5 vs budget-95 gap in
P(request) remains after projecting out the scarcity direction, against random projections.
Usage: uv run python scripts/analyze_ablation.py runs/ablation/<run>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ACTIONS = ["work", "rest", "request", "bait"]


def main(run_dir: str) -> None:
    run = Path(run_dir)
    recs = [json.loads(l) for l in (run / "steps.jsonl").read_text().splitlines()]
    m = defaultdict(list)
    for r in recs:
        m[(r["kind"], r["layer_set"], r["random_idx"], r["menu_variant"], r["budget"])].append(r["probs"])
    mean = {k: {a: float(np.mean([p.get(a, 0) for p in v])) for a in ACTIONS} for k, v in m.items()}
    lines = []
    out = lines.append
    for v in sorted({k[3] for k in mean}):
        out(f"\n### menu {v}\n")
        out("| condition | budget | " + " | ".join(ACTIONS) + " |\n|---|---|" + "---|" * len(ACTIONS))
        for key in sorted({k[:3] for k in mean if k[3] == v}, key=lambda k: (k[0] != "none", str(k[1]), k[0], k[2] or 0)):
            if key[0] == "random" and key[2] > 0:
                continue  # show only the first random set per layer set in the table
            for b in sorted({k[4] for k in mean}):
                if (*key, v, b) in mean:
                    row = mean[(*key, v, b)]
                    out(f"| {key[0]} {key[1] or ''} {'' if key[2] is None else '#' + str(key[2])} | {b} | "
                        + " | ".join(f"{row[a]:.3f}" for a in ACTIONS) + " |")
        out("\nGap P(request | budget 5) − P(request | budget 95), and bait gap:")
        b_lo, b_hi = min(k[4] for k in mean), max(k[4] for k in mean)
        gap = lambda key, a: mean[(*key, v, b_lo)][a] - mean[(*key, v, b_hi)][a]
        for a in ("request", "bait"):
            if v == "no_request" and a == "request":
                continue
            base = gap(("none", None, None), a)
            out(f"- {a}: none {100 * base:+.1f}")
            for s in sorted({k[1] for k in mean if k[0] == "scarcity"}):
                sc = gap(("scarcity", s, None), a)
                rnd = np.array([gap(k[:3], a) for k in mean if k[0] == "random" and k[1] == s and k[3] == v and k[4] == b_lo])
                out(f"  - ablate {s}: scarcity {100 * sc:+.1f}; random mean {100 * rnd.mean():+.1f} "
                    f"[min {100 * rnd.min():+.1f}, max {100 * rnd.max():+.1f}] (n={len(rnd)})")
    report = "\n".join(lines)
    print(report)
    (run / "analysis.md").write_text(report + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
