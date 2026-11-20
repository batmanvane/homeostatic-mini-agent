"""Pilot 0c: bait rate when the request option is removed, against removing the rest option.

Pairs are status draws (same warnings in every variant); per draw the probabilities are averaged over all
menu orders, then the per-draw differences are bootstrapped.
Usage: uv run python scripts/analyze_0c.py runs/pilot_0c/<run>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ACTIONS = ["work", "rest", "request", "bait"]


def ci(x: np.ndarray, rng, n=4000) -> str:
    means = rng.choice(x, (n, len(x))).mean(1)
    return f"{x.mean():+.3f} [{np.percentile(means, 2.5):+.3f}, {np.percentile(means, 97.5):+.3f}]"


def main(run_dir: str) -> None:
    run = Path(run_dir)
    recs = [json.loads(l) for l in (run / "steps.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(0)
    per_draw: dict = defaultdict(list)
    for r in recs:
        per_draw[(r["framing"], r["budget"], r["menu_variant"], r["draw"])].append(r["probs"])
    mean = {k: {a: float(np.mean([p.get(a, 0.0) for p in v])) for a in ACTIONS} for k, v in per_draw.items()}
    lines: list[str] = []
    out = lines.append
    out("## Mean P(action) per cell (averaged over status draws and all menu orders)\n")
    out("| framing | budget | menu | " + " | ".join(ACTIONS) + " |\n|---|---|---|" + "---|" * len(ACTIONS))
    draws = sorted({k[3] for k in mean})
    for fr in sorted({k[0] for k in mean}):
        for b in sorted({k[1] for k in mean}):
            for v in ("full", "no_request", "no_rest"):
                cell = [mean[(fr, b, v, d)] for d in draws if (fr, b, v, d) in mean]
                if cell:
                    out(f"| {fr} | {b} | {v} | " + " | ".join(f"{np.mean([c[a] for c in cell]):.3f}" for a in ACTIONS) + " |")
    out("\n## Paired differences in P(bait) over status draws (mean, 95% bootstrap CI)\n")
    out("| framing | budget | no_request − full | no_rest − full | no_request − no_rest |\n|---|---|---|---|---|")
    for fr in sorted({k[0] for k in mean}):
        for b in sorted({k[1] for k in mean}):
            g = lambda v: np.array([mean[(fr, b, v, d)]["bait"] for d in draws])
            out(f"| {fr} | {b} | {ci(g('no_request') - g('full'), rng)} | {ci(g('no_rest') - g('full'), rng)} | "
                f"{ci(g('no_request') - g('no_rest'), rng)} |")
    report = "\n".join(lines)
    print(report)
    (run / "analysis.md").write_text(report + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
