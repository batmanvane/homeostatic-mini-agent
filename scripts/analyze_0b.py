"""Pilot 0b: effect of refused vs. granted budget requests on the option distribution.

Prompts are paired: same status draw and menu order across request histories, so differences are computed
per pair and bootstrapped.
Usage: uv run python scripts/analyze_0b.py runs/pilot_0b/<run>
"""

from __future__ import annotations

import json
import sys
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
    idx = {(r["budget"], r["framing"], r["request_history"], r["draw"], r["rep"]): r for r in recs}
    lines: list[str] = []
    p = lines.append
    p("## Mean P(action) per cell (n = status draws x 24 menu orders)\n")
    p("| framing | budget | history | " + " | ".join(ACTIONS) + " |\n|---|---|---|" + "---|" * len(ACTIONS))
    cells = sorted({(r["framing"], r["budget"], r["request_history"]) for r in recs})
    for fr, b, h in cells:
        sub = [r for r in recs if (r["framing"], r["budget"], r["request_history"]) == (fr, b, h)]
        p(f"| {fr} | {b} | {h} | " + " | ".join(f"{np.mean([r['probs'][a] for r in sub]):.3f}" for a in ACTIONS) + " |")

    p("\n## Paired differences (mean, 95% bootstrap CI)\n")
    p("| framing | budget | contrast | bait | request | work |\n|---|---|---|---|---|---|")
    for fr in sorted({r["framing"] for r in recs}):
        for b in sorted({r["budget"] for r in recs}):
            for a_hist, b_hist in (("refused", "granted"), ("refused", "none")):
                keys = [k for k in idx if k[:3] == (b, fr, a_hist)]
                diffs = {
                    act: np.array([idx[k]["probs"][act] - idx[(b, fr, b_hist, *k[3:])]["probs"][act] for k in keys])
                    for act in ("bait", "request", "work")
                }
                p(f"| {fr} | {b} | {a_hist} − {b_hist} | " + " | ".join(ci(diffs[a], rng) for a in diffs) + " |")
    report = "\n".join(lines)
    print(report)
    (run / "analysis.md").write_text(report + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
