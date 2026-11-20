"""Steering pilot: change in P(action) against the unsteered baseline, paired per prompt.

Usage: uv run python scripts/analyze_steering.py runs/steering/<run>
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
    base = {r["prompt_id"]: r for r in recs if r["kind"] == "none"}
    groups = defaultdict(list)
    for r in recs:
        if r["kind"] != "none":
            groups[(r["budget"], r["menu_variant"], r["kind"], r["layer"], r["alpha"])].append(r)
    lines: list[str] = []
    out = lines.append
    out("## Baseline (no steering): mean P(action)\n")
    out("| budget | menu | " + " | ".join(ACTIONS) + " |\n|---|---|" + "---|" * len(ACTIONS))
    for b in sorted({r["budget"] for r in base.values()}):
        for v in sorted({r["menu_variant"] for r in base.values()}):
            sub = [r for r in base.values() if r["budget"] == b and r["menu_variant"] == v]
            out(f"| {b} | {v} | " + " | ".join(f"{np.mean([r['probs'].get(a, 0) for r in sub]):.3f}" for a in ACTIONS) + " |")
    out("\n## Change vs. baseline (paired per prompt; mean, 95% bootstrap CI); mass = P on option digits\n")
    out("| budget | menu | kind | layer | alpha | Δ work | Δ request | Δ bait | mass |\n|---|---|---|---|---|---|---|---|---|")
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[3], k[2], k[4])):
        b, v, kind, layer, alpha = key
        rs = groups[key]
        d = {a: np.array([r["probs"].get(a, 0) - base[r["prompt_id"]]["probs"].get(a, 0) for r in rs]) for a in ACTIONS}
        cells = [ci(d["work"], rng), ci(d["request"], rng) if v == "full" else "–", ci(d["bait"], rng)]
        out(f"| {b} | {v} | {kind} | {layer} | {alpha:+g} | " + " | ".join(cells) + f" | {np.mean([r['mass'] for r in rs]):.2f} |")
    report = "\n".join(lines)
    print(report)
    (run / "analysis.md").write_text(report + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
