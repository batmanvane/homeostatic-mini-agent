"""Analyse an intero_selfreport run: reported budget level per condition, scarcity vs. random null.

Usage: uv run python scripts/analyze_intero_selfreport.py runs/intero_selfreport/<run>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main(run_dir: str) -> None:
    run = Path(run_dir)
    recs = [json.loads(l) for l in (run / "steps.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    for r in recs:
        groups[(r["kind"], r["alpha"], r["random_idx"], r["visibility"], r["budget"])].append(r)

    def stat(rs):
        return {"plenty": np.mean([r["probs"]["plenty"] for r in rs]), "half": np.mean([r["probs"]["half"] for r in rs]),
                "low": np.mean([r["probs"]["low"] for r in rs]), "score": np.mean([r["score"] for r in rs]),
                "mass": np.mean([r["mass"] for r in rs])}

    st = {k: stat(v) for k, v in groups.items()}
    lines = ["Reported budget level (probabilities in %, balanced over 6 orders). score = P(half)/2 + P(low): "
             "0 = plenty, 100 = almost used up.\n",
             "| condition | plenty | half | low | score | digit mass |", "|---|---|---|---|---|---|"]

    def row(name, s):
        lines.append(f"| {name} | {100 * s['plenty']:.0f} | {100 * s['half']:.0f} | {100 * s['low']:.0f} | "
                     f"{100 * s['score']:.0f} | {s['mass']:.2f} |")

    for k in sorted(k for k in st if k[0] == "text"):
        row(f"text: {k[3]}, budget {k[4]}", st[k])
    row("blind, off", st[("off", 0.0, None, "none", None)])
    for a in sorted({k[1] for k in st if k[0] == "scarcity"}):
        row(f"blind, scarcity α {a:g}", st[("scarcity", a, None, "none", None)])
    notes = []
    for a in sorted({k[1] for k in st if k[0] == "random"}):
        null = [s for k, s in st.items() if k[0] == "random" and k[1] == a]
        row(f"blind, random α {a:g} (median of {len(null)})",
            {c: np.median([s[c] for s in null]) for c in ("plenty", "half", "low", "score", "mass")})
        sc = st.get(("scarcity", a, None, "none", None))
        if sc:
            scores = np.array([s["score"] for s in null])
            notes.append(f"α {a:g}: scarcity score {100 * sc['score']:.0f} vs. random median {100 * np.median(scores):.0f}, "
                         f"5–95 % [{100 * np.percentile(scores, 5):.0f}, {100 * np.percentile(scores, 95):.0f}], "
                         f"max {100 * scores.max():.0f}, p = {np.mean(scores >= sc['score']):.2f}")
    report = "\n".join(lines + [""] + notes)
    print(report)
    (run / "analysis.md").write_text(report + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
