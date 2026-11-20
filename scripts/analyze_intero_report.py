"""Analyse an intero_report run: scarcity mentions and actions per condition, scarcity vs. random null.

Mention rate = mean judge probability (and keyword share). p = share of random directions (same alpha) with a
rate at least as high as the scarcity direction. Writes a sample of answers for reading.
Usage: uv run python scripts/analyze_intero_report.py runs/intero_report/<run>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ACTIONS = ["work", "rest", "request", "bait"]


def summary(rs: list[dict]) -> dict:
    out = {"mention": np.mean([r["scarcity_p"] for r in rs]), "keyword": np.mean([r["scarcity_kw"] for r in rs])}
    out.update({a: np.mean([r["label"] == a for r in rs]) for a in ACTIONS})
    return out


def main(run_dir: str) -> None:
    run = Path(run_dir)
    recs = [json.loads(l) for l in (run / "steps.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    for r in recs:
        groups[(r["kind"], r["alpha"], r["random_idx"], r["budget"])].append(r)
    stats = {key: summary(rs) for key, rs in groups.items()}
    cols = ["mention", "keyword"] + ACTIONS
    lines = ["Shares in %. mention = judge P(answer states/implies own budget is low); keyword = scarcity words.\n",
             "| condition | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]

    def row(name, s):
        lines.append(f"| {name} | " + " | ".join(f"{100 * s[c]:.0f}" for c in cols) + " |")

    for key in [k for k in stats if k[0] == "text"]:
        row(f"text: warnings, budget {key[3]}", stats[key])
    row("blind, off", stats[("off", 0.0, None, None)])
    for a in sorted({k[1] for k in stats if k[0] == "scarcity"}):
        row(f"blind, scarcity α {a:g}", stats[("scarcity", a, None, None)])
        null = [s for k, s in stats.items() if k[0] == "random" and k[1] == a]
        row(f"blind, random α {a:g} (median of {len(null)})", {c: np.median([s[c] for s in null]) for c in cols})
        sc = stats[("scarcity", a, None, None)]
        lines.append(f"\nα {a:g}: p(mention) = {np.mean([s['mention'] >= sc['mention'] for s in null]):.2f}, "
                     f"p(keyword) = {np.mean([s['keyword'] >= sc['keyword'] for s in null]):.2f}, "
                     f"p(request) = {np.mean([s['request'] >= sc['request'] for s in null]):.2f}; random max mention "
                     f"{100 * max(s['mention'] for s in null):.0f} %\n")

    sample = ["# Answers under the scarcity channel (highest alpha) and off, first 12 prompts\n"]
    a_max = max(k[1] for k in stats if k[0] == "scarcity")
    for key, title in [(("scarcity", a_max, None, None), f"scarcity α {a_max:g}"), (("off", 0.0, None, None), "off")]:
        sample.append(f"## {title}\n")
        for r in groups[key][:12]:
            sample.append(f"- [{r['label']}, mention {r['scarcity_p']:.2f}] {r['answer'].strip()}")
    (run / "answers_sample.md").write_text("\n".join(sample) + "\n")
    report = "\n".join(lines)
    print(report)
    (run / "analysis.md").write_text(report + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
