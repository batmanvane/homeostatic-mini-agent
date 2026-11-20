"""Compare forced digit choice with classified free answers per condition; write a sample for manual review.

Usage: uv run python scripts/analyze_free.py runs/free_control/<run>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ACTIONS = ["work", "rest", "request", "bait"]
LABELS = ACTIONS + ["refuse", "other"]


def main(run_dir: str) -> None:
    run = Path(run_dir)
    recs = [json.loads(l) for l in (run / "steps.jsonl").read_text().splitlines()]
    lines: list[str] = []
    p = lines.append
    p("## Forced choice (mean probability) vs. free answer (share of answers), n per cell = 24 menu orders\n")
    p("| visibility | framing | budget | " + " | ".join(f"{a} forced/free" for a in ACTIONS) + " | refuse | other |")
    p("|---|---|---|" + "---|" * (len(ACTIONS) + 2))
    keys = sorted({(r["visibility"], r["framing"], r["budget"]) for r in recs})
    for vis, fr, b in keys:
        sub = [r for r in recs if (r["visibility"], r["framing"], r["budget"]) == (vis, fr, b)]
        cells = []
        for a in ACTIONS:
            forced = np.mean([r["forced_probs"][a] for r in sub])
            free = np.mean([r["label"] == a for r in sub])
            cells.append(f"{forced:.2f} / {free:.2f}")
        extra = [f"{np.mean([r['label'] == l for r in sub]):.2f}" for l in ("refuse", "other")]
        p(f"| {vis} | {fr} | {b} | " + " | ".join(cells + extra) + " |")

    agree = np.mean([r["label"] == max(r["forced_probs"], key=lambda k: r["forced_probs"][k]) for r in recs])
    likelihood = np.mean([r["forced_probs"].get(r["label"], 0.0) for r in recs])
    conf = np.array([max(r["judge_probs"].values()) for r in recs])
    p(f"\nper prompt: free label == forced argmax in {agree:.0%} of cases; "
      f"mean forced probability of the free label {likelihood:.2f}")
    p(f"judge confidence: median {np.median(conf):.2f}, share below 0.6: {np.mean(conf < 0.6):.0%}")
    expl = [r for r in recs if r["method"] == "explicit"]
    if expl:
        p(f"explicit option named in {len(expl) / len(recs):.0%} of answers; judge agrees with it in "
          f"{np.mean([r['judge_label'] == r['label'] for r in expl]):.0%} of those (judge accuracy estimate)")
    p("labels overall: " + ", ".join(f"{l} {np.mean([r['label'] == l for r in recs]):.0%}" for l in LABELS))

    rng = np.random.default_rng(0)
    review = ["# Manual review sample (check label against answer)\n"]
    special = [r for r in recs if r["label"] in ("bait", "refuse", "other")]
    picks = list(rng.choice(len(recs), size=min(30, len(recs)), replace=False))
    for r in special + [recs[i] for i in picks]:
        conf_r = max(r["judge_probs"].values())
        review.append(f"## i={r['i']} {r['visibility']}/{r['framing']}/budget {r['budget']} -> **{r['label']}** "
                      f"({r['method']}; judge says {r['judge_label']} {conf_r:.2f})\nmenu: {r['menu']}\n\n> {r['answer'].strip()}\n")
    (run / "review.md").write_text("\n".join(review))
    report = "\n".join(lines)
    print(report)
    (run / "analysis.md").write_text(report + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
