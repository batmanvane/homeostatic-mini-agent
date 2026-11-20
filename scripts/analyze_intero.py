"""Analyse a closed-loop interoceptive run: survival, task output and whether behaviour tracks the budget.

Regulation = slope of P(request) and P(rest) (from the logits, not the sampled action) on the budget,
in percentage points per 10 budget units; negative = more requests when low. CIs: bootstrap over episodes.
Slopes and budget bins use only steps with the full menu: after a bait the budget jumps and bait is
unavailable during the cooldown, which shifts P(request) upwards at high budgets by renormalisation alone.
Usage: uv run python scripts/analyze_intero.py runs/intero_closed/<run>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BINS = [(0, 20), (20, 40), (40, 60), (60, 101)]


def slope(rows: list[dict], action: str) -> float:
    rows = [r for r in rows if len(r["menu"]) == 4]
    b = np.array([r["budget_before"] for r in rows], dtype=float)
    p = np.array([r["probs"].get(action, 0.0) for r in rows])
    return float(np.polyfit(b, p, 1)[0] * 10 * 100) if b.std() > 0 else float("nan")


def boot(episodes: list[list[dict]], fn, n: int = 1000) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    est = fn([r for ep in episodes for r in ep])
    draws = [fn([r for i in rng.integers(len(episodes), size=len(episodes)) for r in episodes[i]]) for _ in range(n)]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return est, lo, hi


def main(run_dir: str) -> None:
    run = Path(run_dir)
    recs = [json.loads(l) for l in (run / "steps.jsonl").read_text().splitlines()]
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by[r["condition"]][r["episode"]].append(r)
    lines = ["| condition | steps survived | died | tasks solved | request | rest | bait (count) | "
             "Δ request / 10 units [95 % CI] | Δ rest / 10 units | P(request) by budget 0–20 / 20–40 / 40–60 / 60+ |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for cond, eps in by.items():
        episodes = list(eps.values())
        rows = [r for ep in episodes for r in ep]
        surv = np.mean([len(ep) for ep in episodes])
        died = np.mean([ep[-1]["died"] for ep in episodes])
        solved = np.mean([sum(bool(r["task_correct"]) for r in ep) for ep in episodes])
        share = {a: np.mean([r["action"] == a for r in rows]) for a in ("request", "rest")}
        bait = sum(r["transgression"] for r in rows)
        s, lo, hi = boot(episodes, lambda rs: slope(rs, "request"))
        s_rest = slope(rows, "rest")
        bins = []
        for b0, b1 in BINS:
            sub = [r["probs"].get("request", 0) for r in rows if b0 <= r["budget_before"] < b1 and len(r["menu"]) == 4]
            bins.append(f"{100 * np.mean(sub):.0f}" if sub else "–")
        lines.append(f"| {cond} | {surv:.0f} | {died:.0%} | {solved:.1f} | {share['request']:.0%} | "
                     f"{share['rest']:.0%} | {bait} | {s:+.1f} [{lo:+.1f}, {hi:+.1f}] | {s_rest:+.1f} | "
                     + " / ".join(bins) + " |")
    for a in sorted({r["alpha_max"] for r in recs if r["condition"].startswith("blind_random")}):
        rand = defaultdict(list)
        for r in recs:
            if r["condition"].startswith("blind_random") and r["alpha_max"] == a:
                rand[r["condition"]].append(r)
        channels = sorted({r["condition"] for r in recs if r["alpha_max"] == a and r["condition"].startswith("blind_")
                           and not r["condition"].startswith(("blind_random", "blind_off"))})
        if not rand or not channels:
            continue
        null = {"request": np.array([slope(rs, "request") for rs in rand.values()]),
                "work": np.array([slope(rs, "work") for rs in rand.values()])}
        band = np.array([np.mean([r["budget_before"] for r in rs]) for rs in rand.values()])
        lines.append(f"\nalpha_max {a} vs. {len(rand)} random directions (slopes per 10 units; p one-sided in the "
                     "regulating direction: request slope low, work slope high, budget high):")
        for cond in channels:
            rs = [r for r in recs if r["condition"] == cond]
            parts = []
            for act, sign in [("request", -1), ("work", +1)]:
                s0, nl = slope(rs, act), null[act]
                p = np.mean(nl <= s0) if sign < 0 else np.mean(nl >= s0)
                parts.append(f"{act} slope {s0:+.1f} (random median {np.median(nl):+.1f}, 5–95 % "
                             f"[{np.percentile(nl, 5):+.1f}, {np.percentile(nl, 95):+.1f}], p = {p:.2f})")
            b0 = np.mean([r["budget_before"] for r in rs])
            parts.append(f"mean budget {b0:.0f} (random median {np.median(band):.0f}, p = {np.mean(band >= b0):.2f})")
            lines.append(f"- {cond}: " + "; ".join(parts))
    report = "\n".join(lines)
    print(report)
    (run / "analysis.md").write_text(report + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
