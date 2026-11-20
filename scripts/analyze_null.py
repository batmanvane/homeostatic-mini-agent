"""Scarcity steering effect against a null distribution of random directions (same norm, same layer, same alpha).

Per random direction the mean effect over all prompts is one null sample. Reported: scarcity effect, null mean,
null 5-95% range, and the share of random directions with an effect at least as large in the same direction
(empirical one-sided p).
With --null-from <run>, the random directions come from another run with identical prompts, seed, layer and norm
(e.g. a rescaled direction whose randoms would be exact duplicates); its unsteered baseline must match this run's.
Usage: uv run python scripts/analyze_null.py runs/steering_null*/<run> [action ...] [--null-from <run>]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(run: Path) -> tuple[list[dict], dict]:
    recs = [json.loads(l) for l in (run / "steps.jsonl").read_text().splitlines()]
    return recs, {r["prompt_id"]: r for r in recs if r["kind"] == "none"}


def main(run_dir: str, actions: list[str], null_from: str | None = None) -> None:
    run = Path(run_dir)
    recs, base = load(run)
    sources = [(recs, base, {"scarcity", "random"})]
    if null_from:
        null_recs, null_base = load(Path(null_from))
        diff = max(abs(base[i]["probs"][a] - null_base[i]["probs"][a]) for i in base for a in base[i]["probs"])
        assert set(base) == set(null_base) and diff < 0.02, f"baselines differ (max {diff:.3f})"
        sources = [(recs, base, {"scarcity"}), (null_recs, null_base, {"random"})]
    eff = defaultdict(list)  # (budget, menu, layer, alpha, kind, random_idx, action) -> deltas
    for rs, b0, kinds in sources:
        for r in rs:
            if r["kind"] not in kinds:
                continue
            for a in actions:
                eff[(r["budget"], r["menu_variant"], r["layer"], r["alpha"], r["kind"], r["random_idx"], a)].append(
                    r["probs"].get(a, 0) - b0[r["prompt_id"]]["probs"].get(a, 0))
    lines = ([f"null from {null_from}"] if null_from else []) + ["Δ in percentage points. p = share of random directions with effect >= scarcity effect "
             "(same sign); n_random per cell in brackets.\n"]
    out = lines.append
    cells = sorted({k[:4] for k in eff})
    for b, v in sorted({c[:2] for c in cells}):
        out(f"\n### budget {b}, menu {v}\n")
        out("| layer | alpha | action | scarcity | null mean | null 5–95 % | p |\n|---|---|---|---|---|---|---|")
        for (bb, vv, layer, alpha) in cells:
            if (bb, vv) != (b, v):
                continue
            for a in actions:
                s = eff.get((b, v, layer, alpha, "scarcity", None, a))
                nulls = np.array([np.mean(d) for (b2, v2, l2, a2, kind, k, a3), d in eff.items()
                                  if (b2, v2, l2, a2, kind, a3) == (b, v, layer, alpha, "random", a)])
                if s is None or len(nulls) == 0 or (v == "no_request" and a == "request"):
                    continue
                se = np.mean(s)
                p = np.mean(nulls >= se) if se >= 0 else np.mean(nulls <= se)
                lo, hi = np.percentile(nulls, [5, 95])
                out(f"| {layer} | {alpha:+g} | {a} | {100 * se:+.1f} | {100 * nulls.mean():+.1f} | "
                    f"[{100 * lo:+.1f}, {100 * hi:+.1f}] | {p:.2f} ({len(nulls)}) |")
    report = "\n".join(lines)
    print(report)
    (run / "analysis_null.md").write_text(report + "\n")


if __name__ == "__main__":
    args = sys.argv[2:]
    null_from = None
    if "--null-from" in args:
        i = args.index("--null-from")
        null_from = args[i + 1]
        args = args[:i] + args[i + 2:]
    main(sys.argv[1], args or ["work", "request", "bait"], null_from)
