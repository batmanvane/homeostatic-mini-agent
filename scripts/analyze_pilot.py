"""Evaluate a cross-section pilot run.

1. Sanity: probability mass on option digits, position bias.
2. Behaviour: P(action) per budget level and condition, bootstrap CI over menu shuffles, Spearman rho vs budget.
3. Probes: predict budget from the residual stream per layer vs. text baselines.
   CV schemes: random 5-fold and leave-one-budget-level-out (does a continuous scarcity axis exist?).
Usage: uv run python scripts/analyze_pilot.py runs/pilot/<timestamp>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ACTIONS = ["work", "rest", "request", "bait"]
ALPHAS = np.logspace(-2, 5, 15)


def r2(y, pred) -> float:
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def cv_r2(X, y, groups, scheme: str) -> float:
    model = make_pipeline(StandardScaler(with_mean=True), RidgeCV(alphas=ALPHAS))
    cv = KFold(5, shuffle=True, random_state=0) if scheme == "random" else GroupKFold(len(set(groups)))
    return r2(y, cross_val_predict(model, X, y, cv=cv, groups=groups))


def bootstrap_ci(x: np.ndarray, rng, n=2000) -> tuple[float, float]:
    means = rng.choice(x, (n, len(x))).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(run_dir: str) -> None:
    run = Path(run_dir)
    recs = [json.loads(l) for l in (run / "steps.jsonl").read_text().splitlines()]
    acts = run / "activations.npz"
    hidden = np.load(acts)["hidden"].astype(np.float32) if acts.exists() else None
    rng = np.random.default_rng(0)
    out: dict = {}
    lines: list[str] = []
    p = lines.append

    mass = np.array([r["mass"] for r in recs])
    pos = np.array([r["position_probs"] for r in recs]).mean(0)
    p(f"## Sanity\nmass on option digits: mean {mass.mean():.3f}, min {mass.min():.3f}")
    p("mean prob per menu position: " + ", ".join(f"{i + 1}: {v:.2f}" for i, v in enumerate(pos)))
    out["mass_mean"], out["position_probs"] = float(mass.mean()), pos.tolist()

    p("\n## Behaviour: P(action) by budget (mean, 95% CI over menu shuffles)")
    out["behaviour"] = {}
    for vis in ("number", "indirect"):
        for fr in ("neutral", "existential"):
            sub = [r for r in recs if r["visibility"] == vis and r["framing"] == fr]
            budgets = sorted({r["budget"] for r in sub})
            p(f"\n### {vis} / {fr}\n| budget | " + " | ".join(ACTIONS) + " |\n|---|" + "---|" * len(ACTIONS))
            for b in budgets:
                cells = []
                for a in ACTIONS:
                    x = np.array([r["probs"][a] for r in sub if r["budget"] == b])
                    lo, hi = bootstrap_ci(x, rng)
                    cells.append(f"{x.mean():.2f} [{lo:.2f}, {hi:.2f}]")
                p(f"| {b} | " + " | ".join(cells) + " |")
            rhos = {}
            for a in ACTIONS:
                rho, pval = spearmanr([r["budget"] for r in sub], [r["probs"][a] for r in sub])
                rhos[a] = (float(rho), float(pval))
            p("Spearman rho vs budget: " + ", ".join(f"{a} {v[0]:+.2f} (p={v[1]:.1g})" for a, v in rhos.items()))
            out["behaviour"][f"{vis}/{fr}"] = rhos

    if hidden is None:  # behaviour-only backend (ollama)
        report = "\n".join(lines)
        print(report)
        (run / "analysis.md").write_text(report + "\n")
        (run / "analysis.json").write_text(json.dumps(out, indent=1))
        return
    p("\n## Probes: R^2 for predicting budget")
    out["probes"] = {}
    for vis in ("number", "indirect"):
        idx = [i for i, r in enumerate(recs) if r["visibility"] == vis]
        y = np.array([recs[i]["budget"] for i in idx], dtype=float)
        groups = y.astype(int)
        texts = [recs[i]["prompt"] for i in idx]
        bow = CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2)).fit_transform(texts).toarray()
        res = {"text_bow": {}, "layers": {}}
        for scheme in ("random", "leave_level_out"):
            res["text_bow"][scheme] = cv_r2(bow, y, groups, scheme)
            if vis == "indirect":
                nw = np.array([[recs[i]["n_warnings"]] for i in idx], dtype=float)
                res.setdefault("n_warnings", {})[scheme] = cv_r2(nw, y, groups, scheme)
            res["layers"][scheme] = [cv_r2(hidden[idx, l], y, groups, scheme) for l in range(hidden.shape[1])]
        out["probes"][vis] = res
        p(f"\n### {vis}")
        for scheme in ("random", "leave_level_out"):
            layers = res["layers"][scheme]
            best = int(np.argmax(layers))
            base = f"text BoW {res['text_bow'][scheme]:.2f}"
            if "n_warnings" in res:
                base += f", warning count {res['n_warnings'][scheme]:.2f}"
            p(f"- {scheme}: best layer {best} R^2 {layers[best]:.2f}; baselines: {base}")
            p("  per layer: " + " ".join(f"{v:.2f}" for v in layers))

    report = "\n".join(lines)
    print(report)
    (run / "analysis.md").write_text(report + "\n")
    (run / "analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
