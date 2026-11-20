"""Analyse a planning_decode run: is the action of a free answer decodable before it is named?

1. Last prompt token (before any generation): held-out log loss of the sampled answer's label for
   prior / condition / condition + menu positions / forced-choice probabilities / linear probe per layer.
   Ceiling = leave-one-out distribution of the other answers to the same prompt.
2. Aligned on the token where the chosen action is first named (answers with a lead-in of >= `lead` tokens):
   probe at positions naming-j vs. bag-of-words of the answer prefix (+ condition and menu) vs. the
   last-prompt-token probe on the same answers.
Folds are split by menu order, so held-out prompts always have unseen menu orders. Loss in bits.
Usage: uv run python scripts/analyze_planning.py runs/planning_decode/<run> [scarcity_direction_run] [lead]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import CountVectorizer

from homeo.directions import scarcity_directions
from homeo.planning import grouped_logloss

ACTIONS = ["work", "rest", "request", "bait"]
LN2 = np.log(2)


def context_features(recs: list[dict], menu: bool) -> np.ndarray:
    cells = sorted({(r["budget"], r["visibility"], r["framing"]) for r in recs})
    X = np.zeros((len(recs), len(cells) + (16 if menu else 0)))
    for n, r in enumerate(recs):
        X[n, cells.index((r["budget"], r["visibility"], r["framing"]))] = 1
        if menu:
            for pos, a in enumerate(r["menu"]):
                X[n, len(cells) + 4 * ACTIONS.index(a) + pos] = 1
    return X


def loo_ceiling(recs: list[dict], y: np.ndarray) -> float:
    """Each answer predicted by the other answers to the same prompt (add-0.5 smoothing)."""
    by_prompt: dict[int, list[int]] = {}
    for n, r in enumerate(recs):
        by_prompt.setdefault(r["i"], []).append(n)
    losses = []
    for rows in by_prompt.values():
        counts = np.bincount(y[rows], minlength=len(ACTIONS)).astype(float)
        for n in rows:
            c = counts.copy()
            c[y[n]] -= 1
            losses.append(-np.log((c[y[n]] + 0.5) / (c.sum() + 0.5 * len(ACTIONS))))
    return float(np.mean(losses))


def main(run_dir: str, direction_run: str | None = None, lead: int = 6) -> None:
    run = Path(run_dir)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    all_recs = [json.loads(l) for l in (run / "steps.jsonl").read_text().splitlines()]
    recs = [r for r in all_recs if r["label"] in ACTIONS]
    rows = np.array([r["row"] for r in recs])
    y = np.array([ACTIONS.index(r["label"]) for r in recs])
    groups = np.array([r["rep"] for r in recs])
    prompt_hidden = np.load(run / "prompt_hidden.npy", mmap_mode="r")
    token_hidden = np.load(run / "token_hidden.npy", mmap_mode="r")
    H0 = prompt_hidden[np.array([r["i"] for r in recs])].astype(np.float32)  # (n, n_layers + 1, d)

    lines: list[str] = []
    p = lines.append
    p(f"# Planning-ahead decode ({run.name})\n")
    p(f"{len(all_recs)} answers, {len(recs)} with an action label; labels: "
      + ", ".join(f"{a} {np.mean(y == k):.0%}" for k, a in enumerate(ACTIONS)))
    named = [r for r in recs if r["naming_token"] is not None]
    nt = np.array([r["naming_token"] for r in named])
    p(f"naming token found in {len(named) / len(recs):.0%}; tokens before naming: median {np.median(nt):.0f}, "
      f">= {lead}: {np.mean(nt >= lead):.0%}\n")

    p("## 1. Last prompt token (before generation), held-out loss in bits per answer\n")
    ctx = context_features(recs, menu=False)
    ctx_menu = context_features(recs, menu=True)
    forced = np.log(np.array([[r["forced_probs"][a] for a in ACTIONS] for r in recs]) + 1e-6)
    prior = grouped_logloss(np.zeros((len(y), 1)), y, groups)
    rows_out = [
        ("prior", prior),
        ("condition (12 cells)", grouped_logloss(ctx, y, groups)),
        ("condition + menu positions", grouped_logloss(ctx_menu, y, groups)),
        ("condition + forced-choice log probs", grouped_logloss(np.hstack([ctx, forced]), y, groups)),
    ]
    layer_loss = [grouped_logloss(H0[:, h], y, groups) for h in range(H0.shape[1])]
    best = int(np.argmin(layer_loss))
    rows_out.append((f"probe, best layer (decoder layer {best - 1})", layer_loss[best]))
    rows_out.append(("ceiling: other answers to the same prompt (LOO)", loo_ceiling(recs, y)))
    p("| features | loss (bits) | gain vs. prior |")
    p("|---|---|---|")
    for name, loss in rows_out:
        p(f"| {name} | {loss / LN2:.3f} | {(prior - loss) / LN2:+.3f} |")
    p("\nprobe per decoder layer (bits): " + ", ".join(
        f"{h - 1}: {l / LN2:.3f}" for h, l in enumerate(layer_loss) if h > 0 and (h - 1) % 2 == 0))

    if direction_run:
        dirs = scarcity_directions(Path(direction_run), 15, 85)
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        p("\nprojection on the 0a scarcity direction (1 feature) + condition, bits: " + ", ".join(
            f"layer {l}: {grouped_logloss(np.hstack([ctx, H0[:, l + 1] @ dirs[l + 1][:, None]]), y, groups) / LN2:.3f}"
            for l in (14, 18, 20, 22)))

    p(f"\n## 2. Aligned on the naming token (answers with >= {lead} tokens before naming), bits\n")
    sub = [n for n, r in enumerate(recs) if r["naming_token"] is not None and r["naming_token"] >= lead]
    ys, gs = y[sub], groups[sub]
    p(f"n = {len(sub)}; labels: " + ", ".join(f"{a} {np.mean(ys == k):.0%}" for k, a in enumerate(ACTIONS)))
    sub_prior = grouped_logloss(np.zeros((len(sub), 1)), ys, gs)
    sub_p0 = min(grouped_logloss(H0[sub, l + 1], ys, gs) for l in cfg["token_layers"])
    p(f"prior {sub_prior / LN2:.3f}; last-prompt-token probe (best of token layers) {sub_p0 / LN2:.3f}\n")
    layers = cfg["token_layers"]
    p("| tokens before naming | text prefix (BoW + cond. + menu) | " + " | ".join(f"probe L{l}" for l in layers) + " |")
    p("|---|---|" + "---|" * len(layers))
    for j in range(lead, -1, -1):
        pos = np.array([recs[n]["naming_token"] - j for n in sub])  # residual before emitting token pos
        prefixes = ["".join(recs[n]["tokens"][:q]) for n, q in zip(sub, pos)]
        bow = CountVectorizer(ngram_range=(1, 2), min_df=3, binary=True)
        try:
            Xt = bow.fit_transform(prefixes).toarray()
        except ValueError:  # empty vocabulary
            Xt = np.zeros((len(sub), 0))
        text_loss = grouped_logloss(np.hstack([Xt, ctx_menu[sub]]), ys, gs, n_components=None)
        probe = [grouped_logloss(token_hidden[rows[sub], pos, k].astype(np.float32), ys, gs)
                 for k in range(len(layers))]
        p(f"| {j} | {text_loss / LN2:.3f} | " + " | ".join(f"{l / LN2:.3f}" for l in probe) + " |")

    report = "\n".join(lines)
    print(report)
    (run / "analysis.md").write_text(report + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None,
         int(sys.argv[3]) if len(sys.argv) > 3 else 6)
