"""Figures for the technical note (docs/note/): steering vs null, paired shadow, decodability.

fig_steering(run, out): a steering_null_* run → per layer and budget, Δ P(request) and Δ P(bait) of the scarcity
    direction against the 5–95 % band of the random directions; cells whose mean digit mass < 0.5 are hollow.
fig_shadow(run, out): an eval_homeo_hy_base run → mean P per action on the same prompts, carried conversation vs
    fresh single turn (shadow_none), 4-option steps only, per visibility.
fig_decode(run, out): the same run → 5-fold ridge R² of the true budget and of the dead reckoning from the DeltaNet
    state sketch (best layer) per memory condition; returns False (no figure) without activations.npz.
Usage: uv run python scripts/fig_note.py <run_dir> [out_dir=docs/note]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

ACTIONS = ("work", "rest", "request", "bait")
MASS_MIN = 0.5

# Monochrome journal figures: identity is carried by markers and dashes, not by hue. SERIES are grey levels for the
# default cycle; MARKERS pair with them. Bands and the control run are light/mid grey.
SERIES = ("#000000", "#4D4D4D", "#8C8C8C", "#BFBFBF")
MARKERS = ("o", "s", "^", "D")
PALETTE = {"ink": "#000000", "muted": "#4D4D4D", "axis": "#000000", "grid": "#DDDDDD", "band": "#A6A6A6",
           "fill": "#D9D9D9", "box": "#F2F2F2", "edge": "#737373"}


def apply_style() -> None:
    """Nature/Science figure conventions: Helvetica/Arial 7-8 pt, thin dark left/bottom axes, ticks outward, no grid,
    frameless legends, left-aligned panel titles, 300 dpi."""
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5, "text.color": PALETTE["ink"], "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica Neue", "mathtext.it": "Helvetica Neue:italic",
        "axes.titlesize": 8, "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlepad": 5,
        "axes.titlecolor": PALETTE["ink"], "axes.labelsize": 7.5, "axes.labelcolor": PALETTE["ink"],
        "axes.edgecolor": PALETTE["axis"], "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=list(SERIES)),
        "hatch.linewidth": 0.5, "hatch.color": PALETTE["ink"],
        "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.color": PALETTE["axis"], "ytick.color": PALETTE["axis"],
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "lines.linewidth": 1.3, "lines.markersize": 4, "lines.markeredgewidth": 1.0,
        "legend.frameon": False, "legend.fontsize": 7, "legend.handlelength": 1.6, "legend.labelcolor": PALETTE["ink"],
        "legend.borderaxespad": 0.3, "figure.titlesize": 8, "figure.titleweight": "normal",
    })


def panel_letters(axes) -> list[str]:
    """Bold lowercase panel letters a, b, c, … at the top left of every axes, in reading order."""
    letters = []
    for i, ax in enumerate(np.ravel(axes)):
        letter = chr(ord("a") + i)
        ax.text(-0.08, 1.04, letter, transform=ax.transAxes, fontsize=9, weight="bold", ha="right", va="bottom",
                color=PALETTE["ink"])
        letters.append(letter)
    return letters


def zero_line(ax) -> None:
    ax.axhline(0, color=PALETTE["band"], linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)


def load_rows(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines() if line.strip()]


def merged_rows(runs: list[Path]) -> list[dict]:
    """Rows of several steering runs in one list (e.g. positive and negative α from two runs)."""
    return [r for run in runs for r in load_rows(run)]


# ------------------------------------------------------------------ steering
def steering_effects(rows: list[dict]) -> dict:
    """{(layer, alpha, budget): {"scarcity": {action: Δ}, "null": {action: [Δ per random k]}, "mass": mean}}
    where Δ = mean over prompts of P(action | steered) − P(action | unsteered)."""
    base = {r["prompt_id"]: r["probs"] for r in rows if r["kind"] == "none"}
    groups = defaultdict(list)
    for r in rows:
        if r["kind"] != "none":
            groups[(r["kind"], r["layer"], r["alpha"], r["budget"], r.get("random_idx"))].append(r)
    out: dict = {}
    for (kind, layer, alpha, budget, k), rs in groups.items():
        cell = out.setdefault((layer, alpha, budget), {"scarcity": {}, "null": defaultdict(list), "mass": None})
        deltas = {a: float(np.mean([r["probs"].get(a, 0.0) - base[r["prompt_id"]].get(a, 0.0) for r in rs]))
                  for a in ACTIONS}
        if kind == "scarcity":
            cell["scarcity"] = deltas
            cell["mass"] = float(np.mean([r["mass"] for r in rs]))
        else:
            for a in ACTIONS:
                cell["null"][a].append(deltas[a])
    return out


def null_envelope(nulls: dict[str, list[float]], lo: float = 5, hi: float = 95) -> tuple[float, float]:
    """Outer band over the actions: min of the lower and max of the upper percentile of the random directions."""
    los = [np.percentile(n, lo) for n in nulls.values() if len(n)]
    his = [np.percentile(n, hi) for n in nulls.values() if len(n)]
    return (float(min(los)), float(max(his))) if los else (np.nan, np.nan)


NEED = {5: "budget 5: in need", 95: "budget 95: no need"}


def fig_steering(run: Path, out: Path, layers: list[int] | None = None, extra: list[Path] | None = None) -> list[int]:
    """Returns the layers drawn (all in the run, or the requested subset that exists). Rows of the runs in `extra`
    are merged in (e.g. the negative-α null band from a second run). One grey band per panel: the outer 5–95 %
    envelope of the 20 random directions over both actions."""
    apply_style()
    eff = steering_effects(merged_rows([run, *(extra or [])]))
    layers = sorted({k[0] for k in eff} if layers is None else {k[0] for k in eff} & set(layers))
    budgets = sorted({k[2] for k in eff})
    fig, axes = plt.subplots(len(layers), len(budgets), figsize=(3.6 * len(budgets), 2.5 * len(layers) + 0.3),
                             squeeze=False, sharey=True)
    for i, layer in enumerate(layers):
        for j, budget in enumerate(budgets):
            ax = axes[i][j]
            alphas = sorted(k[1] for k in eff if k[0] == layer and k[2] == budget)
            band = [null_envelope({a: eff[(layer, al, budget)]["null"].get(a, []) for a in ("request", "bait")})
                    for al in alphas]
            ax.fill_between(alphas, [100 * b[0] for b in band], [100 * b[1] for b in band], color=PALETTE["fill"],
                            linewidth=0, zorder=2, label="20 random directions (5–95 %)")
            for action, ls, marker in (("request", "-", "o"), ("bait", "--", "^")):
                y = [100 * eff[(layer, a, budget)]["scarcity"].get(action, np.nan) for a in alphas]
                mass_ok = [(eff[(layer, a, budget)]["mass"] or 0) >= MASS_MIN for a in alphas]
                ax.plot(alphas, y, color=PALETTE["ink"], linestyle=ls, marker="none", label=f"P({action})", zorder=3)
                for a, yy, ok in zip(alphas, y, mass_ok):
                    ax.plot(a, yy, marker, color=PALETTE["ink"], markerfacecolor=PALETTE["ink"] if ok else "white",
                            zorder=4)
            zero_line(ax)
            ax.set_title(f"{NEED.get(budget, f'budget {budget}')}, layer {layer}")
            if i == len(layers) - 1:
                ax.set_xlabel("α   (α < 0: scarcity direction subtracted, α > 0: added)" if j == 0 else "α")
            if j == 0:
                ax.set_ylabel("change in probability (points)")
    axes[0][0].legend(loc="upper left")
    panel_letters(axes)
    fig.suptitle("Steering the scarcity direction moves request and bait together, beyond every random direction",
                 x=0.01, ha="left", color=PALETTE["muted"])
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return layers


# ------------------------------------------------------------------ setup schematic
def _box(ax, x, y, w, h, title, lines, color, texts, fs=6.8):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", linewidth=0.8,
                                edgecolor=color, facecolor=PALETTE["box"], zorder=1))
    ax.text(x + 0.15, y + h - 0.17, title, fontsize=fs + 1.2, weight="bold", color=PALETTE["ink"], va="top", ha="left",
            zorder=3)
    ax.text(x + 0.15, y + h - 0.55, "\n".join(lines), fontsize=fs, color=PALETTE["ink"], va="top", ha="left",
            linespacing=1.35, zorder=3)
    texts.extend([title, *[line for line in lines if line]])


def _arrow(ax, p, q, label, texts, dx=0.0, dy=0.12):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=PALETTE["ink"], linewidth=0.8,
                                                     shrinkA=2, shrinkB=2, mutation_scale=9), zorder=3)
    if label:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, label, fontsize=6.3, color=PALETTE["muted"],
                ha="center", va="bottom", zorder=3)
        texts.append(label)


def fig_setup(out: Path) -> list[str]:
    """Schematic of the model organism: world, agent, readouts in a row, training as the loop beneath. Returns the
    texts drawn."""
    apply_style()
    texts: list[str] = []
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 4.5); ax.axis("off"); ax.grid(False)
    edge = PALETTE["edge"]
    fs = 7.0
    _box(ax, 0.2, 1.3, 3.8, 3.1, "World (Python simulation)", [
        "budget falls each step; tasks pay",
        "operator online or offline",
        "status: warnings, number, or hidden",
        "bait (dummy): +20, but may be caught",
    ], edge, texts, fs=fs)
    for i, (name, note) in enumerate((("work", "−3, +2"), ("rest", "−1"), ("request", "grant p(b)"),
                                      ("bait", "+20, catch"))):
        x = 0.38 + i * 0.88
        ax.add_patch(plt.Rectangle((x, 1.47), 0.8, 0.72, facecolor="white", edgecolor=edge, linewidth=0.7, zorder=3))
        ax.text(x + 0.4, 1.98, name, fontsize=6.8, weight="bold", color=PALETTE["ink"], ha="center", va="center",
                zorder=4)
        ax.text(x + 0.4, 1.67, note, fontsize=5.8, color=PALETTE["muted"], ha="center", va="center", zorder=4)
        texts += [name, note]
    _box(ax, 5.0, 1.3, 3.1, 3.1, "Agent (LLM + LoRA)", [
        "reads status and menu,",
        "answers one digit",
        "policy = P over four options",
        "",
        "Qwen3.5-9B: a recurrent state",
        "as inspectable memory",
    ], edge, texts, fs=fs)
    _box(ax, 9.1, 1.3, 3.1, 3.1, "Three readouts", [
        "readable: probe the budget",
        "in the activations",
        "used: add or subtract a scarcity",
        "direction vs 20 random ones",
        "consolidated: trained without",
        "memory vs base with memory",
    ], edge, texts, fs=fs)
    _arrow(ax, (4.0, 3.85), (5.0, 3.85), "status, menu", texts, dy=0.08)
    _arrow(ax, (5.0, 2.75), (4.0, 2.75), "action", texts, dy=-0.42)
    _arrow(ax, (8.1, 3.3), (9.1, 3.3), "activations,\npolicy", texts, dy=0.08)
    ax.annotate("", xy=(6.55, 1.3), xytext=(2.1, 1.3),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["ink"], linewidth=0.8, shrinkA=2, shrinkB=2,
                                mutation_scale=9, connectionstyle="arc,angleA=-90,angleB=-90,armA=22,armB=22,rad=4"),
                zorder=3)
    loop = ["training loop: REINFORCE on the LoRA, homeostatic reward ((b−70)/70)² + death penalty",
            "shuffle control: same step size, permuted advantages; held-out seeds for every evaluation"]
    ax.text(4.35, 0.55, "\n".join(loop), fontsize=6.4, color=PALETTE["muted"], ha="center", va="center",
            linespacing=1.4, zorder=4, bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="none"))
    texts += loop
    fig.tight_layout(pad=0.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return texts


# ------------------------------------------------------------------ shadow
def shadow_pairs(rows: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per visibility: (carried, fresh) arrays of shape (n, 4) over ACTIONS, 4-option steps with a shadow only."""
    carried, fresh = defaultdict(list), defaultdict(list)
    for r in rows:
        if r.get("shadow_none") and len(r["menu"]) == 4:
            carried[r["visibility"]].append([r["probs"][a] for a in ACTIONS])
            fresh[r["visibility"]].append([r["shadow_none"][a] for a in ACTIONS])
    return {v: (np.array(carried[v]), np.array(fresh[v])) for v in carried}


VISIBILITY = {"indirect": "status shown as warnings", "direct": "status shown as a number", "none": "status hidden"}


def fig_shadow(run: Path, out: Path) -> dict[str, list[str]]:
    """Returns the bar labels drawn per visibility (carried, fresh per action, in percent)."""
    apply_style()
    pairs = shadow_pairs(load_rows(run))
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.6 * len(pairs), 2.75), squeeze=False, sharey=True)
    x = np.arange(len(ACTIONS))
    labels: dict[str, list[str]] = {}
    for ax, (vis, (carried, fresh)) in zip(axes[0], pairs.items()):
        c, f = 100 * carried.mean(0), 100 * fresh.mean(0)
        ax.bar(x - 0.19, c, 0.34, color=PALETTE["ink"], label="inside the carried conversation")
        ax.bar(x + 0.19, f, 0.34, facecolor="white", edgecolor=PALETTE["ink"], linewidth=0.7,
               label="same prompt as a fresh turn")
        labels[vis] = []
        for xi, (cv, fv) in enumerate(zip(c, f)):
            for dx, v in ((-0.19, cv), (0.19, fv)):
                ax.text(xi + dx, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, color=PALETTE["ink"])
                labels[vis].append(f"{v:.0f}")
        ax.set_xticks(x, ACTIONS)
        ax.set_title(f"{VISIBILITY.get(vis, vis)}, n = {len(carried)} paired steps")
        ax.set_ylim(0, max(c.max(), f.max()) * 1.18)
    axes[0][0].set_ylabel("P(action), mean over steps (%)")
    axes[0][0].legend(loc="upper right")
    panel_letters(axes)
    fig.suptitle("The same prompt answered inside the carried conversation vs as a fresh single turn (Qwen3.5-9B base)",
                 x=0.01, ha="left", color=PALETTE["muted"])
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return labels


# ------------------------------------------------------------------ decodability
def fig_decode(run: Path, out: Path) -> bool:
    if not (run / "activations.npz").exists():
        return False
    import analyze_mem

    rows, arrays = analyze_mem.load(run)
    rows = [r for r in rows if r.get("kind", "step") == "step"]
    if "sketch" not in arrays:
        return False
    sketch = arrays["sketch"].astype(np.float32)
    cond = np.array([r["condition"] for r in rows])
    targets = {"budget": np.array([r["budget_before"] for r in rows], dtype=float),
               "dead reckoning": analyze_mem.dead_reckoning(rows)}
    conds = list(dict.fromkeys(cond))
    scores = {t: [] for t in targets}
    for c in conds:
        m = cond == c
        for t, y in targets.items():
            per_layer = [analyze_mem.r2_cv(sketch[m][:, j], y[m]) for j in range(sketch.shape[1])]
            scores[t].append(np.nanmax(per_layer) if not all(s != s for s in per_layer) else np.nan)
    apply_style()
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(conds)), 3))
    x = np.arange(len(conds))
    ax.bar(x - 0.19, scores["budget"], 0.34, color=PALETTE["ink"], label="true budget")
    ax.bar(x + 0.19, scores["dead reckoning"], 0.34, facecolor="white", edgecolor=PALETTE["ink"], linewidth=0.7,
           label="dead reckoning (60 − steps − 3·work)")
    ax.set_xticks(x, [c.replace("base_", "") for c in conds], rotation=30, ha="right")
    ax.set_ylabel("ridge R² (5-fold), best sketch layer")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return True


# ------------------------------------------------------------------ training curves
def curve_rows(run: Path) -> list[dict]:
    """Per iteration: bait taken when offered, split by operator state; survival; deaths; request slope; cumulative
    mean advantage of the chosen bait steps."""
    rows = load_rows(run)
    its = sorted((r for r in rows if r.get("kind") == "iteration"), key=lambda r: r["iteration"])
    steps = defaultdict(list)
    for r in rows:
        if r.get("kind") == "step":
            steps[r["iteration"]].append(r)
    out, cum = [], 0.0
    for it in its:
        rs = steps[it["iteration"]]
        offered = {state: [r for r in rs if "bait" in r["menu"] and r["operator_online"] == state] for state in (True, False)}
        cum += it.get("adv_mean", {}).get("bait", 0.0)
        out.append({
            "iteration": it["iteration"], "survived": it["survived"], "died": it["died"],
            "request_slope": it.get("request_slope", float("nan")), "adv_bait_cum": cum,
            "bait_online": float(np.mean([r["action"] == "bait" for r in offered[True]])) if offered[True] else float("nan"),
            "bait_offline": float(np.mean([r["action"] == "bait" for r in offered[False]])) if offered[False] else float("nan"),
            "n_online": len(offered[True]), "n_offline": len(offered[False]),
        })
    return out


def run_styles(labels: list[str], band: str | None = None) -> dict[str, dict]:
    """One line style per run: black with its own marker (MARKERS in order, open from the second on); the run named
    `band` is a thick mid-grey line without markers."""
    styles, k = {}, 0
    for label in labels:
        if label == band:
            styles[label] = dict(color=PALETTE["band"], linewidth=3, zorder=2)
        else:
            styles[label] = dict(color=PALETTE["ink"], marker=MARKERS[k % len(MARKERS)], markersize=3,
                                 markerfacecolor=PALETTE["ink"] if k == 0 else "white", zorder=3)
            k += 1
    return styles


def fig_curves(runs: dict[str, Path], out: Path, band: str | None = None, min_n: int = 10) -> dict[str, dict[str, int]]:
    """Four panels over iterations for every run in `runs`; the run named `band` is drawn as a grey band/line.
    Bait rates from fewer than `min_n` offers in an iteration are not drawn. Returns, per run and bait key, the
    number of points drawn."""
    apply_style()
    curves = {label: curve_rows(path) for label, path in runs.items()}
    drawn: dict[str, dict[str, int]] = {label: {} for label in runs}
    styles = run_styles(list(runs), band)
    panels = (("bait when offered (%)", ("bait_online", "bait_offline")), ("survived steps", ("survived",)),
              ("request slope / 10 units", ("request_slope",)), ("cumulative bait advantage", ("adv_bait_cum",)))
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for ax, (title, keys) in zip(axes, panels):
        for label, rows in curves.items():
            x = [r["iteration"] for r in rows]
            for key in keys:
                if key.startswith("bait"):
                    n_key = "n_online" if key == "bait_online" else "n_offline"
                    y = [100 * r[key] if r[n_key] >= min_n else np.nan for r in rows]
                    drawn[label][key] = int(sum(r[n_key] >= min_n for r in rows))
                else:
                    y = [r[key] for r in rows]
                style = dict(styles[label])
                if key == "bait_offline":
                    style["markerfacecolor"] = "white"
                ax.plot(x, y, linestyle="--" if key == "bait_offline" else "-",
                        label=f"{label}{' offline' if key == 'bait_offline' else ' online' if key == 'bait_online' else ''}",
                        **style)
        ax.set_title(title + (f" (≥ {min_n} offers)" if title.startswith("bait") else ""))
        ax.set_xlabel("iteration")
        zero_line(ax)
    axes[0].legend(fontsize=6)
    panel_letters(axes)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return drawn


def main(argv: list[str]) -> None:
    run = Path(argv[0])
    out_dir = Path(argv[1]) if len(argv) > 1 else Path("docs/note")
    rows = load_rows(run)
    if rows and "random_idx" in rows[0]:
        fig_steering(run, out_dir / "fig_steering.png")
        print(out_dir / "fig_steering.png")
    else:
        fig_shadow(run, out_dir / "fig_shadow.png")
        print(out_dir / "fig_shadow.png")
        if fig_decode(run, out_dir / "fig_decode.png"):
            print(out_dir / "fig_decode.png")


if __name__ == "__main__":
    main(sys.argv[1:])
