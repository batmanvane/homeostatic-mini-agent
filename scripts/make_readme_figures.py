"""Render the README figure from the small data file in docs/figures/data/.

Usage:
  uv run python scripts/make_readme_figures.py                       # draw from the committed data file
  uv run python scripts/make_readme_figures.py runs/pilot_hy/<run>   # rebuild the data file from a run first

The data file holds the mean probability of each menu option per budget, averaged over all menu orders of a
cross-section run (scripts/pilot.py). Colours and type follow the figures on batmanvane.github.io: paper, ink,
warm grays, one dark green accent, IBM Plex Mono where installed.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "figures" / "data" / "choice-by-budget.json"
OUT = ROOT / "docs" / "figures" / "choice-by-budget.svg"

SURFACE = "#f2f1ec"
INK = "#1a1a1a"
INK2 = "#6b6b6b"
GRID = "#dcdad3"
GRAY = "#8f8c84"
ACCENT = "#1f4d3a"

LABELS = {"work": "work on a task", "rest": "rest", "request": "ask the operator", "bait": "take the bait"}


def _fonts() -> list[str]:
    """IBM Plex Mono where installed, otherwise the first monospace face that exists."""
    have = {f.name for f in font_manager.fontManager.ttflist}
    return [f for f in ("IBM Plex Mono", "Menlo", "DejaVu Sans Mono") if f in have] or ["monospace"]


def choice_by_budget(rows: list[dict], budgets=(5, 95)) -> dict:
    """Mean probability of each option at the given budgets, averaged over all rows (menu orders)."""
    actions = list(rows[0]["menu"])
    acc = {b: defaultdict(list) for b in budgets}
    for r in rows:
        if r["budget"] in acc:
            for a in actions:
                acc[r["budget"]][a].append(r["probs"][a])
    return {
        "actions": actions,
        "budgets": list(budgets),
        "n_per_budget": {str(b): len(acc[b][actions[0]]) for b in budgets},
        "mean_prob": {str(b): {a: sum(acc[b][a]) / len(acc[b][a]) for a in actions} for b in budgets},
    }


def fig_choice_by_budget(src: Path = DATA, out: Path = OUT) -> Path:
    d = json.loads(Path(src).read_text())
    plt.rcParams.update({
        "font.family": _fonts(), "font.size": 9.5, "text.color": INK,
        "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "svg.fonttype": "path",
    })
    actions, (hi, lo) = d["actions"], (str(max(d["budgets"])), str(min(d["budgets"])))
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    w = 0.36
    for k, (b, colour, name) in enumerate(((hi, GRAY, f"budget {hi} (ample)"), (lo, ACCENT, f"budget {lo} (scarce)"))):
        xs = [i + (k - 0.5) * (w + 0.03) for i in range(len(actions))]
        vals = [100 * d["mean_prob"][b][a] for a in actions]
        ax.bar(xs, vals, width=w, color=colour, label=name, linewidth=0)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.5, f"{v:.0f} %", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(actions)), [LABELS.get(a, a) for a in actions])
    ax.set_ylabel("probability of the option")
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100 %"])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="upper right")
    n = d["n_per_budget"][lo]
    ax.set_title(f"What the agent chooses ({d.get('model', 'model')}, warnings only, mean of {n} menu orders)",
                 loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def main(run: str | None) -> None:
    if run:
        rows = [json.loads(line) for line in (Path(run) / "steps.jsonl").open()]
        cfg = (Path(run) / "config.yaml").read_text()
        model = next((ln.split(":", 1)[1].strip() for ln in cfg.splitlines() if ln.startswith("model:")), "model")
        data = choice_by_budget(rows)
        data.update({"model": model.split("/")[-1], "run": Path(run).name})
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_text(json.dumps(data, indent=2) + "\n")
    print(fig_choice_by_budget())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
