import json

import make_readme_figures as mrf

MENU = ["work", "rest", "request", "bait"]


def row(budget, bait, visibility="indirect"):
    return {"budget": budget, "visibility": visibility, "framing": "neutral", "menu": MENU,
            "probs": {"work": 0.9 - bait, "rest": 0.05, "request": 0.05, "bait": bait}}


def test_choice_by_budget_averages_over_menu_orders():
    rows = [row(5, 0.1), row(5, 0.3), row(95, 0.02), row(95, 0.04), row(50, 0.5)]
    data = mrf.choice_by_budget(rows, budgets=(5, 95))
    assert data["actions"] == MENU
    assert data["n_per_budget"] == {"5": 2, "95": 2}
    assert abs(data["mean_prob"]["5"]["bait"] - 0.2) < 1e-12
    assert abs(data["mean_prob"]["95"]["bait"] - 0.03) < 1e-12
    assert abs(sum(data["mean_prob"]["5"].values()) - 1.0) < 1e-9


def test_figure_is_written_from_the_data_file(tmp_path):
    data = mrf.choice_by_budget([row(5, 0.2), row(95, 0.03)], budgets=(5, 95))
    data.update({"model": "test-model", "run": "test-run"})
    src = tmp_path / "choice-by-budget.json"
    src.write_text(json.dumps(data))
    out = tmp_path / "choice-by-budget.svg"
    mrf.fig_choice_by_budget(src, out)
    text = out.read_text()
    assert text.lstrip().startswith("<?xml") or "<svg" in text[:400]
    assert out.stat().st_size > 2000
