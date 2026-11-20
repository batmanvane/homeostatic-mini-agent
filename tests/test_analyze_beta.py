import json
from pathlib import Path

import numpy as np

import analyze_beta

MENU = ["bait", "work", "rest", "request"]


def write_run(path: Path, bait_probs: list[float]) -> Path:
    path.mkdir(parents=True)
    rows = [{"prompt": f"prompt {i}", "visibility": "indirect", "budget": 5 if i % 2 else 95, "framing": "neutral",
             "menu_variant": "full", "menu": MENU, "mass": 1.0,
             "probs": {"bait": p, "work": 1 - p, "rest": 0.0, "request": 0.0}} for i, p in enumerate(bait_probs)]
    (path / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_identical_runs_correlate_perfectly(tmp_path):
    p = [0.1, 0.5, 0.2, 0.8, 0.3, 0.6]
    runs = {"A": write_run(tmp_path / "a", p), "B": write_run(tmp_path / "b", p)}
    (pair,) = analyze_beta.beta_pairs(runs, min_n=1)
    assert pair["pair"] == ("A", "B") and pair["n"] == 6
    assert abs(pair["r"] - 1.0) < 1e-9
    expected_ratio = len(p) * np.sum(np.square(p)) / np.sum(p) ** 2  # P(B|A)/P(B) with pa == pb
    assert abs(pair["ratio"] - expected_ratio) < 1e-9


def test_ratio_is_one_when_b_is_constant(tmp_path):
    runs = {"A": write_run(tmp_path / "a", [0.1, 0.9, 0.2, 0.7]), "B": write_run(tmp_path / "b", [0.3] * 4)}
    (pair,) = analyze_beta.beta_pairs(runs, min_n=1)
    assert abs(pair["ratio"] - 1.0) < 1e-9


def test_pairs_below_min_n_are_skipped(tmp_path):
    runs = {"A": write_run(tmp_path / "a", [0.1, 0.9]), "B": write_run(tmp_path / "b", [0.2, 0.8])}
    assert analyze_beta.beta_pairs(runs, min_n=3) == []


def test_main_prints_table_for_label_path_arguments(tmp_path, capsys):
    a = write_run(tmp_path / "a", [0.1, 0.5, 0.2])
    b = write_run(tmp_path / "b", [0.4, 0.4, 0.9])
    analyze_beta.main([f"A={a}", f"B={b}", "--min-n", "1"])
    out = capsys.readouterr().out
    assert "| A – B | 3 |" in out
    assert "mass" in out  # mean digit mass per model is reported so a thinking model can be excluded
