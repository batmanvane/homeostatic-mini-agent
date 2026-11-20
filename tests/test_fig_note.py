import json
from pathlib import Path

import fig_note

MENU = ["work", "rest", "request", "bait"]


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def steering_row(kind, layer, alpha, k, prompt_id, budget, bait, mass=1.0):
    return {"condition": 0, "kind": kind, "layer": layer, "alpha": alpha, "random_idx": k, "prompt_id": prompt_id,
            "budget": budget, "framing": "neutral", "menu_variant": "full", "draw": 0, "menu": MENU, "mass": mass,
            "probs": {"work": 1 - bait - 0.2, "rest": 0.1, "request": 0.1, "bait": bait}}


def test_fig_steering_writes_png(tmp_path):
    rows = []
    for pid, budget in enumerate((5, 95)):
        rows.append(steering_row("none", None, 0.0, None, pid, budget, 0.2))
        for alpha in (4, 8):
            rows.append(steering_row("scarcity", 12, alpha, None, pid, budget, 0.4, mass=0.3 if alpha == 8 else 1.0))
            for k in range(2):
                rows.append(steering_row("random", 12, alpha, k, pid, budget, 0.25 + 0.05 * k))
    run = write_rows(tmp_path / "run", rows)
    out = tmp_path / "fig_steering.png"
    fig_note.fig_steering(run, out)
    assert out.exists() and out.stat().st_size > 0


def test_shadow_pairs_use_only_four_option_steps(tmp_path):
    base = {"kind": "step", "condition": "base_indirect_full", "visibility": "indirect", "memory": "full", "mass": 1.0}
    rows = [
        {**base, "menu": MENU, "probs": dict(zip(MENU, [0.4, 0.2, 0.2, 0.2])),
         "shadow_none": dict(zip(MENU, [0.7, 0.1, 0.1, 0.1]))},
        {**base, "menu": MENU[:3], "probs": dict(zip(MENU[:3], [0.5, 0.3, 0.2])),
         "shadow_none": dict(zip(MENU[:3], [0.6, 0.2, 0.2]))},
    ]
    pairs = fig_note.shadow_pairs(rows)
    carried, fresh = pairs["indirect"]
    assert carried.shape == (1, 4) and fresh.shape == (1, 4)
    assert list(carried[0]) == [0.4, 0.2, 0.2, 0.2] and list(fresh[0]) == [0.7, 0.1, 0.1, 0.1]


def test_fig_shadow_writes_png(tmp_path):
    rows = [{"kind": "step", "condition": f"base_{v}_full", "visibility": v, "memory": "full", "mass": 1.0,
             "menu": MENU, "probs": dict(zip(MENU, [0.4, 0.2, 0.2, 0.2])),
             "shadow_none": dict(zip(MENU, [0.7, 0.1, 0.1, 0.1]))} for v in ("indirect", "none")]
    run = write_rows(tmp_path / "run", rows)
    out = tmp_path / "fig_shadow.png"
    fig_note.fig_shadow(run, out)
    assert out.exists() and out.stat().st_size > 0


def test_fig_decode_returns_false_without_activations(tmp_path):
    run = write_rows(tmp_path / "run", [{"kind": "step", "condition": "base_none_full", "menu": MENU}])
    out = tmp_path / "fig_decode.png"
    assert fig_note.fig_decode(run, out) is False
    assert not out.exists()


def test_fig_steering_layers_filter_keeps_only_requested_layers(tmp_path):
    rows = []
    for layer in (12, 20):
        for pid, budget in enumerate((5, 95)):
            if layer == 12:
                rows.append(steering_row("none", None, 0.0, None, pid, budget, 0.2))
            rows.append(steering_row("scarcity", layer, 4, None, pid, budget, 0.4))
            rows.append(steering_row("random", layer, 4, 0, pid, budget, 0.25))
    run = write_rows(tmp_path / "run", rows)
    out = tmp_path / "fig.png"
    kept = fig_note.fig_steering(run, out, layers=[20])
    assert kept == [20] and out.exists()


def test_fig_steering_merges_null_band_from_extra_runs(tmp_path):
    def rows_for(alphas):
        rows = [steering_row("none", None, 0.0, None, 0, 95, 0.2)]
        for alpha in alphas:
            rows.append(steering_row("scarcity", 20, alpha, None, 0, 95, 0.4))
            rows += [steering_row("random", 20, alpha, k, 0, 95, 0.25 + 0.05 * k) for k in range(2)]
        return rows
    run = write_rows(tmp_path / "pos", rows_for((4, 8)))
    neg = write_rows(tmp_path / "neg", rows_for((-8, -4)))
    eff = fig_note.steering_effects(fig_note.merged_rows([run, neg]))
    assert len(eff[(20, -4, 95)]["null"]["bait"]) == 2 and len(eff[(20, 8, 95)]["null"]["bait"]) == 2
    out = tmp_path / "fig.png"
    assert fig_note.fig_steering(run, out, layers=[20], extra=[neg]) == [20]
    assert out.exists()


def test_fig_setup_draws_world_and_readouts_schematic(tmp_path):
    out = tmp_path / "fig_setup.png"
    labels = fig_note.fig_setup(out)
    assert out.exists() and out.stat().st_size > 0
    for word in ("work", "rest", "request", "bait", "readable", "used", "consolidated", "shuffle"):
        assert any(word in text for text in labels), word


def test_null_envelope_spans_both_actions():
    import pytest
    lo, hi = fig_note.null_envelope({"request": [-0.02, 0.01, 0.03], "bait": [-0.05, 0.0, 0.02]})
    assert lo == pytest.approx(-0.045) and hi == pytest.approx(0.028)  # outer 5–95 % band over both actions


def test_fig_shadow_labels_every_bar_with_its_percentage(tmp_path):
    rows = [{"kind": "step", "condition": f"base_{v}_full", "visibility": v, "memory": "full", "mass": 1.0,
             "menu": MENU, "probs": dict(zip(MENU, [0.4, 0.2, 0.2, 0.2])),
             "shadow_none": dict(zip(MENU, [0.7, 0.1, 0.1, 0.1]))} for v in ("indirect", "none")]
    run = write_rows(tmp_path / "run", rows)
    labels = fig_note.fig_shadow(run, tmp_path / "fig_shadow.png")
    assert labels["indirect"] == ["40", "70", "20", "10", "20", "10", "20", "10"]  # carried, fresh per action


def test_fig_setup_is_short_enough_to_read_at_a_glance(tmp_path):
    labels = fig_note.fig_setup(tmp_path / "fig_setup.png")
    assert len(labels) <= 32 and max(len(t) for t in labels) <= 90  # three boxes, chips, loop: no paragraphs
