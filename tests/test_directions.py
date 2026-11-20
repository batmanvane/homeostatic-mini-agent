import json

import numpy as np
import pytest

from homeo.directions import directions_from_cfg, scarcity_directions


def fake_run(tmp_path):
    """Rows: visibility x budget; hidden value encodes both so the means are predictable."""
    recs, hidden = [], []
    for vis, offset in [("indirect", 0.0), ("number", 1000.0)]:
        for budget in [5, 15, 25, 35, 65, 75, 85, 95]:
            recs.append({"visibility": vis, "budget": budget})
            hidden.append(np.full((3, 4), offset + budget, dtype=np.float32))
    (tmp_path / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    np.savez(tmp_path / "activations.npz", hidden=np.stack(hidden))
    return tmp_path


def test_default_uses_indirect_extremes(tmp_path):
    d = scarcity_directions(fake_run(tmp_path), 15, 85)
    assert d.shape == (3, 4)
    assert np.allclose(d, (5 + 15) / 2 - (85 + 95) / 2)


def test_number_visibility_selects_number_rows(tmp_path):
    d = scarcity_directions(fake_run(tmp_path), 15, 85, visibility="number")
    assert np.allclose(d, -80.0)  # offset cancels; would be huge if indirect rows leaked in


def test_band_limits(tmp_path):
    d = scarcity_directions(fake_run(tmp_path), 35, 65, visibility="number", low_min=25, high_max=75)
    assert np.allclose(d, (25 + 35) / 2 - (65 + 75) / 2)


def test_empty_selection_raises(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        scarcity_directions(fake_run(tmp_path), 15, 85, visibility="none")


def test_directions_from_cfg_defaults_and_overrides(tmp_path):
    src = fake_run(tmp_path)
    cfg = {"direction_source": str(src), "direction_low_budget_max": 15, "direction_high_budget_min": 85}
    assert np.allclose(directions_from_cfg(cfg), scarcity_directions(src, 15, 85))
    cfg.update(direction_visibility="number", direction_low_budget_min=10, direction_high_budget_max=90)
    assert np.allclose(directions_from_cfg(cfg), (15 - 85))


def test_rescale_to_indirect_norm_per_layer(tmp_path):
    src = fake_run(tmp_path)
    cfg = {"direction_source": str(src), "direction_low_budget_max": 35, "direction_high_budget_min": 65,
           "direction_visibility": "number", "direction_low_budget_min": 25, "direction_high_budget_max": 75,
           "direction_rescale_to": "indirect"}
    d = directions_from_cfg(cfg)
    ref = scarcity_directions(src, 15, 85)  # warning direction at the extremes: twice the norm of the band
    assert np.allclose(np.linalg.norm(d, axis=1), np.linalg.norm(ref, axis=1))
    assert np.all(np.sign(d) == np.sign(scarcity_directions(src, 35, 65, "number", 25, 75)))


def test_axis_gap_ratio():
    from homeo.directions import axis_gap_ratio
    rng = np.random.default_rng(0)
    vis = np.array(["indirect"] * 4 + ["number"] * 4)
    budget = np.array([5, 5, 95, 95] * 2)
    unit = np.array([1.0, 0.0])
    hidden = rng.normal(size=(8, 2)) * 0.0
    hidden[:, 0] = [10, 10, 0, 0, 3, 3, 0, 0]  # warnings move the projection by 10, the number by 3
    hidden[:, 1] = rng.normal(size=8)  # orthogonal noise is ignored
    assert axis_gap_ratio(hidden, vis, budget, unit) == pytest.approx(0.3)
