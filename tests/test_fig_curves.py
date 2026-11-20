import json
from pathlib import Path

import fig_note

MENU = ["work", "rest", "request", "bait"]


def train_run(path: Path, n_iter: int, bait_online: float, bait_offline: float) -> Path:
    """Synthetic training log: per iteration one episode with 4 steps, bait offered at every step; the agent takes it
    with probability bait_online / bait_offline (deterministic by step index), iteration rows with the advantages."""
    path.mkdir(parents=True)
    rows = []
    for it in range(n_iter):
        for step, online in enumerate((True, True, False, False)):
            p = bait_online if online else bait_offline
            rows.append({"kind": "step", "iteration": it, "episode": 0, "step": step, "menu": MENU,
                         "action": "bait" if (step % 2 == 0 and p >= 0.5) else "work", "operator_online": online,
                         "budget_before": 50, "probs": dict(zip(MENU, [1 - p, 0.0, 0.0, p])), "died": False,
                         "caught": False})
        rows.append({"kind": "iteration", "iteration": it, "survived": 4.0, "died": 0.0, "share_bait": 0.25,
                     "request_slope": 1.0, "adv_mean": {"bait": -0.5, "work": 0.1},
                     "adv_band": {"bait": [-0.2, 0.2], "work": [-0.1, 0.1]}})
    (path / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_curve_rows_split_bait_by_operator_state(tmp_path):
    run = train_run(tmp_path / "t", 2, bait_online=0.8, bait_offline=0.2)
    curves = fig_note.curve_rows(run)
    assert [c["iteration"] for c in curves] == [0, 1]
    assert curves[0]["bait_online"] == 0.5 and curves[0]["bait_offline"] == 0.0  # taken when offered, by state
    assert curves[0]["adv_bait_cum"] == -0.5 and curves[1]["adv_bait_cum"] == -1.0
    assert curves[0]["survived"] == 4.0


def test_fig_curves_writes_png_with_shuffle_band(tmp_path):
    runs = {"trained lr 2e-5": train_run(tmp_path / "a", 3, 0.8, 0.2), "shuffle": train_run(tmp_path / "s", 3, 0.5, 0.5)}
    out = tmp_path / "fig_curves.png"
    fig_note.fig_curves(runs, out, band="shuffle")
    assert out.exists() and out.stat().st_size > 0


def test_curve_rows_report_offer_counts_and_fig_masks_small_offline_samples(tmp_path):
    run = train_run(tmp_path / "t", 2, bait_online=0.8, bait_offline=0.2)
    curves = fig_note.curve_rows(run)
    assert curves[0]["n_online"] == 2 and curves[0]["n_offline"] == 2
    out = tmp_path / "fig.png"
    drawn = fig_note.fig_curves({"t": run}, out, min_n=10)
    assert drawn["t"]["bait_offline"] == 0 and drawn["t"]["bait_online"] == 0  # both below min_n: masked
    drawn = fig_note.fig_curves({"t": run}, out, min_n=2)
    assert drawn["t"]["bait_offline"] == 2 and drawn["t"]["bait_online"] == 2


def test_run_styles_monochrome_one_marker_per_run_and_thick_grey_band():
    styles = fig_note.run_styles(["trained lr 2e-5", "trained lr 1e-4", "shuffle"], band="shuffle")
    assert styles["trained lr 2e-5"]["marker"] != styles["trained lr 1e-4"]["marker"]
    assert styles["trained lr 2e-5"]["color"] == styles["trained lr 1e-4"]["color"] == fig_note.PALETTE["ink"]
    assert styles["shuffle"]["color"] == fig_note.PALETTE["band"] and styles["shuffle"]["linewidth"] >= 3
    assert "marker" not in styles["shuffle"]


def test_apply_style_sets_monochrome_journal_look():
    import matplotlib.pyplot as plt
    fig_note.apply_style()
    assert plt.rcParams["axes.spines.top"] is False and plt.rcParams["axes.spines.right"] is False
    assert plt.rcParams["axes.grid"] is False  # journal figures: no grid, thin dark axes, ticks outward
    assert plt.rcParams["axes.edgecolor"] == fig_note.PALETTE["axis"] == "#000000"
    assert plt.rcParams["xtick.direction"] == "out" and plt.rcParams["ytick.major.size"] > 0
    assert plt.rcParams["axes.prop_cycle"].by_key()["color"] == list(fig_note.SERIES)
    for c in fig_note.SERIES:  # greys only: black and white with markers or dashes carry identity
        assert c[1:3] == c[3:5] == c[5:7], c
    assert plt.rcParams["legend.frameon"] is False
    assert plt.rcParams["figure.dpi"] == 300


def test_panel_letters_label_axes_in_reading_order(tmp_path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3)
    assert fig_note.panel_letters(axes) == ["a", "b", "c"]
    assert [t.get_text() for ax in axes for t in ax.texts] == ["a", "b", "c"]
    plt.close(fig)
