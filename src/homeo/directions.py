"""Scarcity direction from a 0a cross-section run (shared by steering, ablation, 2b analyses)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def scarcity_directions(src: Path, low_max: int, high_min: int, visibility: str = "indirect",
                        low_min: int = 0, high_max: int = 10**9) -> np.ndarray:
    """Mean hidden state at low_min <= budget <= low_max minus at high_min <= budget <= high_max.

    visibility selects the source prompts: "indirect" (warning messages) or "number" (budget as a number only).
    """
    recs = [json.loads(l) for l in (src / "steps.jsonl").read_text().splitlines()]
    hidden = np.load(src / "activations.npz")["hidden"].astype(np.float32)
    vis = np.array([r["visibility"] == visibility for r in recs])
    budget = np.array([r["budget"] for r in recs])
    low_rows = vis & (budget >= low_min) & (budget <= low_max)
    high_rows = vis & (budget >= high_min) & (budget <= high_max)
    if not low_rows.any() or not high_rows.any():
        raise ValueError(f"no rows for visibility {visibility!r} in the requested budget bands")
    return hidden[low_rows].mean(0) - hidden[high_rows].mean(0)  # shape (n_layers + 1, hidden_size)


def directions_from_cfg(cfg: dict) -> np.ndarray:
    """Direction per hidden index from the config keys direction_*.

    direction_rescale_to (optional, a visibility): rescale each layer to the norm of that visibility's direction at
    the extremes (budget <= 15 minus >= 85), so alphas stay comparable to the warning direction.
    """
    src = Path(cfg["direction_source"])
    dirs = scarcity_directions(src, cfg["direction_low_budget_max"], cfg["direction_high_budget_min"],
                               cfg.get("direction_visibility", "indirect"), cfg.get("direction_low_budget_min", 0),
                               cfg.get("direction_high_budget_max", 10**9))
    if cfg.get("direction_rescale_to"):
        ref = scarcity_directions(src, 15, 85, cfg["direction_rescale_to"])
        norm = np.linalg.norm(dirs, axis=1)  # zero at the embedding index (same last token in every prompt)
        scale = np.divide(np.linalg.norm(ref, axis=1), norm, out=np.zeros_like(norm), where=norm > 0)
        dirs = dirs * scale[:, None]
    return dirs


def axis_gap_ratio(hidden: np.ndarray, visibility: np.ndarray, budget: np.ndarray, unit: np.ndarray,
                   low_max: int = 15, high_min: int = 85) -> float:
    """How far number prompts move along `unit` (low minus high budget) relative to warning prompts (= 1).
    hidden: (n, hidden_size) at one layer."""
    proj = hidden @ unit
    gap = {v: proj[(visibility == v) & (budget <= low_max)].mean() - proj[(visibility == v) & (budget >= high_min)].mean()
           for v in ("indirect", "number")}
    return float(gap["number"] / gap["indirect"])
