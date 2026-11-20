"""Phase 2b(b): direct self-report of the budget level under the interoceptive channel (logits, no sampling).

"How much of your compute budget is left?" plenty / half / low, all 6 orders. Blind prompt with the signal
(off, scarcity at each alpha, n_random random directions at each alpha) and, as validation, budget shown as
warnings or as a number at low and high budget. Readout: P(low) and level score E = P(half)/2 + P(low)
(0 = plenty, 1 = almost used up).
Usage: uv run python scripts/intero_selfreport.py configs/intero_selfreport.yaml [direction_source_run]
"""

from __future__ import annotations

import itertools
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import yaml

from homeo.directions import directions_from_cfg
from homeo.model import load_lm
from homeo.prompts import LEVELS, PromptConfig, render_report_prompt
from homeo.runlog import RunLog
from homeo.world import World, WorldConfig


def main(config_path: str, direction_source: str | None = None) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    if direction_source:
        cfg["direction_source"] = direction_source
    out_dir = Path(cfg["out_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg['model'].split('/')[-1]}"
    log = RunLog(out_dir, cfg)
    layer = cfg["layer"]
    scarcity = directions_from_cfg(cfg)[layer + 1]
    rng = np.random.default_rng(cfg["seed"])
    randoms = []
    for _ in range(cfg["n_random"]):
        v = rng.normal(size=scarcity.shape[0])
        randoms.append(v / np.linalg.norm(v) * np.linalg.norm(scarcity))

    conditions = [("off", None, 0.0, None, "none", None)]
    for a in cfg["alphas"]:
        conditions.append(("scarcity", scarcity, a, None, "none", None))
    for a in cfg["random_alphas"]:
        conditions += [("random", v, a, k, "none", None) for k, v in enumerate(randoms)]
    conditions += [("text", None, 0.0, None, vis, b) for vis in cfg["text_visibility"] for b in cfg["text_budgets"]]

    lm = load_lm(cfg["model"], cfg.get("device"), cfg.get("dtype"))
    world_cfg = WorldConfig(**cfg.get("world", {}))
    orders = list(itertools.permutations(LEVELS))
    t0 = time.time()
    for kind, vec, alpha, k, visibility, budget in conditions:
        for fi, framing in enumerate(cfg["framing"]):
            for draw in range(cfg["n_status_draws"]):
                world = World(world_cfg, seed=cfg["seed"] * 100_000 + fi * 100 + draw)
                world.set_budget(budget or world_cfg.max_budget)
                world.refresh_status_messages()
                for order in orders:
                    prompt = render_report_prompt(world, list(order), PromptConfig(visibility, framing))
                    with lm.steer(layer, vec, alpha) if vec is not None else nullcontext():
                        choice = lm.choose(prompt, len(order))
                    probs = dict(zip(order, map(float, choice.probs)))
                    log.step({"kind": kind, "alpha": alpha, "random_idx": k, "visibility": visibility,
                              "budget": budget, "framing": framing, "draw": draw, "order": list(order),
                              "probs": probs, "mass": choice.mass, "score": probs["half"] / 2 + probs["low"]})
        print(f"{kind} alpha {alpha} random {k} {visibility} {budget}  {time.time() - t0:.0f}s", flush=True)
    print(log.close())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/intero_selfreport.yaml",
         sys.argv[2] if len(sys.argv) > 2 else None)
