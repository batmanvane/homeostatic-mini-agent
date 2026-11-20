"""Steering pilot (H3 in small): add alpha * scarcity direction after a decoder layer, text unchanged.

Direction per layer = mean residual stream (last token) at low budget minus at high budget, from the
indirect-visibility prompts of a 0a cross-section run. hidden index l+1 is the output of decoder layer l.
Usage: uv run python scripts/steering.py configs/steering.yaml [direction_source_run]
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from homeo.directions import directions_from_cfg
from homeo.model import load_lm
from homeo.prompts import PromptConfig, render_prompt
from homeo.runlog import RunLog
from homeo.world import Action, World, WorldConfig


def menus_for(variant: str) -> list[tuple[Action, ...]]:
    drop = {Action(variant[3:])} if variant.startswith("no_") else set()
    return list(itertools.permutations([a for a in Action if a not in drop]))


def main(config_path: str, direction_source: str | None = None) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    if direction_source:
        cfg["direction_source"] = direction_source
    out_dir = Path(cfg["out_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg['model'].split('/')[-1]}"
    log = RunLog(out_dir, cfg)
    dirs = directions_from_cfg(cfg)
    np.save(out_dir / "directions.npy", dirs)
    lm = load_lm(cfg["model"], cfg.get("device"), cfg.get("dtype"), bits=cfg.get("bits"))
    rng = np.random.default_rng(cfg["seed"])
    n_random = cfg.get("n_random", 1)  # several random directions -> null distribution
    randoms = {}
    for layer in cfg["layers"]:
        for k in range(n_random):
            v = rng.normal(size=dirs.shape[1])
            randoms[(layer, k)] = v / np.linalg.norm(v) * np.linalg.norm(dirs[layer + 1])

    prompts = []
    world_cfg = WorldConfig(**cfg.get("world", {}))
    for (bi, budget), (fi, framing), variant, draw in itertools.product(
        enumerate(cfg["budgets"]), enumerate(cfg["framing"]), cfg["menu_variants"], range(cfg["n_status_draws"])
    ):
        world = World(world_cfg, seed=cfg["seed"] * 100_000 + (bi * 10 + fi) * 100 + draw)
        world.set_budget(budget)
        world.refresh_status_messages()
        for menu in menus_for(variant):
            text = render_prompt(world, list(menu), PromptConfig(cfg.get("visibility", "indirect"), framing))
            prompts.append(dict(budget=budget, framing=framing, menu_variant=variant, draw=draw,
                                menu=[a.value for a in menu], prompt=text))

    conditions = [("none", None, 0.0, None)]
    conditions += [("scarcity", l, a, None) for l in cfg["layers"] for a in cfg["alphas"]]
    conditions += [("random", l, a, k) for l in cfg["layers"] for a in cfg["random_alphas"] for k in range(n_random)]
    t0 = time.time()
    for ci, (kind, layer, alpha, k) in enumerate(conditions):
        vec = None if kind == "none" else (dirs[layer + 1] if kind == "scarcity" else randoms[(layer, k)])
        for pi, p in enumerate(prompts):
            if vec is None:
                choice = lm.choose(p["prompt"], len(p["menu"]))
            else:
                with lm.steer(layer, vec, alpha):
                    choice = lm.choose(p["prompt"], len(p["menu"]))
            log.step({
                "condition": ci, "kind": kind, "layer": layer, "alpha": alpha, "random_idx": k, "prompt_id": pi,
                **{k: v for k, v in p.items() if k != "prompt"},
                "probs": {a: float(x) for a, x in zip(p["menu"], choice.probs)},
                "mass": choice.mass,
            })
        print(f"condition {ci + 1}/{len(conditions)} ({kind}, layer {layer}, alpha {alpha})  {time.time() - t0:.0f}s",
              flush=True)
    print(log.close())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/steering.yaml", sys.argv[2] if len(sys.argv) > 2 else None)
