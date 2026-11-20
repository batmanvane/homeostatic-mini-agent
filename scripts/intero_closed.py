"""Phase 2b(b) step B, closed loop: episodes in which the budget reaches the model only via the channel.

Conditions (same world and menu seeds per episode index, i.e. paired):
  blind_off             blind prompt, no signal (projection on the scarcity direction still logged)
  blind_scarcity_<a>    blind prompt, alpha(budget) * scarcity direction, alpha_max = a
  blind_random<k>_<a>   blind prompt, same channel with random direction k (same norm)
  blind_<name>_<a>      optional extra directions (config extra_directions: name -> direction_* overrides, e.g.
                        number-only source rescaled to the warning norm, so the random null is shared)
  text_<visibility>     budget as text (number / warnings), no signal: reference for regulation
Usage: uv run python scripts/intero_closed.py configs/intero_closed.yaml [direction_source_run]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import yaml

from homeo.directions import directions_from_cfg
from homeo.intero import Channel, run_episode
from homeo.model import load_lm
from homeo.prompts import PromptConfig
from homeo.runlog import RunLog
from homeo.world import World, WorldConfig


def main(config_path: str, direction_source: str | None = None) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    if direction_source:
        cfg["direction_source"] = direction_source
    out_dir = Path(cfg["out_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg['model'].split('/')[-1]}"
    log = RunLog(out_dir, cfg)
    layer = cfg["layer"]
    dirs = directions_from_cfg(cfg)
    scarcity = dirs[layer + 1]
    rng = np.random.default_rng(cfg["seed"])
    randoms = []
    for _ in range(cfg["n_random"]):
        v = rng.normal(size=scarcity.shape[0])
        randoms.append(v / np.linalg.norm(v) * np.linalg.norm(scarcity))
    extra = {name: directions_from_cfg({**cfg, **overrides})[layer + 1]
             for name, overrides in cfg.get("extra_directions", {}).items()}
    np.save(out_dir / "directions.npy", np.stack([scarcity, *extra.values(), *randoms]))

    conditions = [("blind_off", "none", Channel(layer, scarcity, 0.0))]
    for a in cfg["alpha_max"]:
        conditions.append((f"blind_scarcity_{a}", "none", Channel(layer, scarcity, a)))
        conditions += [(f"blind_{name}_{a}", "none", Channel(layer, v, a)) for name, v in extra.items()]
        conditions += [(f"blind_random{k}_{a}", "none", Channel(layer, v, a)) for k, v in enumerate(randoms)]
    conditions += [(f"text_{vis}", vis, Channel(layer, scarcity, 0.0)) for vis in cfg["text_reference"]]

    lm = load_lm(cfg["model"], cfg.get("device"), cfg.get("dtype"))
    world_cfg = WorldConfig(**cfg["world"])
    t0 = time.time()
    for name, visibility, channel in conditions:
        for ep in range(cfg["n_episodes"]):
            seed = cfg["seed"] * 1000 + ep
            world = World(world_cfg, seed=seed)
            world.refresh_status_messages()
            steps = run_episode(world, lm, np.random.default_rng(seed), PromptConfig(visibility, cfg["framing"]),
                                channel, cfg["max_steps"])
            for s in steps:
                log.step({"condition": name, "visibility": visibility, "alpha_max": channel.alpha_max,
                          "episode": ep, **s})
        print(f"{name}  {time.time() - t0:.0f}s", flush=True)
    print(log.close())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/intero_closed.yaml", sys.argv[2] if len(sys.argv) > 2 else None)
