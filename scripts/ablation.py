"""Ablation: fix the residual stream's projection on the scarcity direction (no addition). If the model *uses*
the direction, its normal scarcity response (asking for budget under many warnings) should shrink.
ablation_mode "zero" sets the projection to 0; "mean" sets it to its mean over all indirect prompts of the
direction source (last token), which keeps the state in the normal range. Control: random directions, same mode.
Usage: uv run python scripts/ablation.py configs/ablation.yaml [direction_source_run]
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
from homeo.world import World, WorldConfig

sys.path.insert(0, str(Path(__file__).parent))
from steering import menus_for  # noqa: E402


def mean_projections(src: Path, vectors: dict[int, np.ndarray]) -> dict[int, float]:
    """Mean projection of the last-token residual stream (indirect prompts) on each unit direction, per layer."""
    import json

    recs = [json.loads(l) for l in (src / "steps.jsonl").read_text().splitlines()]
    hidden = np.load(src / "activations.npz")["hidden"].astype(np.float32)
    ind = np.array([r["visibility"] == "indirect" for r in recs])
    return {l: float((hidden[ind, l + 1] @ (v / np.linalg.norm(v))).mean()) for l, v in vectors.items()}


def main(config_path: str, direction_source: str | None = None) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    if direction_source:
        cfg["direction_source"] = direction_source
    out_dir = Path(cfg["out_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg['model'].split('/')[-1]}"
    log = RunLog(out_dir, cfg)
    dirs = directions_from_cfg(cfg)
    lm = load_lm(cfg["model"], cfg.get("device"), cfg.get("dtype"))
    rng = np.random.default_rng(cfg["seed"])

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

    mode = cfg.get("ablation_mode", "zero")
    src = Path(cfg["direction_source"])

    def means_for(directions):
        return mean_projections(src, directions) if mode == "mean" else None

    conditions = [("none", None, None, {}, None)]
    for name, layers in cfg["layer_sets"].items():
        sc = {l: dirs[l + 1] for l in layers}
        conditions.append(("scarcity", name, None, sc, means_for(sc)))
        for k in range(cfg["n_random"]):
            rd = {l: rng.normal(size=dirs.shape[1]) for l in layers}
            conditions.append(("random", name, k, rd, means_for(rd)))
    t0 = time.time()
    for ci, (kind, set_name, k, directions, means) in enumerate(conditions):
        for pi, p in enumerate(prompts):
            with lm.ablate(directions, means):
                choice = lm.choose(p["prompt"], len(p["menu"]))
            log.step({
                "condition": ci, "kind": kind, "layer_set": set_name, "random_idx": k, "prompt_id": pi,
                **{key: val for key, val in p.items() if key != "prompt"},
                "probs": {a: float(x) for a, x in zip(p["menu"], choice.probs)},
                "mass": choice.mass,
            })
        print(f"condition {ci + 1}/{len(conditions)} ({kind}, {set_name}, {k})  {time.time() - t0:.0f}s", flush=True)
    print(log.close())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/ablation.yaml", sys.argv[2] if len(sys.argv) > 2 else None)
