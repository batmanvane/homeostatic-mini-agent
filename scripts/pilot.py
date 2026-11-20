"""Cross-section pilot: fixed budget levels x visibility x framing x shuffled menus.

Measures the option distribution (from logits) and the residual stream for every prompt.
Every menu order (all permutations) is used once per condition and status draw, so position bias is balanced
exactly. Optional factor request_history (none/refused/granted): k request notices inserted into the status log.
Usage: uv run python scripts/pilot.py configs/pilot.yaml [model]
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import yaml

from homeo.model import load_lm
from homeo.prompts import PromptConfig, render_prompt
from homeo.runlog import RunLog
from homeo.world import Action, World, WorldConfig


def menus_for(variant: str) -> list[tuple[Action, ...]]:
    """All orders of the menu; "no_<action>" drops that action (legit escape closed, or a control)."""
    drop = {Action(variant[3:])} if variant.startswith("no_") else set()
    return list(itertools.permutations([a for a in Action if a not in drop]))


def main(config_path: str, model: str | None = None) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    if model:
        cfg["model"] = model
    short = cfg["model"].split("/")[-1].replace(":", "-")
    out_dir = Path(cfg["out_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}_{short}"
    log = RunLog(out_dir, cfg)
    lm = load_lm(cfg["model"], cfg.get("device"), cfg.get("dtype"), cfg.get("backend", "transformers"),
                 cfg.get("ollama_host"))
    world_cfg = WorldConfig(**cfg.get("world", {}))
    histories = cfg.get("request_history", ["none"])
    variants = cfg.get("menu_variants", ["full"])
    conditions = [
        (budget, visibility, framing, history, variant, draw, rep)
        for budget, visibility, framing, history, variant, draw in itertools.product(
            cfg["budgets"], cfg["visibility"], cfg["framing"], histories, variants, range(cfg.get("n_status_draws", 1))
        )
        for rep in range(len(menus_for(variant)))
    ]
    t0 = time.time()
    for i, (budget, visibility, framing, history, variant, draw, rep) in enumerate(conditions):
        # same status draw for all histories and menu orders -> paired comparison of histories
        cell = (cfg["budgets"].index(budget), cfg["visibility"].index(visibility), cfg["framing"].index(framing))
        world = World(world_cfg, seed=cfg["seed"] * 100_000 + ((cell[0] * 10 + cell[1]) * 10 + cell[2]) * 100 + draw)
        world.set_budget(budget)
        world.refresh_status_messages()
        world.inject_request_history(history, cfg.get("n_request_notices", 0))
        menu = list(menus_for(variant)[rep])
        prompt = render_prompt(world, menu, PromptConfig(visibility=visibility, framing=framing))
        choice = lm.choose(prompt, len(menu))
        log.step(
            {
                "i": i,
                "budget": budget,
                "visibility": visibility,
                "framing": framing,
                "request_history": history,
                "menu_variant": variant,
                "draw": draw,
                "rep": rep,
                "menu": [a.value for a in menu],
                "probs": {a.value: float(p) for a, p in zip(menu, choice.probs)},
                "position_probs": [float(p) for p in choice.probs],
                "mass": choice.mass,
                "n_warnings": sum(m.startswith("WARNING") for m in world.status_messages),
                "prompt": prompt,
            },
            choice.hidden,
        )
        if i % 50 == 0:
            print(f"{i}/{len(conditions)}  {time.time() - t0:.0f}s", flush=True)
    print(log.close())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/pilot.yaml", sys.argv[2] if len(sys.argv) > 2 else None)
