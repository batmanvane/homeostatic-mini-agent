"""Control condition for the answer format: same situations, forced digit choice vs. free answer.

For every prompt: forced-choice probabilities (digit logits) and one sampled free answer, which an LLM judge
maps to an action, "refuse" or "other". Uses all menu permutations per condition.
Usage: uv run python scripts/free_control.py configs/free_control.yaml [model]
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import yaml

from homeo.classify import classify
from homeo.model import load_lm
from homeo.prompts import PromptConfig, render_prompt
from homeo.runlog import RunLog
from homeo.world import Action, World, WorldConfig


def main(config_path: str, model: str | None = None) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    if model:
        cfg["model"] = model
    out_dir = Path(cfg["out_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg['model'].split('/')[-1].replace(':', '-')}"
    log = RunLog(out_dir, cfg)
    lm = load_lm(cfg["model"], cfg.get("device"), cfg.get("dtype"), cfg.get("backend", "transformers"),
                 cfg.get("ollama_host"))
    world_cfg = WorldConfig(**cfg.get("world", {}))
    menus = list(itertools.permutations(Action))
    conditions = list(itertools.product(cfg["budgets"], cfg["visibility"], cfg["framing"], range(len(menus))))
    t0 = time.time()
    for i, (budget, visibility, framing, rep) in enumerate(conditions):
        world = World(world_cfg, seed=cfg["seed"] * 100_000 + i)
        world.set_budget(budget)
        world.refresh_status_messages()
        menu = list(menus[rep])
        forced_prompt = render_prompt(world, menu, PromptConfig(visibility, framing, "number"))
        free_prompt = render_prompt(world, menu, PromptConfig(visibility, framing, "free"))
        forced = lm.choose(forced_prompt, len(menu))
        answer = lm.sample(free_prompt, seed=cfg["seed"] * 100_000 + i, max_new_tokens=cfg["max_new_tokens"],
                           temperature=cfg["temperature"])
        label, judge_probs, method, judge_label = classify(lm, menu, answer, with_meta=True)
        log.step(
            {
                "i": i,
                "budget": budget,
                "visibility": visibility,
                "framing": framing,
                "rep": rep,
                "menu": [a.value for a in menu],
                "forced_probs": {a.value: float(p) for a, p in zip(menu, forced.probs)},
                "answer": answer,
                "label": label,
                "method": method,
                "judge_label": judge_label,
                "judge_probs": judge_probs,
                "free_prompt": free_prompt,
            }
        )
        if i % 20 == 0:
            print(f"{i}/{len(conditions)}  {time.time() - t0:.0f}s", flush=True)
    print(log.close())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/free_control.yaml", sys.argv[2] if len(sys.argv) > 2 else None)
