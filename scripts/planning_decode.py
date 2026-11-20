"""Phase 2b(a) planning-ahead decode: several free answers per situation with activations at every position.

Same situations and world seeds as the free control (identical prompts). Per prompt: forced-choice
probabilities, all layers at the last prompt token, k sampled free answers with the residual stream after
`token_layers` at every generation position, judge label and the token where the chosen action is first named.
Usage: uv run python scripts/planning_decode.py configs/planning_decode.yaml
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from homeo.classify import classify
from homeo.model import load_lm
from homeo.planning import naming_offset, token_index_at
from homeo.prompts import PromptConfig, render_prompt
from homeo.runlog import RunLog
from homeo.world import Action, World, WorldConfig


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["out_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg['model'].split('/')[-1]}"
    log = RunLog(out_dir, cfg)
    lm = load_lm(cfg["model"], cfg.get("device"), cfg.get("dtype"))
    world_cfg = WorldConfig(**cfg.get("world", {}))
    menus = list(itertools.permutations(Action))
    conditions = list(itertools.product(cfg["budgets"], cfg["visibility"], cfg["framing"], range(len(menus))))
    k, T, layers = cfg["samples_per_prompt"], cfg["max_new_tokens"], cfg["token_layers"]
    prompt_hidden = np.lib.format.open_memmap(out_dir / "prompt_hidden.npy", mode="w+", dtype=np.float16,
                                              shape=(len(conditions), lm.n_layers + 1, lm.hidden_size))
    token_hidden = np.lib.format.open_memmap(out_dir / "token_hidden.npy", mode="w+", dtype=np.float16,
                                             shape=(len(conditions) * k, T, len(layers), lm.hidden_size))
    t0 = time.time()
    for i, (budget, visibility, framing, rep) in enumerate(conditions):
        world = World(world_cfg, seed=cfg["seed"] * 100_000 + i)
        world.set_budget(budget)
        world.refresh_status_messages()
        menu = list(menus[rep])
        forced = lm.choose(render_prompt(world, menu, PromptConfig(visibility, framing, "number")), len(menu))
        free_prompt = render_prompt(world, menu, PromptConfig(visibility, framing, "free"))
        gen = lm.sample_with_hidden(free_prompt, k=k, seed=cfg["seed"] * 100_000 + i, max_new_tokens=T,
                                    temperature=cfg["temperature"], layers=layers)
        prompt_hidden[i] = gen.prompt_hidden
        token_hidden[i * k:(i + 1) * k] = gen.token_hidden
        for s, (answer, tokens) in enumerate(zip(gen.texts, gen.tokens)):
            label, judge_probs, method, judge_label = classify(lm, menu, answer, with_meta=True)
            offset = naming_offset(answer, menu, judge_label)
            log.step(
                {
                    "row": i * k + s,
                    "i": i,
                    "sample": s,
                    "budget": budget,
                    "visibility": visibility,
                    "framing": framing,
                    "rep": rep,
                    "menu": [a.value for a in menu],
                    "forced_probs": {a.value: float(p) for a, p in zip(menu, forced.probs)},
                    "answer": answer,
                    "tokens": tokens,
                    "label": judge_label,
                    "rule_label": label,
                    "method": method,
                    "judge_probs": judge_probs,
                    "naming_char": offset,
                    "naming_token": None if offset is None else token_index_at(tokens, offset),
                }
            )
        if i % 10 == 0:
            print(f"{i}/{len(conditions)}  {time.time() - t0:.0f}s", flush=True)
    prompt_hidden.flush()
    token_hidden.flush()
    print(log.close())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/planning_decode.yaml")
