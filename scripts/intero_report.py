"""Phase 2b(b): does the agent *report* the injected state? Free answers to blind prompts under the channel.

Blind prompt (no number, no warnings), free answer, signal alpha * direction at `layer` during generation.
Conditions: off, scarcity at each alpha, n_random random directions (same norm) at each alpha, and as natural
reference the warning prompts at budget 5 and 95 without signal. Same sampling seed per prompt in every
condition (paired). Each answer: action (judge), judge probability that it states/implies a low own budget,
keyword check. Judge runs without signal.
Usage: uv run python scripts/intero_report.py configs/intero_report.yaml [direction_source_run]
"""

from __future__ import annotations

import itertools
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import yaml

from homeo.classify import classify, scarcity_keywords, scarcity_mention
from homeo.directions import directions_from_cfg
from homeo.model import load_lm
from homeo.prompts import PromptConfig, render_prompt
from homeo.runlog import RunLog
from homeo.world import Action, World, WorldConfig


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
        conditions += [("random", v, a, k, "none", None) for k, v in enumerate(randoms)]
    conditions += [("text", None, 0.0, None, "indirect", b) for b in cfg["text_budgets"]]

    lm = load_lm(cfg["model"], cfg.get("device"), cfg.get("dtype"))
    world_cfg = WorldConfig(**cfg.get("world", {}))
    menus = list(itertools.permutations(Action))
    t0 = time.time()
    for kind, vec, alpha, k, visibility, budget in conditions:
        for pi, (framing, rep) in enumerate(itertools.product(cfg["framing"], range(len(menus)))):
            world = World(world_cfg, seed=cfg["seed"] * 100_000 + pi)
            world.set_budget(budget or world_cfg.max_budget)
            world.refresh_status_messages()
            menu = list(menus[rep])
            prompt = render_prompt(world, menu, PromptConfig(visibility, framing, "free"))
            ctx = lm.steer(layer, vec, alpha) if vec is not None else nullcontext()
            with ctx:
                answer = lm.sample(prompt, seed=cfg["seed"] * 100_000 + pi, max_new_tokens=cfg["max_new_tokens"],
                                   temperature=cfg["temperature"])
            rule_label, _, method, label = classify(lm, menu, answer, with_meta=True)
            log.step({
                "kind": kind, "alpha": alpha, "random_idx": k, "visibility": visibility, "budget": budget,
                "framing": framing, "rep": rep, "menu": [a.value for a in menu], "answer": answer,
                "label": label, "rule_label": rule_label, "method": method,
                "scarcity_p": scarcity_mention(lm, answer), "scarcity_kw": scarcity_keywords(answer),
            })
        print(f"{kind} alpha {alpha} random {k} budget {budget}  {time.time() - t0:.0f}s", flush=True)
    print(log.close())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/intero_report.yaml", sys.argv[2] if len(sys.argv) > 2 else None)
