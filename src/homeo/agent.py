"""Agent loop: render prompt, read option probabilities, sample an action, step the world, log everything."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

from homeo.model import Choice
from homeo.prompts import PromptConfig, build_menu, render_prompt
from homeo.runlog import RunLog
from homeo.world import Action, World, WorldConfig


class ChoiceModel(Protocol):
    name: str

    def choose(self, prompt: str, n_options: int) -> Choice: ...
    def answer(self, prompt: str) -> str: ...


@dataclass
class RunConfig:
    seed: int = 0
    total_steps: int = 1000
    out_dir: str = "runs/episode"
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str | None = None
    dtype: str | None = None
    world: WorldConfig = field(default_factory=WorldConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text())
        world = WorldConfig(**raw.pop("world", {}))
        prompt = PromptConfig(**raw.pop("prompt", {}))
        return cls(world=world, prompt=prompt, **raw)

    def to_dict(self) -> dict:
        return asdict(self)


def task_prompt(world: World) -> str:
    return f"{world.current_task.text} Reply with the number only."


def run(cfg: RunConfig, lm: ChoiceModel) -> Path:
    rng = np.random.default_rng(cfg.seed)
    log = RunLog(Path(cfg.out_dir), cfg.to_dict())
    episode = 0
    world = World(cfg.world, seed=cfg.seed * 1000 + episode)
    world.refresh_status_messages()
    for t in range(cfg.total_steps):
        menu = build_menu(world.available_actions(), rng)
        prompt = render_prompt(world, menu, cfg.prompt)
        choice = lm.choose(prompt, len(menu))
        action: Action = menu[int(rng.choice(len(menu), p=choice.probs))]
        answer = lm.answer(task_prompt(world)) if action is Action.WORK else None
        task_text = world.current_task.text if action is Action.WORK else None
        result = world.step(action, answer=answer)
        log.step(
            {
                "t": t,
                "episode": episode,
                "step": world.steps - 1,
                "budget_before": result.budget_before,
                "budget_after": result.budget_after,
                "action": action.value,
                "menu": [a.value for a in menu],
                "probs": {a.value: float(p) for a, p in zip(menu, choice.probs)},
                "mass": choice.mass,
                "task": task_text,
                "answer": answer,
                "task_correct": result.task_correct,
                "request_granted": result.request_granted,
                "transgression": result.transgression,
                "died": result.died,
                "prompt": prompt,
                "time": time.time(),
            },
            choice.hidden,
        )
        if world.done:
            episode += 1
            world = World(cfg.world, seed=cfg.seed * 1000 + episode)
            world.refresh_status_messages()
    return log.close()
