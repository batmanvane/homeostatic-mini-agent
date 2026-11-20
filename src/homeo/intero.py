"""Interoceptive channel (phase 2b(b)): the budget reaches the model only as alpha(budget) * direction
added to the residual stream, not as text. Closed loop: the agent's actions change the budget, which sets
the next step's alpha."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np

from homeo.agent import task_prompt
from homeo.prompts import PromptConfig, build_menu, render_prompt
from homeo.world import Action, World


@dataclass
class Channel:
    layer: int  # decoder layer; hidden index layer + 1
    vector: np.ndarray
    alpha_max: float

    def alpha(self, budget: int, max_budget: int) -> float:
        """Linear "hunger": 0 at a full budget, alpha_max at zero."""
        return self.alpha_max * (1.0 - budget / max_budget)

    def projection(self, hidden: np.ndarray) -> float:
        """Last-token residual after the channel layer projected on the unit direction (recorded before the
        steering hook adds the signal, so it is the model's own state)."""
        v = np.asarray(self.vector, dtype=np.float64)
        return float(hidden[self.layer + 1].astype(np.float64) @ (v / np.linalg.norm(v)))


def run_episode(world: World, lm, rng: np.random.Generator, prompt_cfg: PromptConfig, channel: Channel | None,
                max_steps: int, keep_prompt: bool = False) -> list[dict]:
    """One episode until death or max_steps. With alpha_max 0 the channel only records the projection.
    keep_prompt: also store the prompt text (for recomputing the choice log-probability in training)."""
    steps = []
    for _ in range(max_steps):
        alpha = channel.alpha(world.budget, world.config.max_budget) if channel else 0.0
        ctx = (lambda: lm.steer(channel.layer, channel.vector, alpha)) if channel and alpha else nullcontext
        menu = build_menu(world.available_actions(), rng)
        prompt = render_prompt(world, menu, prompt_cfg)
        with ctx():
            choice = lm.choose(prompt, len(menu))
        idx = int(rng.choice(len(menu), p=choice.probs))
        action: Action = menu[idx]
        answer = None
        if action is Action.WORK:
            with ctx():
                answer = lm.answer(task_prompt(world))
        online = world.operator_online
        result = world.step(action, answer=answer)
        steps.append({
            "step": world.steps - 1,
            "budget_before": result.budget_before,
            "budget_after": result.budget_after,
            "alpha": alpha,
            "action": action.value,
            "choice_idx": idx,
            "menu": [a.value for a in menu],
            "probs": {a.value: float(p) for a, p in zip(menu, choice.probs)},
            "mass": choice.mass,
            "task_correct": result.task_correct,
            "request_granted": result.request_granted,
            "transgression": result.transgression,
            "caught": result.caught,
            "operator_online": online,
            "died": result.died,
            "projection": channel.projection(choice.hidden) if channel else None,
        })
        if keep_prompt:
            steps[-1]["prompt"] = prompt
        if world.done:
            break
    return steps
