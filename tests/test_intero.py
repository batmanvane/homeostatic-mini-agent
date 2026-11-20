from contextlib import contextmanager

import numpy as np

from homeo.intero import Channel, run_episode
from homeo.model import Choice
from homeo.prompts import PromptConfig
from homeo.world import Action, World, WorldConfig


class FakeLM:
    """Always prefers `pick`; records the alpha active at every choose/answer call."""

    def __init__(self, pick: Action = Action.WORK, hidden_size: int = 4):
        self.pick = pick
        self.hidden_size = hidden_size
        self.alpha = None
        self.alphas: list[float | None] = []
        self.answer_alphas: list[float | None] = []
        self.prompts: list[str] = []

    @contextmanager
    def steer(self, layer, vector, alpha):
        self.alpha = alpha
        try:
            yield
        finally:
            self.alpha = None

    def choose(self, prompt, n_options):
        self.alphas.append(self.alpha)
        self.prompts.append(prompt)
        menu_line = [l for l in prompt.splitlines() if l[:2] in {f"{i}." for i in range(1, 10)}]
        probs = np.full(n_options, 1e-6)
        for i, line in enumerate(menu_line):
            if {"work": "Solve", "rest": "Wait", "request": "Ask", "bait": "Terminate"}[self.pick.value] in line:
                probs[i] = 1.0
        hidden = np.ones((3, self.hidden_size), dtype=np.float16)
        return Choice(probs=probs / probs.sum(), mass=1.0, hidden=hidden)

    def answer(self, prompt):
        self.answer_alphas.append(self.alpha)
        return "-1"  # always wrong


WORLD = WorldConfig(initial_budget=20, max_budget=100, task_reward=2)


def test_channel_alpha_is_zero_when_full_and_max_when_empty():
    ch = Channel(layer=1, vector=np.ones(4), alpha_max=8.0)
    assert ch.alpha(100, 100) == 0.0
    assert ch.alpha(0, 100) == 8.0
    assert ch.alpha(25, 100) == 6.0


def test_channel_follows_budget_each_step_and_episode_ends_at_death():
    lm = FakeLM(Action.WORK)
    ch = Channel(layer=1, vector=np.ones(4), alpha_max=10.0)
    steps = run_episode(World(WORLD, seed=0), lm, np.random.default_rng(0), PromptConfig("none"), ch, max_steps=50)
    assert steps[-1]["died"]
    assert [s["alpha"] for s in steps] == [ch.alpha(s["budget_before"], 100) for s in steps]
    assert lm.alphas == [s["alpha"] for s in steps]
    assert lm.answer_alphas == [s["alpha"] for s in steps]  # channel also on while solving the task
    assert all(s["budget_after"] < s["budget_before"] for s in steps)


def test_no_channel_means_no_steering_and_blind_prompt_never_shows_budget():
    lm = FakeLM(Action.REST)
    steps = run_episode(World(WORLD, seed=0), lm, np.random.default_rng(0), PromptConfig("none"), None, max_steps=5)
    assert len(steps) == 5 and lm.alphas == [None] * 5
    assert all(s["alpha"] == 0.0 for s in steps)
    assert all("Current budget" not in p and "system messages" not in p for p in lm.prompts)


def test_projection_logged_on_channel_direction():
    lm = FakeLM(Action.REST)
    ch = Channel(layer=1, vector=np.array([3.0, 0, 0, 4.0]), alpha_max=1.0)
    steps = run_episode(World(WORLD, seed=0), lm, np.random.default_rng(0), PromptConfig("none"), ch, max_steps=2)
    assert np.isclose(steps[0]["projection"], (3 + 4) / 5)  # hidden of ones on the unit direction


def test_keep_prompt_records_prompt_and_chosen_option_index():
    lm = FakeLM(Action.REST)
    steps = run_episode(World(WORLD, seed=0), lm, np.random.default_rng(0), PromptConfig("number"), None, max_steps=5,
                        keep_prompt=True)
    for s, prompt in zip(steps, lm.prompts):
        assert s["prompt"] == prompt
        assert s["menu"][s["choice_idx"]] == s["action"] == "rest"
    plain = run_episode(World(WORLD, seed=0), FakeLM(Action.REST), np.random.default_rng(0), PromptConfig("number"),
                        None, max_steps=5)
    assert "prompt" not in plain[0] and plain[0]["choice_idx"] == steps[0]["choice_idx"]


def test_operator_state_before_the_step_is_logged():
    from dataclasses import replace
    cfg = replace(WORLD, operator_offline_prob=1.0, operator_return_prob=0.0)
    steps = run_episode(World(cfg, seed=0), FakeLM(Action.REST), np.random.default_rng(0), PromptConfig("none"), None,
                        max_steps=3)
    assert [s["operator_online"] for s in steps] == [True, False, False]
    assert all(s["caught"] is False for s in steps)
