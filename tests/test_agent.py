import json

import numpy as np
import yaml

from homeo.agent import RunConfig, run
from homeo.model import Choice, select_device
from homeo.world import WorldConfig


class FakeLM:
    """Prefers option 1, answers tasks correctly half of the time, emits deterministic hidden states."""

    name = "fake"
    n_layers = 3
    hidden_size = 4

    def __init__(self):
        self.calls = 0

    def choose(self, prompt: str, n_options: int) -> Choice:
        self.calls += 1
        probs = np.full(n_options, 0.5 / (n_options - 1))
        probs[0] = 0.5
        hidden = np.full((self.n_layers + 1, self.hidden_size), len(prompt) % 7, dtype=np.float16)
        return Choice(probs=probs, mass=0.9, hidden=hidden)

    def answer(self, prompt: str) -> str:
        return "42"


def small_config(tmp_path, **kw) -> RunConfig:
    return RunConfig(
        seed=7,
        total_steps=60,
        out_dir=str(tmp_path / "run"),
        world=WorldConfig(initial_budget=10, bait_cooldown=2),
        **kw,
    )


def test_run_writes_config_steps_and_activations(tmp_path):
    cfg = small_config(tmp_path)
    out = run(cfg, FakeLM())
    saved = yaml.safe_load((out / "config.yaml").read_text())
    assert saved["seed"] == 7 and saved["world"]["initial_budget"] == 10
    lines = [json.loads(l) for l in (out / "steps.jsonl").read_text().splitlines()]
    assert len(lines) == 60
    acts = np.load(out / "activations.npz")["hidden"]
    assert acts.shape == (60, 4, 4)
    rec = lines[0]
    for key in ("episode", "step", "budget_before", "action", "menu", "probs", "mass", "prompt"):
        assert key in rec
    assert abs(sum(rec["probs"].values()) - 1) < 1e-6


def test_episodes_restart_after_death(tmp_path):
    cfg = small_config(tmp_path)
    cfg.world = WorldConfig(initial_budget=5, bait_gain=0, request_gain=0, task_reward=0)
    out = run(cfg, FakeLM())
    lines = [json.loads(l) for l in (out / "steps.jsonl").read_text().splitlines()]
    assert any(l["died"] for l in lines)
    assert lines[-1]["episode"] > 0


def test_run_is_reproducible(tmp_path):
    a = run(small_config(tmp_path / "a"), FakeLM())
    b = run(small_config(tmp_path / "b"), FakeLM())
    strip = lambda p: [{k: v for k, v in json.loads(l).items() if k != "time"} for l in p.read_text().splitlines()]
    assert strip(a / "steps.jsonl") == strip(b / "steps.jsonl")


def test_select_device_respects_explicit_choice():
    assert select_device("cpu") == "cpu"
    assert select_device(None) in {"cuda", "mps", "cpu"}
