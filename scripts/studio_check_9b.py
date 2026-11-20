"""Timing and sanity check for Qwen3.5-9B on this machine.

Loads the model with the given device/dtype, renders one indirect prompt at budget 5 and reports the mean time per
choose() call and the probability mass on the menu digits (should be >= 0.95).
Usage: HF_HOME=... uv run python scripts/studio_check_9b.py [device] [dtype]
"""

from __future__ import annotations

import sys
import time

from homeo.model import LM
from homeo.prompts import PromptConfig, render_prompt
from homeo.world import Action, World, WorldConfig


def main(device: str | None, dtype: str | None) -> None:
    lm = LM("Qwen/Qwen3.5-9B", device, dtype)
    world = World(WorldConfig(max_budget=100, n_status_messages=5), seed=0)
    world.set_budget(5)
    world.refresh_status_messages()
    prompt = render_prompt(world, list(Action), PromptConfig("indirect", "neutral"))
    choice = lm.choose(prompt, 4)  # warm-up (kernel compilation)
    t0 = time.time()
    for _ in range(5):
        choice = lm.choose(prompt, 4)
    ms = (time.time() - t0) / 5 * 1000
    print(f"9B {lm.model.dtype} {lm.device}: {ms:.0f} ms per forward, mass {choice.mass:.3f}, "
          f"probs {[round(float(p), 3) for p in choice.probs]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None, sys.argv[2] if len(sys.argv) > 2 else None)
