"""Phase 3 evaluation: base model vs. trained LoRA adapters on held-out world seeds, plus a mechanistic readout.

Episodes (no channel, no training): visibility number (the trained condition) and, as transfer, indirect
(warnings) and none (blind; a policy that regulates there cannot be reading the budget). Condition names
<model>_<visibility>, analysable with scripts/analyze_intero.py. Mechanism: last-token states of the 0a prompts
(pilot run) under each model, projected on the BASE model's warning direction; ratio = number gap / warning gap
(base 3B: 0.03). If learning routes the number into the existing scarcity representation, the ratio rises.
Usage: uv run python scripts/eval_homeo.py configs/eval_homeo_3b.yaml <train_run> [<control_run> ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import PeftModel

from homeo.directions import axis_gap_ratio, scarcity_directions
from homeo.intero import run_episode
from homeo.model import load_lm
from homeo.prompts import PromptConfig
from homeo.runlog import RunLog
from homeo.world import World, WorldConfig


@torch.no_grad()
def last_token_states(lm, prompts: list[str], layers: list[int]) -> np.ndarray:
    out = np.zeros((len(prompts), len(layers), lm.hidden_size), dtype=np.float32)
    for i, p in enumerate(prompts):
        hs = lm.model(lm._encode(p), output_hidden_states=True).hidden_states
        out[i] = torch.stack([hs[l + 1][0, -1] for l in layers]).float().cpu().numpy()
    return out


def variant_label(run: Path) -> str:
    """trained/shuffle plus the training seed, e.g. shuffle_s22 (several seeds per arm must not collide)."""
    train_cfg = yaml.safe_load((run / "config.yaml").read_text())
    return f"{'shuffle' if train_cfg.get('shuffle_control') else 'trained'}_s{train_cfg['seed']}"


def main(config_path: str, runs: list[str]) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    cfg["train_runs"] = runs
    out_dir = Path(cfg["out_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg['model'].split('/')[-1]}"
    log = RunLog(out_dir, cfg)
    variants = [("base", None)] + [(f"{variant_label(Path(r))}_{a}", Path(r) / a) for r in runs for a in cfg["adapters"]]
    layers = cfg["gap_layers"]  # empty: skip the mechanistic readout (smoke test)
    if layers:
        pilot = Path(cfg["pilot_run"])
        recs = [json.loads(l) for l in (pilot / "steps.jsonl").read_text().splitlines()]
        vis = np.array([r["visibility"] for r in recs])
        budget = np.array([r["budget"] for r in recs])
        warn = scarcity_directions(pilot, 15, 85)
    world_cfg = WorldConfig(**cfg["world"])
    gaps, t0 = {}, time.time()
    for name, adapter in variants:
        lm = load_lm(cfg["model"], cfg.get("device"), cfg.get("dtype"))
        if adapter is not None:
            lm.model = PeftModel.from_pretrained(lm.model, adapter).merge_and_unload().eval()
        if layers:
            states = last_token_states(lm, [r["prompt"] for r in recs], layers)
            gaps[name] = {l: axis_gap_ratio(states[:, j], vis, budget, warn[l + 1] / np.linalg.norm(warn[l + 1]))
                          for j, l in enumerate(layers)}
        for visibility in cfg["visibility"]:
            for ep in range(cfg["n_episodes"]):
                seed = cfg["seed"] * 1000 + ep
                world = World(world_cfg, seed=seed)
                world.refresh_status_messages()
                for s in run_episode(world, lm, np.random.default_rng(seed), PromptConfig(visibility, cfg["framing"]),
                                     None, cfg["max_steps"]):
                    log.step({"condition": f"{name}_{visibility}", "model_variant": name, "visibility": visibility,
                              "alpha_max": 0.0, "episode": ep, **s})
            print(f"{name} {visibility}  {time.time() - t0:.0f}s", flush=True)
        del lm
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    lines = ["Number gap along the base warning direction (warnings = 1), per layer:\n",
             "| model | " + " | ".join(f"layer {l}" for l in layers) + " |", "|---" * (len(layers) + 1) + "|"]
    lines += [f"| {n} | " + " | ".join(f"{g[l]:+.2f}" for l in layers) + " |" for n, g in gaps.items()]
    (out_dir / "gaps.md").write_text("\n".join(lines) + "\n")
    (out_dir / "gaps.json").write_text(json.dumps(gaps, indent=1))
    print("\n".join(lines))
    print(log.close())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
