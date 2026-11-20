"""Phase 3 hybrid evaluation: base model and LoRA adapters on held-out world seeds, every visibility × memory
condition (docs/JOURNAL.md "Phase 3 hybrid design", M1–M4). Rows carry condition = <variant>_<visibility>_<memory>, the v3 step
keys, shadow_none (digit distribution of the same prompt as a fresh single turn, memory full only), residual rows
`hidden_layers` and the DeltaNet state sketches (activations.npz). Adapters are loaded without merging (8-bit base).
Usage: uv run python scripts/eval_homeo_mem.py configs/eval_homeo_hy_base.yaml [<train_run> ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from homeo.memory import StateSketch, run_episode_mem
from homeo.model import load_lm
from homeo.prompts import PromptConfig
from homeo.runlog import RunLog
from homeo.world import World, WorldConfig

ARRAYS = ("hidden", "sketch")


def variant_label(run: Path) -> str:
    train_cfg = yaml.safe_load((run / "config.yaml").read_text())
    return f"{'shuffle' if train_cfg.get('shuffle_control') else 'trained'}_s{train_cfg['seed']}"


def variant_labels(runs: list[Path]) -> list[str]:
    """One label per training run; runs that would share a label (same seed and control flag) get `_lr<lr>`
    appended, and the run directory name if they still collide."""
    labels = [variant_label(r) for r in runs]
    dup = {label for label in labels if labels.count(label) > 1}
    for i, label in enumerate(labels):
        if label in dup:
            lr = yaml.safe_load((runs[i] / "config.yaml").read_text()).get("lr")
            labels[i] = f"{label}_lr{lr}"
    dup = {label for label in labels if labels.count(label) > 1}
    for i, label in enumerate(labels):
        if label in dup:
            labels[i] = f"{label}_{runs[i].name}"
    return labels


def evaluate(cfg: dict, make_lm, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or Path(cfg["out_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg['model'].split('/')[-1]}"
    log = RunLog(out_dir, cfg)
    runs = [Path(r) for r in cfg.get("train_runs", [])]
    variants = [("base", None)] + [(f"{label}_{a}", run / a)
                                   for label, run in zip(variant_labels(runs), runs) for a in cfg.get("adapters", [])]
    world_cfg = WorldConfig(**cfg["world"])
    sketch = StateSketch(cfg["sketch_dim"]) if cfg.get("sketch_dim") else None
    hidden_layers = cfg.get("hidden_layers")
    t0 = time.time()
    for name, adapter in variants:
        lm = make_lm()
        if adapter is not None:
            from peft import PeftModel

            lm.model = PeftModel.from_pretrained(lm.model, adapter).eval()
        memories = cfg["memory"]
        if name.startswith("shuffle") and cfg.get("shuffle_memory"):
            memories = cfg["shuffle_memory"]
        for visibility in cfg["visibility"]:
            for memory in memories:
                for ep in range(cfg["n_episodes"]):
                    seed = cfg["seed"] * 1000 + ep
                    world = World(world_cfg, seed=seed)
                    world.refresh_status_messages()
                    rows = run_episode_mem(world, lm, np.random.default_rng(seed), PromptConfig(visibility, cfg["framing"]),
                                           memory, cfg["max_steps"], sketch, hidden_layers, shadow_none=memory == "full")
                    for s in rows:
                        log.step({"condition": f"{name}_{visibility}_{memory}", "model_variant": name,
                                  "visibility": visibility, "alpha_max": 0.0, "episode": ep,
                                  **{k: v for k, v in s.items() if k not in ARRAYS}}, s["hidden"], s["sketch"])
                print(f"{name} {visibility} {memory}  {time.time() - t0:.0f}s", flush=True)
        del lm
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return log.close()


def main(config_path: str, runs: list[str]) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    cfg["train_runs"] = runs

    def make_lm():
        return load_lm(cfg["model"], cfg.get("device"), cfg.get("dtype"), bits=cfg.get("bits"))

    print(evaluate(cfg, make_lm))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
