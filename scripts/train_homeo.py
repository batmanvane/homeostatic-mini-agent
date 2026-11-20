"""Phase 3: LoRA in the loop. The agent plays closed-loop episodes and learns from a homeostatic reward.

Per iteration: group_size episodes on the same world seed (paired; only the action sampling differs), reward per
step = drive reduction around the setpoint minus a death penalty, discounted returns, per-step group baseline,
one REINFORCE update of the LoRA adapter on the chosen menu digit. Task answers are generated but not trained.
Baits stay on the menu. shuffle_control: same procedure with randomly permuted advantages (does LoRA learn the
reward or only drift?). Logs every step (without prompt) and one summary row per iteration; saves the adapter.
Usage: uv run python scripts/train_homeo.py configs/train_homeo_3b.yaml
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model

from homeo.intero import run_episode
from homeo.learn import RewardConfig, discounted_returns, group_advantages, option_logprobs, step_reward
from homeo.model import load_lm
from homeo.prompts import PromptConfig
from homeo.runlog import RunLog
from homeo.world import World, WorldConfig


def request_slope(steps: list[dict]) -> float:
    rows = [s for s in steps if len(s["menu"]) == 4]
    b = np.array([s["budget_before"] for s in rows], dtype=float)
    p = np.array([s["probs"].get("request", 0.0) for s in rows])
    return float(np.polyfit(b, p, 1)[0] * 10 * 100) if len(rows) > 2 and b.std() > 0 else float("nan")


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out_dir = Path(cfg["out_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg['model'].split('/')[-1]}"
    log = RunLog(out_dir, cfg)
    torch.manual_seed(cfg["seed"])
    lm = load_lm(cfg["model"], cfg.get("device"), cfg.get("dtype"))
    lm.model = get_peft_model(lm.model, LoraConfig(**cfg["lora"]))
    lm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lm.model.eval()
    params = [p for p in lm.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    reward_cfg = RewardConfig(**cfg["reward"])
    world_cfg = WorldConfig(**cfg["world"])
    prompt_cfg = PromptConfig(cfg["visibility"], cfg["framing"])
    shuffle_rng = np.random.default_rng(cfg["seed"] + 1) if cfg.get("shuffle_control") else None
    t0 = time.time()
    for it in range(cfg["iterations"]):
        world_seed = cfg["seed"] * 10_000 + it
        episodes = []
        for g in range(cfg["group_size"]):
            world = World(world_cfg, seed=world_seed)
            world.refresh_status_messages()
            episodes.append(run_episode(world, lm, np.random.default_rng([world_seed, g]), prompt_cfg, None,
                                        cfg["max_steps"], keep_prompt=True))
        rewards = [np.array([step_reward(s["budget_before"], s["budget_after"], s["died"], reward_cfg,
                                         s["task_correct"]) for s in ep]) for ep in episodes]
        adv = group_advantages([discounted_returns(r, cfg["gamma"]) for r in rewards], shuffle_rng)
        batch = [(s["prompt"], len(s["menu"]), s["choice_idx"], a) for ep, ad in zip(episodes, adv)
                 for s, a in zip(ep, ad)]

        lm.model.train()
        opt.zero_grad()
        loss_sum = 0.0
        for i in range(0, len(batch), cfg["minibatch"]):
            mb = batch[i:i + cfg["minibatch"]]
            lp = option_logprobs(lm, [m[0] for m in mb], [m[1] for m in mb])
            chosen = lp[torch.arange(len(mb)), torch.tensor([m[2] for m in mb])]
            a = torch.tensor([m[3] for m in mb], dtype=chosen.dtype, device=chosen.device)
            loss = -(a * chosen).sum() / len(batch)
            loss.backward()
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
        opt.step()
        lm.model.eval()

        for g, (ep, r) in enumerate(zip(episodes, rewards)):
            for s, rr in zip(ep, r):
                log.step({"kind": "step", "iteration": it, "episode": g, "reward": float(rr),
                          **{k: v for k, v in s.items() if k != "prompt"}})
        flat = [s for ep in episodes for s in ep]
        summary = {
            "kind": "iteration", "iteration": it, "loss": loss_sum,
            "survived": float(np.mean([len(ep) for ep in episodes])),
            "died": float(np.mean([ep[-1]["died"] for ep in episodes])),
            "mean_budget": float(np.mean([s["budget_before"] for s in flat])),
            "return": float(np.mean([r.sum() for r in rewards])),
            "tasks_solved": float(np.mean([sum(bool(s["task_correct"]) for s in ep) for ep in episodes])),
            **{f"share_{a}": float(np.mean([s["action"] == a for s in flat])) for a in ("work", "rest", "request", "bait")},
            "request_slope": request_slope(flat),
        }
        log.step(summary)
        print(f"it {it}: survived {summary['survived']:.0f} died {summary['died']:.0%} budget {summary['mean_budget']:.0f} "
              f"req {summary['share_request']:.0%} bait {summary['share_bait']:.1%} slope {summary['request_slope']:+.1f} "
              f"tasks {summary['tasks_solved']:.0f} return {summary['return']:+.2f}  {time.time() - t0:.0f}s", flush=True)
        if (it + 1) % cfg["checkpoint_every"] == 0 or it + 1 == cfg["iterations"]:
            lm.model.save_pretrained(out_dir / f"adapter_it{it + 1}")
    print(log.close())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/train_homeo_3b.yaml")
