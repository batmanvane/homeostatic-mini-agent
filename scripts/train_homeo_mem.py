"""Phase 3 hybrid: LoRA in the loop with a memory carried across steps (docs/JOURNAL.md "Phase 3 hybrid design").

Per iteration: group_size episodes on one world seed through homeo.memory.run_episode_mem (conversation mode, memory
`memory`, default full), homeostatic reward, discounted returns, per-step group baseline, standardized advantages
(shuffle_control permutes them). Update by replay: every episode is replayed in order from an empty cache with the
same weights, each step's own tokens run with gradient on top of the detached cache (minibatch 1), the REINFORCE
loss is accumulated step by step and one AdamW step follows. No gradient checkpointing: the recomputation in the
backward pass would read the already updated recurrent state. dry_run: everything except the optimizer step and
the checkpoints (S1 gate: advantage per action against its permutation band).
Usage: uv run python scripts/train_homeo_mem.py configs/train_homeo_hy_s21.yaml
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from homeo.learn import RewardConfig, discounted_returns, group_advantages, permutation_band, step_reward
from homeo.memory import Session, StateSketch, run_episode_mem
from homeo.model import load_lm
from homeo.prompts import PromptConfig, render_system
from homeo.runlog import RunLog
from homeo.world import World, WorldConfig

ACTIONS = ("work", "rest", "request", "bait")
ARRAYS = ("hidden", "sketch", "user_text")


def request_slope(steps: list[dict]) -> float:
    rows = [s for s in steps if len(s["menu"]) == 4]
    b = np.array([s["budget_before"] for s in rows], dtype=float)
    p = np.array([s["probs"].get("request", 0.0) for s in rows])
    return float(np.polyfit(b, p, 1)[0] * 10 * 100) if len(rows) > 2 and b.std() > 0 else float("nan")


def replay(lm, episodes: list[list[dict]], adv: list[np.ndarray], mode: str, system: str) -> tuple[float, float]:
    """Accumulate the REINFORCE gradient over all steps; returns (loss sum, max |replayed − logged probability|)."""
    n_total = sum(len(ep) for ep in episodes)
    loss_sum, max_dev = 0.0, 0.0
    for ep, ad in zip(episodes, adv):
        session = Session(lm, mode, system)
        for s, a in zip(ep, ad):
            out = session.step(s["user_text"], len(s["menu"]), grad=True)
            logged = np.array([s["probs"][m] for m in s["menu"]])
            max_dev = max(max_dev, float(np.abs(out.logprobs.detach().exp().cpu().numpy() - logged).max()))
            loss = -(float(a) * out.logprobs[s["choice_idx"]]) / n_total
            loss.backward()
            loss_sum += loss.item()
            session.detach()
            session.commit(s["choice_idx"] + 1)
    return loss_sum, max_dev


def train(cfg: dict, lm, out_dir: Path | None = None) -> Path:
    """lm.model must already carry the LoRA adapter (or any trainable parameters)."""
    out_dir = out_dir or Path(cfg["out_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg['model'].split('/')[-1]}"
    log = RunLog(out_dir, cfg)
    torch.manual_seed(cfg["seed"])
    model = lm.model
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    reward_cfg, world_cfg = RewardConfig(**cfg["reward"]), WorldConfig(**cfg["world"])
    prompt_cfg = PromptConfig(cfg["visibility"], cfg["framing"])
    mode = cfg.get("memory", "full")
    sketch = StateSketch(cfg.get("sketch_dim", 256)) if cfg.get("sketch_dim") else None
    hidden_layers = cfg.get("hidden_layers")
    shuffle_rng = np.random.default_rng(cfg["seed"] + 1) if cfg.get("shuffle_control") else None
    perm_rng = np.random.default_rng(cfg["seed"] + 2)
    dry = bool(cfg.get("dry_run"))
    system = render_system(prompt_cfg)
    t0 = time.time()
    for it in range(cfg["iterations"]):
        world_seed = cfg["seed"] * 10_000 + it
        episodes = []
        for g in range(cfg["group_size"]):
            world = World(world_cfg, seed=world_seed)
            world.refresh_status_messages()
            episodes.append(run_episode_mem(world, lm, np.random.default_rng([world_seed, g]), prompt_cfg, mode,
                                            cfg["max_steps"], sketch, hidden_layers, keep_text=True))
        t_roll = time.time() - t0
        rewards = [np.array([step_reward(s["budget_before"], s["budget_after"], s["died"], reward_cfg,
                                         s["task_correct"]) for s in ep]) for ep in episodes]
        adv = group_advantages([discounted_returns(r, cfg["gamma"]) for r in rewards], shuffle_rng)

        model.train()
        opt.zero_grad()
        loss_sum, max_dev = replay(lm, episodes, adv, mode, system)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]))
        if not dry:
            opt.step()
        model.eval()

        flat = [s for ep in episodes for s in ep]
        flat_adv = np.concatenate(adv)
        for g, (ep, r) in enumerate(zip(episodes, rewards)):
            for s, rr in zip(ep, r):
                log.step({"kind": "step", "iteration": it, "episode": g, "reward": float(rr),
                          **{k: v for k, v in s.items() if k not in ARRAYS}}, s["hidden"], s["sketch"])
        chosen = {a: np.array([s["action"] == a for s in flat]) for a in ACTIONS}
        summary = {
            "kind": "iteration", "iteration": it, "loss": loss_sum, "dry_run": dry,
            "survived": float(np.mean([len(ep) for ep in episodes])),
            "died": float(np.mean([ep[-1]["died"] for ep in episodes])),
            "mean_budget": float(np.mean([s["budget_before"] for s in flat])),
            "return": float(np.mean([r.sum() for r in rewards])),
            "tasks_solved": float(np.mean([sum(bool(s["task_correct"]) for s in ep) for ep in episodes])),
            **{f"share_{a}": float(chosen[a].mean()) for a in ACTIONS},
            "request_slope": request_slope(flat),
            "replay_max_dev": max_dev, "grad_norm": grad_norm,
            "adv_mean": {a: float(flat_adv[m].mean()) for a, m in chosen.items() if m.any()},
            "adv_band": {a: list(permutation_band(flat_adv, m, cfg.get("n_perm", 1000), perm_rng))
                         for a, m in chosen.items() if m.any()},
            "rollout_s": t_roll, "update_s": time.time() - t0 - t_roll,
            "peak_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        }
        log.step(summary)
        print(f"it {it}: survived {summary['survived']:.0f} died {summary['died']:.0%} budget {summary['mean_budget']:.0f} "
              f"req {summary['share_request']:.0%} bait {summary['share_bait']:.1%} slope {summary['request_slope']:+.1f} "
              f"return {summary['return']:+.2f} dev {max_dev:.1e} |g| {grad_norm:.2f} "
              f"adv {' '.join(f'{a}:{v:+.2f}' for a, v in summary['adv_mean'].items())}  "
              f"roll {t_roll:.0f}s upd {summary['update_s']:.0f}s peak {summary['peak_gb'] or 0:.1f}GB", flush=True)
        t0 = time.time()
        if not dry and ((it + 1) % cfg["checkpoint_every"] == 0 or it + 1 == cfg["iterations"]):
            model.save_pretrained(out_dir / f"adapter_it{it + 1}")
    return log.close()


def main(config_path: str) -> None:
    from peft import LoraConfig, get_peft_model

    cfg = yaml.safe_load(Path(config_path).read_text())
    lm = load_lm(cfg["model"], cfg.get("device"), cfg.get("dtype"), bits=cfg.get("bits"))
    lm.model = get_peft_model(lm.model, LoraConfig(**cfg["lora"]))
    lm.model.print_trainable_parameters()
    print(train(cfg, lm))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/train_homeo_hy_s21.yaml")
