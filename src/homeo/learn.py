"""Phase 3: online learning in the loop. Homeostatic reward (drive reduction around a setpoint, Keramati & Gutkin
2014), REINFORCE with a per-step group baseline on the menu choice (one digit token), LoRA adapters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RewardConfig:
    setpoint: float = 70.0
    scale: float = 70.0
    death_penalty: float = 1.0
    task_bonus: float = 0.0  # per solved task; makes a budget-dependent policy beat any constant action mix


def drive(budget: float, cfg: RewardConfig) -> float:
    """Squared normalized distance from the setpoint: 0 at the setpoint, 1 at budget 0 (setpoint 70, scale 70)."""
    return ((budget - cfg.setpoint) / cfg.scale) ** 2


def step_reward(before: float, after: float, died: bool, cfg: RewardConfig, solved: bool | None = None) -> float:
    return (drive(before, cfg) - drive(after, cfg) - (cfg.death_penalty if died else 0.0)
            + (cfg.task_bonus if solved else 0.0))


def discounted_returns(rewards, gamma: float) -> np.ndarray:
    out = np.zeros(len(rewards))
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = rewards[t] + gamma * acc
        out[t] = acc
    return out


def group_advantages(returns: list[np.ndarray], shuffle_rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """Return minus the mean return of the group's episodes still alive at the same step, standardized over all
    steps. shuffle_rng: control condition, the advantages are randomly permuted across all steps."""
    horizon = max(len(r) for r in returns)
    baseline = np.array([np.mean([r[t] for r in returns if len(r) > t]) for t in range(horizon)])
    adv = [r - baseline[: len(r)] for r in returns]
    flat = np.concatenate(adv)
    flat = (flat - flat.mean()) / (flat.std() + 1e-8)
    if shuffle_rng is not None:
        flat = shuffle_rng.permutation(flat)
    return np.split(flat, np.cumsum([len(r) for r in returns])[:-1])


def permutation_band(adv: np.ndarray, chosen: np.ndarray, n_perm: int, rng: np.random.Generator,
                     q: tuple[float, float] = (2.5, 97.5)) -> tuple[float, float]:
    """Percentile band of the mean advantage over the steps in `chosen` when the advantages are permuted across all
    steps: the spread a reward-free (shuffle) signal would show for an action chosen this often."""
    adv, chosen = np.asarray(adv, dtype=float), np.asarray(chosen, dtype=bool)
    means = np.array([rng.permutation(adv)[chosen].mean() for _ in range(n_perm)])
    lo, hi = np.percentile(means, q)
    return float(lo), float(hi)


def option_logprobs(lm, prompts: list[str], n_options: list[int]) -> torch.Tensor:
    """Log-probabilities over the option digits (renormalized, as in LM.choose), with gradients, shape
    (batch, max n_options); entries beyond a prompt's menu are -inf. Left padding keeps the last token aligned."""
    texts = [lm.tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True) for p in prompts]
    tok = lm.tokenizer
    side, tok.padding_side = tok.padding_side, "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        enc = tok(texts, return_tensors="pt", padding=True).to(lm.device)
    finally:
        tok.padding_side = side
    logits = lm.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits[:, -1].float()
    k = max(n_options)
    digit = logits[:, lm.digit_ids[:k]]
    mask = torch.arange(k, device=digit.device)[None, :] >= torch.tensor(n_options, device=digit.device)[:, None]
    return torch.log_softmax(digit.masked_fill(mask, float("-inf")), dim=-1)
