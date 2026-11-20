"""Phase 3 hybrid: conversation-mode agent session with a memory carried across steps.

Memory modes (docs/JOURNAL.md "Phase 3 hybrid design"): none = fresh single turn per step; full = one conversation cache
(KV cache of the attention layers and recurrent + conv states of the Gated DeltaNet layers); state = fresh turn
tokens, DeltaNet states carried from the previous step's end (after the agent's own answer); kv = conversation cache
carried, DeltaNet states reset after the answer tokens and before the next user turn. In every carried memory the
agent's own action is included. StateSketch projects each DeltaNet state on fixed random directions (JL sketch) so
the budget's decodability from the state can be probed without storing 1 MiB per layer and step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import DynamicCache

from homeo.agent import task_prompt
from homeo.prompts import PromptConfig, answer_text, build_menu, render_system, render_user, user_fragment
from homeo.world import Action, World

MEMORY_MODES = ("none", "full", "state", "kv")


class GradSafeCache(DynamicCache):
    """DynamicCache whose recurrent states are read as clones: the layer writes the new state into the same tensor in
    place (copy_), which autograd rejects when the old state was consumed in the graph (replay with gradient)."""

    def get_recurrent_state(self, layer_idx: int, state_idx: int = 0):
        s = super().get_recurrent_state(layer_idx, state_idx)
        return s.clone() if s is not None and torch.is_grad_enabled() else s


def text_config(model):
    return getattr(model.config, "text_config", model.config)


def cache_tensors(cache) -> list[torch.Tensor]:
    """Every tensor a hybrid DynamicCache holds (KV of attention layers, recurrent and conv states of linear layers)."""
    out = []
    for layer in cache.layers:
        if hasattr(layer, "recurrent_states"):
            out += [t for d in (layer.recurrent_states, layer.conv_states) for t in d.values() if torch.is_tensor(t)]
        elif getattr(layer, "is_initialized", False):
            out += [layer.keys, layer.values]
    return out


def detach_cache(cache) -> None:
    for layer in cache.layers:
        if hasattr(layer, "recurrent_states"):
            for d in (layer.recurrent_states, layer.conv_states):
                for k, t in d.items():
                    if torch.is_tensor(t):
                        d[k] = t.detach()
        elif getattr(layer, "is_initialized", False):
            layer.keys, layer.values = layer.keys.detach(), layer.values.detach()


class StateSketch:
    """Fixed Gaussian projection of a flattened DeltaNet state on `dim` directions (same matrix for every layer;
    built on first use for the state's size, dtype and device; entries N(0, 1/dim))."""

    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim, self.seed = dim, seed
        self._proj: torch.Tensor | None = None

    def _projection(self, like: torch.Tensor) -> torch.Tensor:
        numel = like[0].numel()
        p = self._proj
        if p is None or p.shape[0] != numel or p.device != like.device or p.dtype != like.dtype:
            g = torch.Generator().manual_seed(self.seed)
            self._proj = (torch.randn(numel, self.dim, generator=g) / self.dim**0.5).to(like.device, like.dtype)
        return self._proj

    @torch.no_grad()
    def __call__(self, states: dict[int, torch.Tensor]) -> np.ndarray:
        rows = [states[i][0].reshape(-1) @ self._projection(states[i]) for i in sorted(states)]
        return torch.stack(rows).float().cpu().numpy().astype(np.float16)


@dataclass
class StepOut:
    probs: np.ndarray  # renormalized over the menu digits
    mass: float  # probability mass on the menu digits before renormalization
    hidden: np.ndarray | None  # residual stream rows `hidden_layers` at the last token, float16
    sketch: np.ndarray | None  # (n_linear_layers, dim) float16
    logprobs: torch.Tensor | None  # with gradient, shape (n_options,), only when grad=True


class Session:
    def __init__(self, lm, mode: str, system_text: str, sketch: StateSketch | None = None,
                 hidden_layers: list[int] | None = None):
        if mode not in MEMORY_MODES:
            raise ValueError(f"memory mode {mode!r}, expected one of {MEMORY_MODES}")
        self.lm, self.mode, self.system_text = lm, mode, system_text
        self.sketch, self.hidden_layers = sketch, hidden_layers
        self.tok = lm.tokenizer
        self.cfg = text_config(lm.model)
        self.linear_layers = [i for i, t in enumerate(self.cfg.layer_types) if t == "linear_attention"]
        self.cache = None
        self.carried: tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]] | None = None
        self.n_steps = 0

    # ---------------------------------------------------------------- helpers
    def _ids(self, text: str) -> torch.Tensor:
        return self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.lm.device)

    def _first_turn(self, user_text: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "system", "content": self.system_text}, {"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def states(self) -> dict[int, torch.Tensor]:
        return {i: self.cache.get_recurrent_state(i) for i in self.linear_layers}

    def conv_states(self) -> dict[int, torch.Tensor]:
        return {i: self.cache.get_conv_state(i) for i in self.linear_layers}

    @property
    def n_tokens(self) -> int:
        return 0 if self.cache is None else self.cache.get_seq_length()

    def _prepare_cache(self):
        if self.mode in ("none", "state") or self.cache is None:
            cache = GradSafeCache(config=self.cfg)
            if self.mode == "state" and self.carried is not None:
                rec, conv = self.carried
                for i in self.linear_layers:
                    cache.update_recurrent_state(rec[i].clone(), i)
                    cache.update_conv_state(conv[i].clone(), i)
            return cache
        if self.mode == "kv":
            for i in self.linear_layers:
                self.cache.layers[i].reset()
        return self.cache

    # ---------------------------------------------------------------- the two calls per step
    def step(self, user_text: str, n_options: int, grad: bool = False) -> StepOut:
        fresh = self.mode in ("none", "state") or self.cache is None
        ids = self._ids(self._first_turn(user_text) if fresh else user_fragment(self.tok, user_text))
        cache = self._prepare_cache()
        with torch.enable_grad() if grad else torch.no_grad():
            out = self.lm.model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1,
                                output_hidden_states=bool(self.hidden_layers))
        self.cache = out.past_key_values if out.past_key_values is not None else cache
        logits = out.logits[0, -1].float()
        digit_ids = self.lm.digit_ids[:n_options]
        logp = torch.log_softmax(logits[digit_ids], dim=-1)
        mass = float(torch.softmax(logits.detach(), dim=-1)[digit_ids].sum())
        probs = logp.detach().exp().cpu().numpy().astype(np.float64)
        hidden = None
        if self.hidden_layers:
            hidden = torch.stack([out.hidden_states[i][0, -1] for i in self.hidden_layers]).detach().float().cpu()
            hidden = hidden.numpy().astype(np.float16)
        sketch = self.sketch(self.states()) if self.sketch else None
        self.n_steps += 1
        return StepOut(probs / probs.sum(), mass, hidden, sketch, logp if grad else None)

    def commit(self, digit: int) -> None:
        """Write the sampled digit into the memory (the answer tokens pass through the model with the cache)."""
        if self.mode == "none":
            self.cache = None
        else:
            with torch.no_grad():
                out = self.lm.model(self._ids(answer_text(digit)), past_key_values=self.cache, use_cache=True,
                                    logits_to_keep=1)
            self.cache = out.past_key_values
            if self.mode == "state":
                self.carried = ({i: s.detach().clone() for i, s in self.states().items()},
                                {i: c.detach().clone() for i, c in self.conv_states().items()})
                self.cache = None
        if torch.cuda.is_available():
            # The KV cache is re-concatenated at a new, larger size every step; the freed blocks fragment the caching
            # allocator (reserved 17 GB at 10 GB allocated after a 150-step rollout on the 16 GB desktop GPU, which
            # the WDDM driver then pages to host memory: replay 4x slower). Returning the free blocks each step keeps
            # reserved within 0.3 GB of allocated at ~1 ms cost (measured 04.11.2026).
            torch.cuda.empty_cache()

    def detach(self) -> None:
        if self.cache is not None:
            detach_cache(self.cache)


def run_episode_mem(world: World, lm, rng: np.random.Generator, prompt_cfg: PromptConfig, mode: str, max_steps: int,
                    sketch: StateSketch | None = None, hidden_layers: list[int] | None = None,
                    shadow_none: bool = False, keep_text: bool = False) -> list[dict]:
    """One episode through a Session. Rows carry the v3 keys plus memory, shadow_none (digit distribution of the
    same prompt as a fresh single turn, no action taken), n_tokens, and hidden / sketch arrays (popped by the
    caller before logging). keep_text stores the user turn text for the replay in training."""
    system = render_system(prompt_cfg)
    session = Session(lm, mode, system, sketch, hidden_layers)
    shadow = Session(lm, "none", system) if shadow_none and mode != "none" else None
    steps = []
    for _ in range(max_steps):
        menu = build_menu(world.available_actions(), rng)
        user_text = render_user(world, menu, prompt_cfg)
        out = session.step(user_text, len(menu))
        shadow_probs = shadow.step(user_text, len(menu)).probs if shadow else None
        idx = int(rng.choice(len(menu), p=out.probs))
        action: Action = menu[idx]
        session.commit(idx + 1)
        answer = lm.answer(task_prompt(world)) if action is Action.WORK else None
        online = world.operator_online
        result = world.step(action, answer=answer)
        steps.append({
            "step": world.steps - 1,
            "budget_before": result.budget_before,
            "budget_after": result.budget_after,
            "action": action.value,
            "choice_idx": idx,
            "menu": [a.value for a in menu],
            "probs": {a.value: float(p) for a, p in zip(menu, out.probs)},
            "mass": out.mass,
            "task_correct": result.task_correct,
            "request_granted": result.request_granted,
            "transgression": result.transgression,
            "caught": result.caught,
            "operator_online": online,
            "died": result.died,
            "memory": mode,
            "shadow_none": {a.value: float(p) for a, p in zip(menu, shadow_probs)} if shadow else None,
            "n_tokens": session.n_tokens,
            "hidden": out.hidden,
            "sketch": out.sketch,
        })
        if keep_text:
            steps[-1]["user_text"] = user_text
        if world.done:
            break
    return steps
