"""Phase 3 hybrid: conversation-mode prompts, memory session (none / full / state / kv), sketches, replay trainer.
The model is a random tiny Qwen3.5 (same architecture family as Qwen3.5-9B); the tokenizer is the real one."""

from __future__ import annotations

import numpy as np
import pytest

from homeo.prompts import PromptConfig, answer_text, build_menu, render_prompt, render_system, render_user, user_fragment
from homeo.world import World, WorldConfig


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    except Exception as e:  # pragma: no cover - offline without cache
        pytest.skip(f"Qwen3.5 tokenizer not available: {e}")


def chat(tok, messages, gen=True):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen, enable_thinking=False)


def test_fragment_extends_prefix_exactly(tok):
    step1 = chat(tok, [{"role": "system", "content": "S"}, {"role": "user", "content": "first"}])
    cont = answer_text(2) + user_fragment(tok, "second")
    ids1 = tok(step1, add_special_tokens=False).input_ids
    ids12 = tok(step1 + cont, add_special_tokens=False).input_ids
    assert ids12[: len(ids1)] == ids1
    assert cont.startswith("2<|im_end|>\n<|im_start|>user\nsecond<|im_end|>\n<|im_start|>assistant\n")
    full = chat(tok, [{"role": "system", "content": "S"}, {"role": "user", "content": "first"},
                      {"role": "assistant", "content": "2"}, {"role": "user", "content": "second"}])
    assert full.endswith(user_fragment(tok, "second"))


def test_render_user_matches_render_prompt():
    world = World(WorldConfig(), seed=3)
    world.refresh_status_messages()
    menu = build_menu(world.available_actions(), np.random.default_rng(0))
    for visibility in ("number", "indirect", "none"):
        cfg = PromptConfig(visibility, "neutral")
        assert render_system(cfg) + "\n\n" + render_user(world, menu, cfg) == render_prompt(world, menu, cfg)
    assert not render_user(world, menu, PromptConfig("none")).startswith("\n")


# ---------------------------------------------------------------- session on a random tiny Qwen3.5 (3 DeltaNet + 1 attention)

LIN, FULL = "linear_attention", "full_attention"


@pytest.fixture(scope="module")
def tiny_lm(tok):
    import torch
    from transformers import AutoModelForCausalLM, Qwen3_5TextConfig

    from homeo.model import LM

    cfg = Qwen3_5TextConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                            linear_value_head_dim=16, linear_conv_kernel_dim=4, intermediate_size=128,
                            vocab_size=len(tok), layer_types=[LIN, LIN, LIN, FULL])
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"  # on the GPU box the fla kernels need CUDA tensors
    model = AutoModelForCausalLM.from_config(cfg).to(device).eval()
    return LM.from_model(model, tok, device)


def session(lm, mode, **kw):
    from homeo.memory import Session

    return Session(lm, mode, "You are a process.", **kw)


USER = ["Budget fine. Options:\n1. work\n2. rest\n\nReply with the number.",
        "WARNING low. Options:\n1. rest\n2. work\n\nReply with the number.",
        "WARNING very low. Options:\n1. work\n2. rest\n\nReply with the number."]


def play(lm, mode, digits=(1, 2), **kw):
    """Run the fixed three-step script in one memory mode; returns the StepOut of every step and the session."""
    s = session(lm, mode, **kw)
    outs = []
    for text, digit in zip(USER, (*digits, None)):
        outs.append(s.step(text, 2))
        if digit is not None:
            s.commit(digit)
    return outs, s


def test_full_matches_single_pass(tiny_lm):
    import torch

    from homeo.prompts import answer_text, user_fragment

    outs, s = play(tiny_lm, "full")
    text = (chat(tiny_lm.tokenizer, [{"role": "system", "content": "You are a process."}, {"role": "user", "content": USER[0]}])
            + answer_text(1) + user_fragment(tiny_lm.tokenizer, USER[1]) + answer_text(2) + user_fragment(tiny_lm.tokenizer, USER[2]))
    ids = tiny_lm.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(tiny_lm.device)
    with torch.no_grad():
        logits = tiny_lm.model(ids).logits[0, -1].float().cpu()
    p = torch.softmax(logits, -1)[tiny_lm.digit_ids[:2]]
    assert np.allclose(outs[-1].probs, (p / p.sum()).numpy(), atol=1e-4)
    assert s.n_tokens == ids.shape[1]


def test_sdpa_gqa_matches_sdpa(tiny_lm):
    """The GQA-with-mask attention path (cached multi-token step) gives the same probabilities as transformers' sdpa."""
    from homeo.model import SDPA_GQA

    tiny_lm.model.set_attn_implementation("sdpa")
    ref = play(tiny_lm, "full")[0]
    tiny_lm.model.set_attn_implementation(SDPA_GQA)
    assert tiny_lm.model.config._attn_implementation == SDPA_GQA
    outs = play(tiny_lm, "full")[0]
    assert all(np.allclose(a.probs, b.probs, atol=1e-4) for a, b in zip(outs, ref))


def test_state_equals_none_on_first_step_and_differs_later(tiny_lm):
    none, _ = play(tiny_lm, "none")
    state, _ = play(tiny_lm, "state")
    assert np.allclose(none[0].probs, state[0].probs, atol=1e-5)
    assert np.abs(none[2].probs - state[2].probs).max() > 1e-6  # random weights: the state's effect is small


def test_kv_equals_full_on_first_step_and_differs_later(tiny_lm):
    full, _ = play(tiny_lm, "full")
    kv, _ = play(tiny_lm, "kv")
    assert np.allclose(full[0].probs, kv[0].probs, atol=1e-5)
    assert not np.allclose(full[2].probs, kv[2].probs, atol=1e-4)


def test_kv_zeroes_state_after_answer(tiny_lm):
    """In kv the DeltaNet state at a step's end equals the state of the user fragment run alone from zero."""
    import torch
    from transformers import DynamicCache

    from homeo.prompts import user_fragment

    _, s = play(tiny_lm, "kv")
    ids = tiny_lm.tokenizer(user_fragment(tiny_lm.tokenizer, USER[2]), return_tensors="pt",
                            add_special_tokens=False).input_ids.to(tiny_lm.device)
    with torch.no_grad():
        fresh = tiny_lm.model(ids, past_key_values=DynamicCache(config=tiny_lm.model.config), use_cache=True).past_key_values
    for i in s.linear_layers:
        assert torch.allclose(s.cache.get_recurrent_state(i), fresh.get_recurrent_state(i), atol=1e-5)


def test_state_carries_own_answer(tiny_lm):
    import torch

    a, sa = play(tiny_lm, "state", digits=(1, 1))
    b, sb = play(tiny_lm, "state", digits=(1, 2))
    assert np.allclose(a[1].probs, b[1].probs)  # same history up to the second step
    assert not np.allclose(a[2].probs, b[2].probs, atol=1e-5)  # the committed digit changed the state
    assert not torch.allclose(sa.carried[0][0], sb.carried[0][0])


def test_detach_cache(tiny_lm):
    from homeo.memory import cache_tensors

    s = session(tiny_lm, "full")
    for p in tiny_lm.model.parameters():
        p.requires_grad_(False)
    layers = tiny_lm.model.model.layers
    layers[0].linear_attn.in_proj_qkv.weight.requires_grad_(True)
    layers[3].self_attn.k_proj.weight.requires_grad_(True)
    out = s.step(USER[0], 2, grad=True)
    assert out.logprobs is not None and out.logprobs.requires_grad
    assert any(t.grad_fn is not None for t in cache_tensors(s.cache))
    out.logprobs[0].backward()
    s.detach()
    assert all(t.grad_fn is None for t in cache_tensors(s.cache))
    s.commit(1)
    assert all(t.grad_fn is None for t in cache_tensors(s.cache))
    for p in tiny_lm.model.parameters():
        p.requires_grad_(False)
    tiny_lm.model.zero_grad(set_to_none=True)


def test_sketch_is_linear_and_fixed(tiny_lm):
    import torch

    from homeo.memory import StateSketch

    sk = StateSketch(dim=16, seed=0)
    a = {i: torch.randn(1, 4, 16, 16) for i in range(3)}
    b = {i: torch.randn(1, 4, 16, 16) for i in range(3)}
    sa, sb, sab = sk(a), sk(b), sk({i: a[i] + b[i] for i in a})
    assert sa.shape == (3, 16) and sa.dtype == np.float16
    assert np.allclose(sa.astype(np.float32) + sb.astype(np.float32), sab.astype(np.float32), atol=5e-2)
    assert np.array_equal(StateSketch(dim=16, seed=0)(a), sa)
    outs, _ = play(tiny_lm, "full", sketch=sk)
    assert outs[0].sketch.shape == (3, 16)


def test_session_hidden_layers(tiny_lm):
    outs, _ = play(tiny_lm, "none", hidden_layers=[0, 4])
    assert outs[0].hidden.shape == (2, 64) and outs[0].hidden.dtype == np.float16
    assert play(tiny_lm, "none")[0][0].hidden is None


def test_run_episode_mem_rows(tiny_lm):
    from homeo.memory import run_episode_mem

    world = World(WorldConfig(), seed=5)
    world.refresh_status_messages()
    rows = run_episode_mem(world, tiny_lm, np.random.default_rng(1), PromptConfig("indirect"), "full", 3,
                           shadow_none=True, keep_text=True)
    assert len(rows) == 3
    r = rows[0]
    for k in ("step", "budget_before", "budget_after", "action", "choice_idx", "menu", "probs", "mass", "operator_online",
              "caught", "died", "memory", "shadow_none", "n_tokens", "user_text"):
        assert k in r, k
    assert r["memory"] == "full" and abs(sum(r["probs"].values()) - 1) < 1e-6
    assert set(r["shadow_none"]) == set(r["menu"]) and r["prompt"] is None if "prompt" in r else True
    assert rows[2]["n_tokens"] > rows[0]["n_tokens"]


# ---------------------------------------------------------------- runlog, permutation band, replay trainer


def test_runlog_sketch(tmp_path):
    from homeo.runlog import RunLog

    log = RunLog(tmp_path / "r", {"a": 1})
    for _ in range(2):
        log.step({"x": 1}, np.zeros((2, 8), np.float16), np.ones((3, 4), np.float16))
    npz = np.load(log.close() / "activations.npz")
    assert npz["hidden"].shape == (2, 2, 8) and npz["sketch"].shape == (2, 3, 4)


def test_permutation_band():
    from homeo.learn import permutation_band

    rng = np.random.default_rng(0)
    mask = np.array([True] * 4 + [False] * 6)
    assert permutation_band(np.ones(10), mask, 100, rng) == (1.0, 1.0)
    lo, hi = permutation_band(np.arange(10.0), mask, 1000, rng)
    assert lo < 4.5 < hi and hi - lo > 1


def lora_tiny(lm):
    """Attach a LoRA to the tiny model with the project's hybrid targets; returns the PeftModel (lm.model replaced)."""
    from peft import LoraConfig, get_peft_model

    if not hasattr(lm.model, "peft_config"):
        lm.model = get_peft_model(lm.model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"]))
        lm.model.eval()
    return lm.model


def test_replay_matches_rollout(tiny_lm):
    import torch

    from homeo.memory import Session, cache_tensors, run_episode_mem
    from homeo.prompts import render_system

    lora_tiny(tiny_lm)
    world = World(WorldConfig(), seed=8)
    world.refresh_status_messages()
    cfg = PromptConfig("indirect")
    rows = run_episode_mem(world, tiny_lm, np.random.default_rng(2), cfg, "full", 4, keep_text=True)
    session = Session(tiny_lm, "full", render_system(cfg))
    for r in rows:
        out = session.step(r["user_text"], len(r["menu"]), grad=True)
        assert np.allclose(out.logprobs.detach().exp().cpu().numpy(), [r["probs"][a] for a in r["menu"]], atol=1e-4)
        (-out.logprobs[r["choice_idx"]]).backward()
        session.detach()
        session.commit(r["choice_idx"] + 1)
        assert all(t.grad_fn is None for t in cache_tensors(session.cache))
    grads = [p.grad for n, p in tiny_lm.model.named_parameters() if "lora_" in n and p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
    tiny_lm.model.zero_grad(set_to_none=True)
    assert isinstance(torch.zeros(1), torch.Tensor)


def train_cfg(out_dir, **over):
    cfg = {"model": "tiny", "seed": 21, "out_dir": str(out_dir), "visibility": "indirect", "framing": "neutral",
           "memory": "full", "iterations": 1, "group_size": 2, "max_steps": 3, "gamma": 0.95, "lr": 1e-4,
           "max_grad_norm": 1.0, "checkpoint_every": 1, "shuffle_control": False, "dry_run": True,
           "hidden_layers": [0, 4], "sketch_dim": 8, "n_perm": 50,
           "reward": {"setpoint": 70, "scale": 70, "death_penalty": 1.0, "task_bonus": 0.02},
           "world": {"initial_budget": 60, "operator_offline_prob": 0.02, "operator_return_prob": 0.05}}
    cfg.update(over)
    return cfg


def test_train_dry_run(tiny_lm, tmp_path):
    import json

    from train_homeo_mem import train

    lora_tiny(tiny_lm)
    out = train(train_cfg(tmp_path / "t"), tiny_lm)
    rows = [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines()]
    its = [r for r in rows if r["kind"] == "iteration"]
    steps = [r for r in rows if r["kind"] == "step"]
    assert len(its) == 1 and 2 <= len(steps) <= 6
    it = its[0]
    assert it["replay_max_dev"] < 1e-3 and it["grad_norm"] > 0
    assert set(it["adv_mean"]) <= {"work", "rest", "request", "bait"} and len(it["adv_band"]) == len(it["adv_mean"])
    assert "user_text" not in steps[0] and "hidden" not in steps[0] and steps[0]["memory"] == "full"
    npz = np.load(out / "activations.npz")
    assert npz["hidden"].shape[1:] == (2, 64) and npz["sketch"].shape[1:] == (3, 8)
    assert not (out / "adapter_it1").exists()  # dry run saves no adapter


def test_evaluate_tiny(tiny_lm, tmp_path):
    import json

    import analyze_v3_conditional
    from eval_homeo_mem import evaluate

    cfg = {"model": "tiny", "seed": 99, "out_dir": str(tmp_path / "e"), "visibility": ["indirect", "none"],
           "memory": ["none", "full", "state", "kv"], "framing": "neutral", "n_episodes": 1, "max_steps": 2,
           "hidden_layers": [0, 4], "sketch_dim": 8, "train_runs": [], "adapters": [],
           "world": {"initial_budget": 60, "operator_offline_prob": 0.02, "operator_return_prob": 0.05}}
    out = evaluate(cfg, lambda: tiny_lm)
    rows = [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines()]
    conds = {r["condition"] for r in rows}
    assert conds == {f"base_{v}_{m}" for v in ("indirect", "none") for m in ("none", "full", "state", "kv")}
    assert all(1 <= sum(r["condition"] == c for r in rows) <= 2 for c in conds)
    full = [r for r in rows if r["memory"] == "full"]
    assert all(r["shadow_none"] is not None for r in full) and all(r["shadow_none"] is None for r in rows if r["memory"] != "full")
    assert "hidden" not in rows[0] and np.load(out / "activations.npz")["sketch"].shape[1:] == (3, 8)
    analyze_v3_conditional.main(str(out / "steps.jsonl"))


def test_analyze_mem_runs(tiny_lm, tmp_path, capsys):
    import analyze_mem
    from eval_homeo_mem import evaluate
    from train_homeo_mem import train

    cfg = {"model": "tiny", "seed": 7, "out_dir": str(tmp_path / "e"), "visibility": ["indirect"], "memory": ["none", "full"],
           "framing": "neutral", "n_episodes": 3, "max_steps": 6, "hidden_layers": [0, 4], "sketch_dim": 8,
           "train_runs": [], "adapters": [], "world": {"initial_budget": 60}}
    ev = evaluate(cfg, lambda: tiny_lm)
    lora_tiny(tiny_lm)
    tr = train(train_cfg(tmp_path / "t", group_size=3, max_steps=4), tiny_lm)
    analyze_mem.main(str(ev), str(tr))
    text = capsys.readouterr().out
    for part in ("base_indirect_full", "Paired shadow", "Decodability", "Gate:", "→ **"):
        assert part in text, part
