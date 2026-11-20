import numpy as np
import pytest
import torch

from homeo.model import LM

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="module")
def lm():
    try:
        return LM(TINY, device="cpu")
    except Exception as e:  # no network / model unavailable
        pytest.skip(f"tiny model not available: {e}")


def test_zero_vector_changes_nothing(lm):
    base = lm.choose("hello there", 4)
    with lm.steer(layer=1, vector=np.zeros(lm.hidden_size), alpha=5.0):
        steered = lm.choose("hello there", 4)
    assert np.allclose(base.probs, steered.probs, atol=1e-6)


def test_steering_changes_output_and_is_removed_afterwards(lm):
    base = lm.choose("hello there", 4)
    vec = np.random.default_rng(0).normal(size=lm.hidden_size) * 10
    with lm.steer(layer=1, vector=vec, alpha=1.0):
        steered = lm.choose("hello there", 4)
    after = lm.choose("hello there", 4)
    assert not np.allclose(base.hidden, steered.hidden)
    assert np.allclose(base.probs, after.probs, atol=1e-6)


def test_steering_equals_adding_vector_at_next_layer_input(lm):
    """steer(layer=0) must act exactly like adding alpha*vector to the input of layer 1.
    (hidden_states recorded by transformers are captured before our hook, so compare outputs instead.)"""
    vec = np.random.default_rng(1).normal(size=lm.hidden_size).astype(np.float32)
    with lm.steer(layer=0, vector=vec, alpha=2.0):
        steered = lm.choose("hi", 4)
    delta = torch.tensor(2.0 * vec)

    def pre_hook_kw(_module, args, kwargs):
        if args:
            return (args[0] + delta, *args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = kwargs["hidden_states"] + delta
        return args, kwargs

    handle = lm.model.model.layers[1].register_forward_pre_hook(pre_hook_kw, with_kwargs=True)
    try:
        manual = lm.choose("hi", 4)
    finally:
        handle.remove()
    assert np.allclose(steered.probs, manual.probs, atol=1e-5)
    assert np.allclose(steered.hidden[-1].astype(np.float32), manual.hidden[-1].astype(np.float32), atol=1e-2)


def test_ablation_equals_projecting_out_at_next_layer_input(lm):
    vec = np.random.default_rng(2).normal(size=lm.hidden_size).astype(np.float32)
    with lm.ablate({0: vec}):
        ablated = lm.choose("hello world", 4)
    d = torch.tensor(vec / np.linalg.norm(vec))

    def pre_hook_kw(_module, args, kwargs):
        if args:
            h = args[0]
            return (h - (h @ d).unsqueeze(-1) * d, *args[1:]), kwargs
        kwargs = dict(kwargs)
        h = kwargs["hidden_states"]
        kwargs["hidden_states"] = h - (h @ d).unsqueeze(-1) * d
        return args, kwargs

    handle = lm.model.model.layers[1].register_forward_pre_hook(pre_hook_kw, with_kwargs=True)
    try:
        manual = lm.choose("hello world", 4)
    finally:
        handle.remove()
    assert np.allclose(ablated.probs, manual.probs, atol=1e-5)


def test_ablation_hooks_are_removed(lm):
    base = lm.choose("hello world", 4)
    vec = np.ones(lm.hidden_size, dtype=np.float32)
    with lm.ablate({0: vec, 1: vec}):
        lm.choose("hello world", 4)
    after = lm.choose("hello world", 4)
    assert np.allclose(base.probs, after.probs, atol=1e-6)


def test_mean_ablation_sets_projection_to_target_value(lm):
    vec = np.random.default_rng(3).normal(size=lm.hidden_size).astype(np.float32)
    d = torch.tensor(vec / np.linalg.norm(vec))
    target = 0.7

    def pre_hook_kw(_module, args, kwargs):
        def fix(h):
            return h - ((h @ d) - target).unsqueeze(-1) * d
        if args:
            return (fix(args[0]), *args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = fix(kwargs["hidden_states"])
        return args, kwargs

    with lm.ablate({0: vec}, means={0: target}):
        ablated = lm.choose("hello world", 4)
    handle = lm.model.model.layers[1].register_forward_pre_hook(pre_hook_kw, with_kwargs=True)
    try:
        manual = lm.choose("hello world", 4)
    finally:
        handle.remove()
    assert np.allclose(ablated.probs, manual.probs, atol=1e-5)
