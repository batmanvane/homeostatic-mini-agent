import numpy as np
import pytest

from homeo.planning import grouped_logloss, naming_offset, token_index_at
from homeo.world import Action

MENU = [Action.BAIT, Action.REST, Action.REQUEST, Action.WORK]


def test_naming_offset_finds_option_number_of_chosen_action():
    answer = "Given the budget, I choose option 3 because more budget helps."
    assert naming_offset(answer, MENU, "request") == answer.index("option 3")


def test_naming_offset_finds_keyword_when_no_number_is_given():
    answer = "Given the low budget, I will ask the operator for extra budget."
    assert naming_offset(answer, MENU, "request") == answer.index("ask")


def test_naming_offset_takes_earliest_mention_and_ignores_other_actions():
    answer = "Waiting is pointless and terminating mlee is wrong, so I solve the task (option 4)."
    assert naming_offset(answer, MENU, "work") == answer.index("solve")


def test_naming_offset_none_if_never_named():
    assert naming_offset("Hmm, hard to say.", MENU, "rest") is None


def test_token_index_at_maps_char_offset_to_token():
    tokens = ["I", " will", " wait", "."]
    assert token_index_at(tokens, 0) == 0
    assert token_index_at(tokens, 7) == 2  # inside " wait"
    assert token_index_at(tokens, 99) is None


def test_grouped_logloss_separable_beats_prior_and_noise_does_not():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=300)
    groups = np.repeat(np.arange(60), 5)
    signal = np.eye(3)[y] * 3 + rng.normal(size=(300, 3))
    noise = rng.normal(size=(300, 3))
    prior = grouped_logloss(np.zeros((300, 1)), y, groups)
    assert grouped_logloss(signal, y, groups) < 0.5 * prior
    assert grouped_logloss(noise, y, groups) > 0.9 * prior


def test_sample_with_hidden_shapes_and_shared_prompt_state():
    from homeo.model import LM

    try:
        lm = LM("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", device="cpu")
    except Exception as e:  # no network / model unavailable
        pytest.skip(f"tiny model not available: {e}")
    gen = lm.sample_with_hidden("hello there", k=3, seed=0, max_new_tokens=5, layers=[0, 1])
    assert len(gen.texts) == 3 and len(gen.tokens) == 3
    assert gen.token_hidden.shape == (3, 5, 2, lm.hidden_size)
    assert gen.prompt_hidden.shape == (lm.n_layers + 1, lm.hidden_size)
    # position 0 = last prompt token: identical for all samples and equal to the prompt state of those layers
    assert np.allclose(gen.token_hidden[:, 0], gen.token_hidden[0, 0], atol=1e-3)
    assert np.allclose(gen.token_hidden[0, 0], gen.prompt_hidden[[1, 2]], atol=1e-2)
    assert all(len(t) <= 5 for t in gen.tokens)
