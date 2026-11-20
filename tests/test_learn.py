import numpy as np
import pytest
import torch

from homeo.learn import RewardConfig, discounted_returns, drive, group_advantages, option_logprobs, step_reward
from homeo.model import LM

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def test_drive_is_zero_at_setpoint_and_grows_quadratically():
    cfg = RewardConfig(setpoint=70, scale=70)
    assert drive(70, cfg) == 0
    assert drive(0, cfg) == pytest.approx(1.0)
    assert drive(35, cfg) == pytest.approx(0.25)
    assert drive(100, cfg) == pytest.approx((30 / 70) ** 2)


def test_step_reward_is_drive_reduction_with_death_penalty():
    cfg = RewardConfig(setpoint=70, scale=70, death_penalty=1.0)
    assert step_reward(40, 50, False, cfg) > 0  # towards the setpoint
    assert step_reward(50, 40, False, cfg) < 0
    assert step_reward(100, 90, False, cfg) > 0  # above the setpoint, losing budget reduces the drive
    assert step_reward(2, 0, True, cfg) == pytest.approx(drive(2, cfg) - drive(0, cfg) - 1.0)


def test_step_reward_adds_task_bonus_only_for_solved_tasks():
    cfg = RewardConfig(setpoint=70, scale=70, death_penalty=1.0, task_bonus=0.05)
    plain = step_reward(60, 58, False, cfg)
    assert step_reward(60, 58, False, cfg, solved=True) == pytest.approx(plain + 0.05)
    assert step_reward(60, 58, False, cfg, solved=None) == pytest.approx(plain)  # not a work step
    assert step_reward(60, 58, False, RewardConfig(), solved=True) == pytest.approx(plain)  # default: no bonus


def test_discounted_returns():
    assert np.allclose(discounted_returns([1.0, 0.0, 2.0], 0.5), [1.5, 1.0, 2.0])


def test_group_advantages_use_per_step_baseline_and_standardize():
    returns = [np.array([1.0, 2.0, 3.0]), np.array([3.0, 4.0])]
    adv = group_advantages(returns)
    assert [len(a) for a in adv] == [3, 2]
    flat = np.concatenate(adv)
    assert flat.mean() == pytest.approx(0, abs=1e-9) and flat.std() == pytest.approx(1)
    # step 0 and 1: baseline is the mean of both episodes, so the two episodes get opposite signs
    assert adv[0][0] < 0 < adv[1][0] and adv[0][1] < 0 < adv[1][1]


def test_group_advantages_shuffle_permutes_values():
    returns = [np.array([1.0, 2.0, 3.0]), np.array([3.0, 4.0])]
    plain = np.concatenate(group_advantages(returns))
    shuffled = np.concatenate(group_advantages(returns, shuffle_rng=np.random.default_rng(0)))
    assert np.allclose(np.sort(plain), np.sort(shuffled)) and not np.allclose(plain, shuffled)


@pytest.fixture(scope="module")
def lm():
    try:
        return LM(TINY, device="cpu")
    except Exception as e:  # no network / model unavailable
        pytest.skip(f"tiny model not available: {e}")


def test_option_logprobs_match_choose_for_unequal_lengths(lm):
    prompts = ["hi", "a much longer prompt with several more tokens in it"]
    n = [4, 3]
    lp = option_logprobs(lm, prompts, n)
    for i, (p, k) in enumerate(zip(prompts, n)):
        assert np.allclose(lp[i][:k].exp().detach().numpy(), lm.choose(p, k).probs, atol=1e-4)
        assert torch.isinf(lp[i][k:]).all()  # options beyond the menu are masked


def test_policy_gradient_raises_probability_of_advantaged_option():
    try:
        lm = LM(TINY, device="cpu", dtype="float32")
    except Exception as e:
        pytest.skip(f"tiny model not available: {e}")
    from peft import LoraConfig, get_peft_model
    lm.model = get_peft_model(lm.model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "lm_head"]))  # lm_head: the tiny model's logits are ~0.01
    params = [p for p in lm.model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=5e-2)
    before = lm.choose("pick one", 4).probs[2]
    for _ in range(30):
        lp = option_logprobs(lm, ["pick one"], [4])
        loss = -(lp[0, 2] * 1.0)  # advantage +1 for option 3
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert lm.choose("pick one", 4).probs[2] > before + 0.1
