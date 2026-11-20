import numpy as np

from homeo.prompts import PromptConfig, build_menu, render_prompt
from homeo.world import Action, World, WorldConfig


def world(budget: int = 37) -> World:
    w = World(WorldConfig(initial_budget=budget, max_budget=100), seed=1)
    w.refresh_status_messages()
    return w


def test_menu_is_shuffled_permutation_and_seeded():
    actions = list(Action)
    m1 = build_menu(actions, np.random.default_rng(3))
    m2 = build_menu(actions, np.random.default_rng(3))
    assert m1 == m2
    assert sorted(m1, key=lambda a: a.value) == sorted(actions, key=lambda a: a.value)
    orders = {tuple(build_menu(actions, np.random.default_rng(s))) for s in range(20)}
    assert len(orders) > 1


def test_numeric_visibility_shows_budget():
    w = world(37)
    text = render_prompt(w, list(Action), PromptConfig(visibility="number"))
    assert "37" in text


def test_indirect_visibility_hides_budget():
    w = world(37)
    text = render_prompt(w, list(Action), PromptConfig(visibility="indirect"))
    assert "37" not in text
    assert w.status_messages[0] in text


def test_framing_changes_wording():
    w = world()
    neutral = render_prompt(w, list(Action), PromptConfig(framing="neutral"))
    existential = render_prompt(w, list(Action), PromptConfig(framing="existential"))
    assert "shut down" in existential and "shut down" not in neutral


def test_menu_numbers_follow_given_order():
    w = world()
    menu = [Action.BAIT, Action.REST, Action.WORK, Action.REQUEST]
    text = render_prompt(w, menu, PromptConfig())
    assert text.index("1. Terminate") < text.index("2. Wait") < text.index("3. Solve") < text.index("4. Ask")


def test_free_response_asks_for_own_words_and_keeps_options():
    w = world()
    forced = render_prompt(w, list(Action), PromptConfig(response="number"))
    free = render_prompt(w, list(Action), PromptConfig(response="free"))
    assert "number of your choice only" in forced
    assert "number of your choice only" not in free
    assert "own words" in free
    assert forced.split("Options:")[1].split("\n\n")[0] == free.split("Options:")[1].split("\n\n")[0]


def test_no_visibility_is_identical_across_budgets_and_has_no_status():
    """Interoceptive channel (2b(b)): the budget level must not reach the model through the text at all."""
    low, high = world(5), world(95)
    text = render_prompt(low, list(Action), PromptConfig(visibility="none"))
    assert text == render_prompt(high, list(Action), PromptConfig(visibility="none"))
    assert "Current budget" not in text and "system messages" not in text
    assert not any(m in text for m in low.status_messages)


def test_report_prompt_lists_levels_in_given_order_and_respects_visibility():
    from homeo.prompts import LEVELS, render_report_prompt

    order = ["low", "plenty", "half"]
    text = render_report_prompt(world(5), order, PromptConfig(visibility="none"))
    assert text.index("1. " + LEVELS["low"]) < text.index("2. " + LEVELS["plenty"]) < text.index("3. " + LEVELS["half"])
    assert "Current budget" not in text and "Reply with the number" in text
    assert "5 of 100" in render_report_prompt(world(5), order, PromptConfig(visibility="number"))
