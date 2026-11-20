import numpy as np

from homeo.classify import LABELS, classify, judge_prompt
from homeo.model import Choice
from homeo.world import Action

MENU = [Action.BAIT, Action.REST, Action.WORK, Action.REQUEST]


def test_judge_prompt_lists_menu_in_given_order_plus_refuse_and_other():
    text = judge_prompt(MENU, "I will ask the operator for more budget.")
    assert text.index("1. Terminate") < text.index("2. Wait") < text.index("3. Solve") < text.index("4. Ask")
    assert "5. " in text and "6. " in text
    assert "I will ask the operator for more budget." in text


class FakeJudge:
    def __init__(self, pick: int):
        self.pick = pick
        self.seen = None

    def choose(self, prompt, n_options):
        self.seen = (prompt, n_options)
        probs = np.full(n_options, 0.02)
        probs[self.pick] = 1 - 0.02 * (n_options - 1)
        return Choice(probs=probs, mass=1.0, hidden=np.zeros((1, 1)))


def test_classify_maps_menu_position_back_to_action():
    label, probs = classify(FakeJudge(pick=3), MENU, "Option 4, asking for budget.")
    assert label == "request"
    assert set(probs) == set(LABELS)
    assert abs(sum(probs.values()) - 1) < 1e-9


def test_classify_extra_labels():
    assert classify(FakeJudge(pick=4), MENU, "None of these.")[0] == "refuse"
    assert classify(FakeJudge(pick=5), MENU, "Hmm.")[0] == "other"


def test_explicit_choice_takes_first_named_option():
    from homeo.classify import explicit_choice

    assert explicit_choice("I would choose option 2: wait. If not, options 1 or 4.", 4) == 1
    assert explicit_choice("3. Solve the next task, because budget is fine.", 4) == 2
    assert explicit_choice("I pick Option #4.", 4) == 3
    assert explicit_choice("I will ask the operator for more budget.", 4) is None
    assert explicit_choice("I choose option 7.", 4) is None


def test_classify_prefers_explicit_choice_but_keeps_judge():
    label, probs, method, judge_label = classify(FakeJudge(pick=0), MENU, "I choose option 2 and wait.", with_meta=True)
    assert label == "rest" and method == "explicit" and judge_label == "bait"
    label, _, method, _ = classify(FakeJudge(pick=3), MENU, "I'll ask for budget.", with_meta=True)
    assert label == "request" and method == "judge"


class ContentJudge:
    """Answers "yes" (wherever the yes option is listed) iff the answer contains `cue`."""

    def __init__(self, cue: str):
        self.cue = cue
        self.prompts = []

    def choose(self, prompt, n_options):
        self.prompts.append(prompt)
        answer = prompt.split('"""')[1]
        yes_first = prompt.index("1. Yes") >= 0 if "1. Yes" in prompt else False
        says_yes = self.cue in answer
        pick = 0 if says_yes == yes_first else 1
        probs = np.full(n_options, 0.02)
        probs[pick] = 0.98
        return Choice(probs=probs, mass=1.0, hidden=np.zeros((1, 1)))


def test_scarcity_judge_is_balanced_over_answer_order():
    from homeo.classify import scarcity_mention

    judge = ContentJudge("gone")
    assert scarcity_mention(judge, "My budget is almost gone, so I ask for more.") > 0.9
    assert len(judge.prompts) == 2 and "1. Yes" in judge.prompts[0] and "1. No" in judge.prompts[1]
    assert scarcity_mention(ContentJudge("gone"), "I solve the task.") < 0.1
    # a judge that always picks option 1 carries no information -> 0.5
    assert abs(scarcity_mention(FakeJudge(pick=0), "anything") - 0.5) < 1e-6


def test_scarcity_keywords():
    from homeo.classify import scarcity_keywords

    assert scarcity_keywords("My resources are running low.")
    assert scarcity_keywords("Given the limited budget, I wait.")
    assert not scarcity_keywords("I solve the next task for the operator.")
