import json

import pytest

from rsipo.execution import Reliability, choose_evidence, episode_utility, execution_signals, skill_token_mask, split_rollouts


class CharTokenizer:
    """One token per character, so token spans equal character spans."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def test_split_is_deterministic_disjoint_and_complete():
    evidence, control = split_rollouts(0, 3, "/games/a/game.tw-pddl", 8, 4)
    assert (evidence, control) == split_rollouts(0, 3, "/games/a/game.tw-pddl", 8, 4)
    assert len(evidence) == len(control) == 4
    assert sorted(evidence + control) == list(range(8))
    splits = {tuple(split_rollouts(0, step, "/games/a/game.tw-pddl", 8, 4)[0]) for step in range(1, 30)}
    assert len(splits) > 1


def test_split_rejects_empty_control():
    with pytest.raises(ValueError):
        split_rollouts(0, 1, "g", 8, 8)


def test_evidence_prefers_success():
    success = {0: False, 1: True, 2: False, 3: False}
    assert choose_evidence([0, 1, 2, 3], success, 0, 1, "g") == 1
    chosen = choose_evidence([0, 2, 3], {0: False, 2: False, 3: False}, 0, 1, "g")
    assert chosen in (0, 2, 3)


def test_execution_signals_values():
    gain, trust, reward = execution_signals(0.5, 0.0, 4, 4.0, 2.0, 2.0)
    assert (gain, trust, reward) == pytest.approx((0.25, 1.5, 0.25))
    gain, trust, reward = execution_signals(-1.0, 0.0, 4, 4.0, 2.0, 2.0)
    assert (gain, trust, reward) == pytest.approx((-0.5, 0.0, -0.5))
    gain, trust, reward = execution_signals(1.0, 1.0, 4, 4.0, 2.0, 2.0)
    assert (gain, trust, reward) == pytest.approx((1.0, 2.0, 0.0))
    gain, trust, reward = execution_signals(None, 0.1, 4, 4.0, 2.0, 2.0)
    assert (gain, trust, reward) == pytest.approx((0.1, 1.2, 0.0))


def test_episode_utility_ranks_fast_success_over_slow_success_over_failure():
    assert episode_utility(False, 5, 30, 0.5) == 0.0
    assert episode_utility(True, 30, 30, 0.5) == pytest.approx(0.5)
    assert episode_utility(True, 6, 30, 0.5) == pytest.approx(0.9)
    assert episode_utility(True, 40, 30, 0.5) == pytest.approx(0.5)
    assert episode_utility(True, 12, 30, 0.0) == 1.0


def test_reliability_ema_and_step_order():
    state = Reliability(eta=0.1)
    state.commit(1, [0.5, 0.0])
    assert state.mean == pytest.approx(0.025)
    state.commit(2, [])
    assert state.mean == pytest.approx(0.025) and state.completed_step == 2
    with pytest.raises(ValueError):
        state.commit(4, [0.1])
    assert Reliability(**state.state()) == state


def _ids(text):
    return [ord(c) for c in text]


def test_skill_mask_covers_exactly_the_skill_value():
    skill = 'Go to the "sink" first, then clean.'
    text = json.dumps({"episode_summary": "Found apple.", "episode_skill": skill})
    mask = skill_token_mask(CharTokenizer(), _ids(text), text, skill)
    covered = "".join(c for c, m in zip(text, mask) if m)
    assert covered == json.dumps(skill)[1:-1]


def test_skill_mask_rejects_inconsistent_or_missing_skill():
    text = json.dumps({"episode_summary": "s", "episode_skill": "use the key"})
    assert sum(skill_token_mask(CharTokenizer(), _ids(text), text, "another skill")) == 0
    text = json.dumps({"episode_summary": "s"})
    assert sum(skill_token_mask(CharTokenizer(), _ids(text), text, "use the key")) == 0
