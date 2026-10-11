import json

import pytest

from rsipo.execution import TypeStats, gain_noise_variance, pooled_signals, score_pooled, task_type

ALF = "/data/alfworld/json_2.1.1/train/{}-Apple-None-Fridge-1/trial_T2019/game.tw-pddl"


def test_task_type_per_environment():
    assert task_type(ALF.format("pick_and_place_simple")) == "pick_and_place"
    assert task_type(ALF.format("pick_two_obj_and_place")) == "pick_two_obj_and_place"
    assert task_type(ALF.format("look_at_obj_in_light")) == "look_at_obj_in_light"
    for kind in ("heat", "cool", "clean"):
        assert task_type(ALF.format(f"pick_{kind}_then_place_in_recep")) == f"pick_{kind}_then_place_in_recep"
    search = json.dumps({"q": "who wrote it?", "g": ["x"], "d": "hotpotqa"}, sort_keys=True)
    assert task_type(search) == "search:hotpotqa"
    assert task_type(json.dumps({"q": "q", "g": []})) == "search:unknown"
    assert task_type("webshop:3:517") == "webshop"
    assert task_type("/somewhere/else") == "other"


def test_noise_variance_is_positive_when_every_rollout_failed():
    assert gain_noise_variance(0.0, 4, 0.0, 4) == pytest.approx(2 * (1 / 6) * (5 / 6) / 4)
    assert gain_noise_variance(0.5, 4, 0.5, 4) == pytest.approx(2 * 0.25 / 4)


def test_type_stats_bias_correction_and_order():
    stats = TypeStats(eta=0.1)
    assert stats.estimate("a") == (0.0, 0.0)
    first = stats.updated(1, {"a": [(0.5, 0.01), (-0.1, 0.03)]})
    mean, between = first.estimate("a")
    assert mean == pytest.approx(0.2)
    assert between == pytest.approx(max(0.0, (0.25 + 0.01) / 2 - 0.04 - 0.02))
    assert stats.completed_step == 0 and first.completed_step == 1
    second = first.updated(2, {"b": [(0.3, 0.0)]})
    assert second.estimate("a") == pytest.approx(first.estimate("a"))  # an absent type is not decayed
    assert second.estimate("b") == pytest.approx((0.3, 0.0))
    with pytest.raises(ValueError):
        second.updated(4, {})
    assert TypeStats(**json.loads(json.dumps(second.state()))) == second


def test_pooled_signals_shrinkage():
    # No detectable spread between tasks: trust follows the type mean, the skill reward still sees the task's own delta.
    gain, trust, reward, weight = pooled_signals(0.5, 0.1, 0.1, 0.0, 2.0, 2.0)
    assert (gain, trust, reward, weight) == pytest.approx((0.1, 1.2, 0.4, 0.0))
    gain, trust, reward, weight = pooled_signals(0.5, 0.1, 0.1, 0.1, 2.0, 2.0)
    assert (gain, trust, reward, weight) == pytest.approx((0.3, 1.6, 0.4, 0.5))
    gain, trust, reward, weight = pooled_signals(None, 0.0, -0.8, 0.2, 2.0, 2.0)
    assert (gain, trust, reward, weight) == pytest.approx((-0.8, 0.0, 0.0, 0.0))
    assert pooled_signals(0.0, 0.0, 0.0, 0.0, 2.0, 2.0) == pytest.approx((0.0, 1.0, 0.0, 0.0))


def _record(game, delta, valid=True, teacher=0.5, control=0.25):
    return {
        "game_file": game, "delta": delta, "skill_valid": valid, "teacher_utility": teacher, "control_utility": control,
        "teacher_success": [1, 0, 1, 0], "control_ids": [0, 1, 2, 3],
    }


def test_score_pooled_fills_records_and_penalizes_invalid_skills():
    records = [
        _record(ALF.format("pick_and_place_simple"), 0.25),
        _record(ALF.format("pick_and_place_simple"), -0.25, teacher=0.0, control=0.25),
        _record(ALF.format("look_at_obj_in_light"), None, valid=False),
    ]
    advanced = score_pooled(records, TypeStats(eta=0.1), 1, 2.0, 2.0, 0.2)
    assert advanced.completed_step == 1 and set(advanced.moments) == {"pick_and_place"}
    type_mean, between = advanced.estimate("pick_and_place")
    assert type_mean == pytest.approx(0.0)
    for record in records[:2]:
        assert record["type_mean"] == pytest.approx(type_mean)
        assert 0.0 <= record["shrink_weight"] <= 1.0
        assert record["gain"] == pytest.approx(record["shrink_weight"] * record["delta"] + (1 - record["shrink_weight"]) * type_mean)
        assert record["skill_reward"] == pytest.approx(record["delta"] - type_mean)
    assert records[2]["gain"] == 0.0 and records[2]["trust"] == 1.0 and records[2]["skill_reward"] == -0.2
    assert "noise_var" not in records[2]
