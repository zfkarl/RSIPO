import json
from types import SimpleNamespace

import numpy as np

from rsipo.collector import _task_keys


def config(seed=0, group=8):
    return SimpleNamespace(env=SimpleNamespace(seed=seed, rollout=SimpleNamespace(n=group)))


def test_alfworld_uses_the_game_file_of_each_rollout():
    envs = SimpleNamespace(gamefile=["/g/a/game.tw-pddl", "/g/a/game.tw-pddl", "/g/b/game.tw-pddl"], envs=SimpleNamespace())
    batch = SimpleNamespace(non_tensor_batch={})
    assert _task_keys(batch, envs, config()) == ["/g/a/game.tw-pddl", "/g/a/game.tw-pddl", "/g/b/game.tw-pddl"]


def test_webshop_key_carries_the_group_worker_seed():
    raw = SimpleNamespace(last_goal_idxs=[700, 700, 900, 900], group_n=2)
    keys = _task_keys(SimpleNamespace(non_tensor_batch={}), SimpleNamespace(envs=raw), config(seed=5, group=2))
    assert keys == ["webshop:5:700", "webshop:5:700", "webshop:6:900", "webshop:6:900"]


def test_search_keys_repeat_each_question_for_its_group():
    kwargs = np.array([
        {"question": "q1", "ground_truth": {"target": ["a"]}, "data_source": "nq"},
        {"question": "q2", "ground_truth": {"target": ["b"]}, "data_source": "hotpotqa"},
    ], dtype=object)
    envs = SimpleNamespace(envs=SimpleNamespace())
    keys = _task_keys(SimpleNamespace(non_tensor_batch={"env_kwargs": kwargs}), envs, config(group=3))
    assert len(keys) == 6 and keys[0] == keys[2] != keys[3]
    assert json.loads(keys[3]) == {"d": "hotpotqa", "g": {"target": ["b"]}, "q": "q2"}
