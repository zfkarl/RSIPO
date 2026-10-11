"""The Search and WebShop replay managers serve every call vanilla_multi_turn_loop makes on its envs."""
from agent_system.environments.env_manager import SearchEnvironmentManager, WebshopEnvironmentManager
from rsipo.environments import SearchSkillManager, WebshopSkillManager

BATCH = [[{"active_masks": True}, {"active_masks": False}], [{"active_masks": True}, {"active_masks": True}]]


def _replay(cls, inner):
    manager = object.__new__(cls)
    manager.manager = object.__new__(inner)
    return manager


def _evaluate(manager, infos):
    return manager.success_evaluator(total_infos=infos, total_batch_list=BATCH, episode_rewards=None, episode_lengths=None)


def test_search_replay_reports_success_like_training():
    infos = [[{"won": True, "data_source": "nq"}, {}],
             [{"won": True, "data_source": "nq"}, {"won": False, "data_source": "hotpotqa"}]]
    out = _evaluate(_replay(SearchSkillManager, SearchEnvironmentManager), infos)
    assert list(out["success_rate"]) == [1.0, 0.0]
    assert list(out["nq_success_rate"]) == [1.0]
    assert list(out["hotpotqa_success_rate"]) == [0.0]


def test_webshop_replay_reports_success_and_score_like_training():
    infos = [[{"won": True, "task_score": 1.0}, {}],
             [{"won": False, "task_score": 0.3}, {"won": False, "task_score": 0.5}]]
    out = _evaluate(_replay(WebshopSkillManager, WebshopEnvironmentManager), infos)
    assert list(out["success_rate"]) == [1.0, 0.0]
    assert list(out["webshop_task_score (not success_rate)"]) == [1.0, 0.5]


def test_replay_managers_cover_the_rollout_loop_calls():
    for cls in (SearchSkillManager, WebshopSkillManager):
        for name in ("reset", "step", "success_evaluator", "close"):
            assert callable(getattr(cls, name)), (cls.__name__, name)
