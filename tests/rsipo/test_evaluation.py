"""rsipo/evaluation.py: per-trajectory outcomes from the val envs' infos, Search-QA and WebShop metrics, rounds."""
import json

import numpy as np
from omegaconf import OmegaConf

from rsipo import evaluation


def test_training_configs_without_the_eval_group_keep_training():
    config = OmegaConf.create({"rsipo": {"enable": True}})
    OmegaConf.set_struct(config, True)
    assert config.get("rsipo_eval") is None


def test_final_infos_takes_each_trajectorys_last_active_step():
    batch = [[{"active_masks": True}, {"active_masks": True}, {"active_masks": False}], [{"active_masks": True}]]
    infos = [[{"won": False}, {"won": True}, {"won": False}], [{"won": False}]]
    assert [info["won"] for info in evaluation.final_infos(infos, batch)] == [True, False]


def test_search_metrics_are_per_question_by_dataset():
    records = [{"won": 1.0, "data_source": source} for source in evaluation.SEARCH_SOURCES]
    records += [{"won": 0.0, "data_source": "nq"}] * 3
    out = evaluation.round_metrics("search", records)
    assert out["accuracy_pct"]["nq"] == 25.0
    assert out["accuracy_pct"]["bamboogle"] == 100.0
    assert out["questions_by_source"]["nq"] == 4
    assert abs(out["macro_accuracy_pct"] - (25.0 + 600.0) / 7) < 1e-9
    assert abs(out["overall_accuracy_pct"] - 700.0 / 10) < 1e-9
    keys = evaluation.wandb_metrics("search", [out])
    assert keys["eval/search_nq_acc_pct_round1"] == 25.0
    assert keys["eval/search_macro_acc_pct_mean"] == keys["eval/search_macro_acc_pct_max"] == out["macro_accuracy_pct"]


def test_webshop_metrics_score_and_success():
    records = [{"won": 1.0, "task_score": 1.0, "data_source": None}, {"won": 0.0, "task_score": 0.5, "data_source": None}]
    assert evaluation.round_metrics("webshop", records) == {"episodes": 2, "score_pct": 75.0, "success_pct": 50.0}


class _Raw:
    def __init__(self):
        self._rng = np.random.RandomState(7)


class _Envs:
    def __init__(self):
        self.envs = _Raw()
        self.calls = 0

    def success_evaluator(self, **kwargs):
        self.calls += 1
        return {"success_rate": np.array([float(i["won"]) for i in evaluation.final_infos(kwargs["total_infos"],
                                                                                          kwargs["total_batch_list"])])}


class _Trainer:
    def __init__(self, tmp_path, env_name, rounds):
        self.config = OmegaConf.create({
            "rsipo_eval": {"dir": str(tmp_path / "eval"), "rounds": rounds},
            "env": {"env_name": env_name, "seed": 0, "max_steps": 15, "history_length": 2},
            "data": {"val_batch_size": 3},
            "trainer": {"resume_from_path": "/x/checkpoints/global_step_150"},
            "actor_rollout_ref": {"rollout": {"val_kwargs": {"temperature": 0.4, "do_sample": True, "n": 1}}},
        })
        self.val_envs = _Envs()
        self.global_steps = 0
        self.goals = []

    def _load_checkpoint(self):
        self.global_steps = 150

    def _validate(self):
        goals = self.val_envs.envs._rng.choice(500, size=3, replace=False).tolist()
        self.goals.append(goals)
        batch = [[{"active_masks": True}] for _ in goals]
        infos = [[{"won": goal % 2 == 0, "task_score": 0.5 + 0.5 * (goal % 2 == 0)}] for goal in goals]
        self.val_envs.success_evaluator(total_infos=infos, total_batch_list=batch, episode_rewards=None, episode_lengths=None)
        return {"val/success_rate": 0.5, "val/empty": float("nan")}


def test_webshop_rounds_reuse_the_first_validation_draw(tmp_path):
    trainer = _Trainer(tmp_path, "Webshop", rounds=3)
    summary = evaluation.evaluate(trainer)
    first = np.random.RandomState(1000).choice(500, size=3, replace=False).tolist()
    assert trainer.goals == [first, first, first]
    assert summary["complete"] and summary["checkpoint_step"] == 150 and len(summary["rounds"]) == 3
    written = json.loads((tmp_path / "eval" / "summary.json").read_text())
    assert written["rounds"][0]["validate_metrics"]["val/empty"] is None
    expected = 100.0 * np.mean([goal % 2 == 0 for goal in first])
    assert abs(written["metrics"]["eval/webshop_success_pct_mean"] - expected) < 1e-9
    assert {"eval/webshop_score_pct_max", "eval/webshop_success_pct_round3"} <= set(written["metrics"])
    assert len((tmp_path / "eval" / "round-2.jsonl").read_text().splitlines()) == 3
