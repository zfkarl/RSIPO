"""Checkpoint evaluation through the trainer's own validation loop: SEED's val envs, prompts, and val sampling.

_validate reports a mean of per-batch means. Here every round also records each trajectory's final outcome from the
environment infos that success_evaluator receives (the same infos SEED scores), so per-dataset accuracy is an exact
per-question mean. Configured by ``+rsipo_eval.dir=PATH`` and ``+rsipo_eval.rounds=N``.
"""
import json
import time
from pathlib import Path

import numpy as np

from .execution import atomic_json

SEARCH_SOURCES = ("nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle")


def final_infos(total_infos, total_batch_list):
    """The info of each trajectory's last active step, as SEED's _process_batch reads it."""
    out = []
    for infos, steps in zip(total_infos, total_batch_list):
        last = [info for info, step in zip(infos, steps) if step["active_masks"]]
        if not last:
            raise RuntimeError("A validation trajectory has no active step")
        out.append(last[-1])
    return out


class OutcomeRecorder:
    """Wraps the val envs' success_evaluator and keeps one record per trajectory."""

    def __init__(self, envs):
        self.evaluate = envs.success_evaluator
        self.records = []

    def __call__(self, **kwargs):
        for info in final_infos(kwargs["total_infos"], kwargs["total_batch_list"]):
            record = {"won": float(info["won"]), "data_source": info.get("data_source")}
            if "task_score" in info:
                record["task_score"] = float(info["task_score"])
            self.records.append(record)
        return self.evaluate(**kwargs)


def round_metrics(env_name, records):
    if "search" in env_name:
        by_source = {}
        for record in records:
            by_source.setdefault(record["data_source"], []).append(record["won"])
        accuracy = {source: 100.0 * float(np.mean(values)) for source, values in by_source.items()}
        out = {"questions": len(records), "questions_by_source": {k: len(v) for k, v in by_source.items()},
               "accuracy_pct": accuracy, "overall_accuracy_pct": 100.0 * float(np.mean([r["won"] for r in records]))}
        if all(source in accuracy for source in SEARCH_SOURCES):
            out["macro_accuracy_pct"] = float(np.mean([accuracy[source] for source in SEARCH_SOURCES]))
        return out
    if "webshop" in env_name:
        return {"episodes": len(records),
                "score_pct": 100.0 * float(np.mean([r["task_score"] for r in records])),
                "success_pct": 100.0 * float(np.mean([r["won"] for r in records]))}
    return {"episodes": len(records), "success_pct": 100.0 * float(np.mean([r["won"] for r in records]))}


def wandb_metrics(env_name, rounds):
    """Flat eval/* keys for the loggers: the mean and max over rounds, and each round."""
    if "search" in env_name:
        series = {f"eval/search_{source}_acc_pct": [r["accuracy_pct"].get(source) for r in rounds] for source in SEARCH_SOURCES}
        series["eval/search_macro_acc_pct"] = [r.get("macro_accuracy_pct") for r in rounds]
        series["eval/search_overall_acc_pct"] = [r["overall_accuracy_pct"] for r in rounds]
    elif "webshop" in env_name:
        series = {"eval/webshop_score_pct": [r["score_pct"] for r in rounds],
                  "eval/webshop_success_pct": [r["success_pct"] for r in rounds]}
    else:
        series = {"eval/success_pct": [r["success_pct"] for r in rounds]}
    out = {}
    for key, values in series.items():
        values = [v for v in values if v is not None]
        if not values:
            continue
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_max"] = float(np.max(values))
        for index, value in enumerate(values, 1):
            out[f"{key}_round{index}"] = float(value)
    return out


def evaluate(trainer):
    config = trainer.config
    out = Path(config.rsipo_eval.dir)
    out.mkdir(parents=True, exist_ok=True)
    rounds = int(config.rsipo_eval.get("rounds", 1))
    env_name = str(config.env.env_name).lower()
    trainer._load_checkpoint()
    if trainer.global_steps == 0:
        raise RuntimeError("Evaluation needs trainer.resume_mode=resume_path and a checkpoint")
    recorder = OutcomeRecorder(trainer.val_envs)
    trainer.val_envs.success_evaluator = recorder
    summary = {
        "env": env_name,
        "checkpoint": str(config.trainer.resume_from_path),
        "checkpoint_step": int(trainer.global_steps),
        "protocol": {
            "val_batch_size": int(config.data.val_batch_size),
            "temperature": float(config.actor_rollout_ref.rollout.val_kwargs.temperature),
            "do_sample": bool(config.actor_rollout_ref.rollout.val_kwargs.do_sample),
            "n": int(config.actor_rollout_ref.rollout.val_kwargs.n),
            "max_steps": int(config.env.max_steps),
            "history_length": int(config.env.history_length),
            "env_seed": int(config.env.seed) + 1000,
            "skills_at_test": False,
        },
        "rounds": [],
    }
    for index in range(1, rounds + 1):
        if "webshop" in env_name:
            # The val workers draw goals from this generator on every reset; restoring it gives every round the same
            # 128 goals, the first draw that SEED's validation makes.
            trainer.val_envs.envs._rng = np.random.RandomState(int(config.env.seed) + 1000)
        recorder.records = []
        start = time.time()
        stock = trainer._validate()
        records = recorder.records
        if not records:
            raise RuntimeError("Validation produced no trajectories")
        with (out / f"round-{index}.jsonl").open("w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        result = {"round": index, "seconds": time.time() - start, **round_metrics(env_name, records),
                  "validate_metrics": {key: float(value) if np.isfinite(value) else None for key, value in stock.items()}}
        summary["rounds"].append(result)
        summary["metrics"] = wandb_metrics(env_name, summary["rounds"])
        atomic_json(out / "summary.json", summary)
        print(f"RSIPO evaluation round {index}/{rounds}: {json.dumps({k: v for k, v in result.items() if k != 'validate_metrics'})}",
              flush=True)
    summary["complete"] = True
    atomic_json(out / "summary.json", summary)
    return summary
