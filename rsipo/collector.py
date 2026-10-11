"""SEED's trajectory collector, plus the per-step context RSIPO needs to replay tasks with a skill."""
import json
from copy import deepcopy

import numpy as np

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

from .environments import SearchSkillManager, SkillConditionedManager, WebshopSkillManager


def _task_keys(gen_batch, envs, config):
    """One stable id per rollout: an ALFWorld game, a WebShop (worker seed, goal), or a Search question.

    A WebShop goal index only names a goal together with the worker seed, because the goal list is generated
    from that seed; SEED's workers use env.seed + group index.
    """
    raw = getattr(envs, "envs", None)
    goal_idxs = getattr(raw, "last_goal_idxs", None)
    if goal_idxs is not None:
        group, base = int(raw.group_n), int(config.env.seed)
        return [f"webshop:{base + index // group}:{int(goal)}" for index, goal in enumerate(goal_idxs)]
    kwargs = gen_batch.non_tensor_batch.get("env_kwargs") if gen_batch.non_tensor_batch is not None else None
    if kwargs is not None and len(kwargs) and isinstance(kwargs[0], dict) and "question" in kwargs[0]:
        keys = [
            json.dumps(
                {"q": item["question"], "g": item["ground_truth"], "d": item.get("data_source", "")},
                sort_keys=True, default=str,
            )
            for item in kwargs
        ]
        # The prompts are one row per task; the rollout loop repeats each task env.rollout.n times, adjacently.
        return [key for key in keys for _ in range(int(config.env.rollout.n))]
    return [str(game) for game in envs.gamefile]


class RsipoCollector(TrajectoryCollector):
    def multi_turn_loop(self, gen_batch, actor_rollout_wg, envs, is_train=True):
        if not is_train:
            return super().multi_turn_loop(gen_batch, actor_rollout_wg, envs, is_train)
        prompts = deepcopy(gen_batch)
        output = super().multi_turn_loop(gen_batch, actor_rollout_wg, envs, is_train)
        rows = output.non_tensor_batch
        self.rsipo_context = {
            "step": int(gen_batch.meta_info["global_step"]),
            "gen_batch": prompts,
            "task_keys": _task_keys(prompts, envs, self.config),
            "traj_uid": np.asarray(rows["traj_uid"], dtype=object).copy(),
            "sample_id": np.asarray(rows["sample_id"], dtype=np.int64).copy(),
            "rollout_id": np.asarray(rows["rollout_id"], dtype=np.int64).copy(),
            "step_num": np.asarray(rows["step_num"], dtype=np.int64).copy(),
            "anchor_obs": np.asarray(rows["anchor_obs"], dtype=object).copy(),
        }
        return output

    def replay_with_skills(self, gen_batch, actor_rollout_wg, task_keys, skills, rollouts_per_game, seed):
        """Run the current policy on each task with its skill in context; returns (won flags, episode lengths, initial observations, tokens)."""
        config = deepcopy(self.config)
        config.env.rollout.n = int(rollouts_per_game)
        collector = TrajectoryCollector(config, self.tokenizer, self.processor)
        flat_keys = [key for key in task_keys for _ in range(rollouts_per_game)]
        flat_skills = [skill for skill in skills for _ in range(rollouts_per_game)]
        env_name = str(config.env.env_name).lower()
        if env_name == "search":
            manager = SearchSkillManager([json.loads(key) for key in flat_keys], flat_skills, config, seed)
        elif env_name == "webshop":
            manager = WebshopSkillManager(flat_keys, flat_skills, config)
        else:
            manager = SkillConditionedManager(flat_keys, flat_skills, config, seed)
        try:
            prompts = deepcopy(gen_batch).select_idxs(np.arange(len(task_keys)))
            prompts = prompts.repeat(repeat_times=rollouts_per_game, interleave=True)
            parts = collector.vanilla_multi_turn_loop(
                gen_batch=prompts, actor_rollout_wg=actor_rollout_wg, envs=manager, phase="train"
            )
            generated_tokens = 0
            for trajectory in parts[0]:
                for step in trajectory:
                    if step.get("active_masks"):
                        width = step["responses"].shape[-1]
                        generated_tokens += int(step["attention_mask"][-width:].sum())
            if any(outcome is None for outcome in manager.outcomes):
                raise RuntimeError("Replay rollout is missing environment outcomes")
            return list(manager.outcomes), list(manager.lengths), list(manager.initial_observations), generated_tokens
        finally:
            manager.close()
