"""Replay ALFWorld training games with a skill inserted into the observation, as SEED's teacher sees it."""
from collections import defaultdict, deque
from functools import partial
from pathlib import Path

import numpy as np

from agent_system.environments.env_manager import AlfWorldEnvironmentManager, _projection_requires_think
from agent_system.environments.env_package.alfworld.projection import alfworld_projection
from scripts.sft._common.pipeline import FixedGameFileBatchEnv
from seed.prompting import build_augmented_observation_text

ALF_CONFIG = Path(__file__).resolve().parents[1] / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"


class OrderedBatch(FixedGameFileBatchEnv):
    """TextWorld may permute a batch of games on reset; restore the requested order."""

    def reset(self):
        obs, image, infos = super().reset()
        by_file = defaultdict(deque)
        for index, info in enumerate(infos):
            by_file[str(Path(info["extra.gamefile"]).resolve())].append(index)
        order = []
        for path in self.game_files:
            queue = by_file[str(Path(path).resolve())]
            if not queue:
                raise RuntimeError("TextWorld reset did not cover the requested game multiset")
            order.append(queue.popleft())
        if any(by_file.values()):
            raise RuntimeError("Unexpected games returned by TextWorld")
        self.order = order
        self.inverse = np.argsort(order).tolist()
        return [obs[i] for i in order], image, [infos[i] for i in order]

    def step(self, actions):
        physical = [actions[i] for i in self.inverse]
        obs, image, rewards, dones, infos = super().step(physical)
        return (
            [obs[i] for i in self.order],
            image,
            [rewards[i] for i in self.order],
            [dones[i] for i in self.order],
            [infos[i] for i in self.order],
        )

    @property
    def get_admissible_commands(self):
        return [self.prev_admissible_commands[i] for i in self.order]


class SkillConditionedManager(AlfWorldEnvironmentManager):
    def __init__(self, game_files, skills, config, seed):
        self.expected = [str(Path(path).resolve()) for path in game_files]
        self.skills = list(skills)
        if len(self.skills) != len(self.expected):
            raise ValueError("One skill is required per replayed game")
        raw = OrderedBatch(game_files=self.expected, alf_config_path=str(ALF_CONFIG), seed=seed)
        projection = partial(alfworld_projection, require_think=_projection_requires_think(config))
        super().__init__(raw, projection, config)

    def _with_skill(self, obs):
        obs["text_base"] = list(obs["text"])
        obs["text"] = [
            build_augmented_observation_text(observation=text, episode_skill=skill)
            for text, skill in zip(obs["text"], self.skills)
        ]
        return obs

    def reset(self, kwargs=None):
        obs, infos = super().reset(kwargs)
        actual = [str(Path(info["extra.gamefile"]).resolve()) for info in infos]
        if actual != self.expected:
            raise RuntimeError("Replay environment order changed")
        self.initial_observations = list(obs["anchor"])
        self.outcomes = [None] * len(infos)
        self.finished = [False] * len(infos)
        self.lengths = [0] * len(infos)
        return self._with_skill(obs), infos

    def step(self, actions):
        obs, rewards, dones, infos = super().step(actions)
        for index, info in enumerate(infos):
            if not self.finished[index]:
                if "won" not in info:
                    raise RuntimeError("ALFWorld did not return its won flag")
                self.outcomes[index] = bool(info["won"])
                self.lengths[index] += 1
                self.finished[index] = bool(dones[index])
        return self._with_skill(obs), rewards, dones, infos

    def close(self):
        self.envs.close()


def _track(manager, infos, dones):
    for index, info in enumerate(infos):
        if manager.finished[index]:
            continue
        if "won" not in info:
            raise RuntimeError("Environment did not return its won flag")
        manager.outcomes[index] = bool(info["won"])
        manager.lengths[index] += 1
        manager.finished[index] = bool(dones[index])


class SearchSkillManager:
    """Replay the same Search-QA questions with a skill inserted into each observation."""

    def __init__(self, specs, skills, config, seed):
        from functools import partial

        from agent_system.environments.env_manager import SearchEnvironmentManager
        from agent_system.environments.env_package.search import build_search_envs, search_projection

        self.specs = [
            {"question": spec["q"], "ground_truth": spec["g"], "data_source": spec.get("d", "")}
            for spec in specs
        ]
        self.skills = list(skills)
        if len(self.specs) != len(self.skills):
            raise ValueError("One skill is required per replayed question")
        import asyncio

        # SearchMultiProcessEnv installs its own event loop as the thread's current loop; keep the caller's.
        try:
            previous = asyncio.get_event_loop()
        except RuntimeError:
            previous = None
        raw = build_search_envs(
            seed=seed, env_num=len(self.specs), group_n=1, is_train=True, env_config=config.env,
        )
        asyncio.set_event_loop(previous)
        self.manager = SearchEnvironmentManager(raw, partial(search_projection), config)
        self.outcomes = [None] * len(self.specs)
        self.finished = [False] * len(self.specs)
        self.lengths = [0] * len(self.specs)
        self.initial_observations = []

    def _with_skill(self, obs):
        obs["text"] = [
            build_augmented_observation_text(observation=text, episode_skill=skill)
            for text, skill in zip(obs["text"], self.skills)
        ]
        return obs

    def reset(self, kwargs=None):
        obs, infos = self.manager.reset(kwargs=self.specs)
        self.initial_observations = list(obs["anchor"])
        self.outcomes = [None] * len(self.specs)
        self.finished = [False] * len(self.specs)
        self.lengths = [0] * len(self.specs)
        return self._with_skill(obs), infos

    def step(self, actions):
        obs, rewards, dones, infos = self.manager.step(actions)
        _track(self, infos, dones)
        return self._with_skill(obs), rewards, dones, infos

    def success_evaluator(self, **kwargs):
        return self.manager.success_evaluator(**kwargs)

    def close(self):
        raw = self.manager.envs
        executor = getattr(raw, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        loop = getattr(raw, "_loop", None)
        if loop is not None and not loop.is_closed():
            loop.close()


class _SeededWebshopEnvs:
    """WebShop workers built with the original groups' seeds, so each goal index names the same goal as in training."""

    def __init__(self, seeds, goal_idxs, env_kwargs):
        import ray

        from agent_system.environments.env_package.webshop.envs import WebshopWorker

        from .placement import node_resource

        actor = ray.remote(num_cpus=0.1)(WebshopWorker)
        self.goal_idxs = [int(goal) for goal in goal_idxs]
        self._workers = [
            actor.options(resources=node_resource(index)).remote(int(seed), dict(env_kwargs))
            for index, seed in enumerate(seeds)
        ]

    def reset(self):
        import ray

        results = ray.get([worker.reset.remote(goal) for worker, goal in zip(self._workers, self.goal_idxs)])
        return [obs for obs, _ in results], [info for _, info in results]

    def step(self, actions):
        import ray

        results = ray.get([worker.step.remote(action) for worker, action in zip(self._workers, actions)])
        obs, rewards, dones, infos = (list(part) for part in zip(*results))
        return obs, rewards, dones, infos

    def close(self):
        import ray

        for worker in self._workers:
            ray.kill(worker)


class WebshopSkillManager:
    """Replay the same WebShop goals (keys "webshop:<worker seed>:<goal>") with a skill inserted into each observation."""

    def __init__(self, task_keys, skills, config):
        import os
        from functools import partial

        from agent_system.environments.env_manager import WebshopEnvironmentManager, _projection_requires_think
        from agent_system.environments.env_package.webshop import webshop_projection

        parsed = [key.split(":") for key in task_keys]
        if any(len(parts) != 3 or parts[0] != "webshop" for parts in parsed):
            raise ValueError("WebShop replay keys must look like webshop:<seed>:<goal>")
        self.skills = list(skills)
        if len(parsed) != len(self.skills):
            raise ValueError("One skill is required per replayed goal")
        data_root = os.path.join(os.path.dirname(__file__), "..", "agent_system/environments/env_package/webshop/webshop/data")
        suffix = "_1000" if bool(config.env.webshop.use_small) else ""
        env_kwargs = {
            "observation_mode": "text", "num_products": None,
            "human_goals": bool(config.env.webshop.human_goals),
            "file_path": os.path.join(data_root, f"items_shuffle{suffix}.json"),
            "attr_path": os.path.join(data_root, f"items_ins_v2{suffix}.json"),
        }
        raw = _SeededWebshopEnvs([int(seed) for _, seed, _ in parsed], [int(goal) for _, _, goal in parsed], env_kwargs)
        projection = partial(webshop_projection, require_think=_projection_requires_think(config))
        self.manager = WebshopEnvironmentManager(raw, projection, config)
        self.outcomes = [None] * len(parsed)
        self.finished = [False] * len(parsed)
        self.lengths = [0] * len(parsed)
        self.initial_observations = []

    def _with_skill(self, obs):
        obs["text"] = [
            build_augmented_observation_text(observation=text, episode_skill=skill)
            for text, skill in zip(obs["text"], self.skills)
        ]
        return obs

    def reset(self, kwargs=None):
        obs, infos = self.manager.reset(kwargs)
        self.initial_observations = list(obs["anchor"])
        self.outcomes = [None] * len(self.skills)
        self.finished = [False] * len(self.skills)
        self.lengths = [0] * len(self.skills)
        return self._with_skill(obs), infos

    def step(self, actions):
        obs, rewards, dones, infos = self.manager.step(actions)
        _track(self, infos, dones)
        return self._with_skill(obs), rewards, dones, infos

    def success_evaluator(self, **kwargs):
        return self.manager.success_evaluator(**kwargs)

    def close(self):
        self.manager.envs.close()
