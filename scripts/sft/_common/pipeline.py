#!/usr/bin/env python3
"""Build verified episode-skill SFT data for ALFWorld.

This script is intentionally standalone: it adds a new offline pipeline without
editing the existing trainer or SEED implementation. It reuses the current
ALFWorld prompt manager, SEED episode-only analyzer prompt, and SEED skill
injection format.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from openai import OpenAI
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.prompt_agent.local_vllm_alfworld import (  # noqa: E402
    LocalOpenAIAgent,
    collect_actions_concurrently,
    has_action_format,
    json_safe,
    load_env_file,
    resolve_extra_body,
)
from seed.prompting import build_augmented_observation_text  # noqa: E402


TASK_TYPES = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]

RAW_TO_TASK_TYPE = {
    "pick_and_place_simple": "pick_and_place",
    "pick_and_place": "pick_and_place",
    "pick_two_obj_and_place": "pick_two_obj_and_place",
    "look_at_obj_in_light": "look_at_obj_in_light",
    "pick_heat_then_place_in_recep": "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep": "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep": "pick_clean_then_place_in_recep",
}

OUTPUT_FILES = [
    "sampled_tasks.jsonl",
    "baseline_rollouts.jsonl",
    "candidate_skills.jsonl",
    "skill_validations.jsonl",
    "accepted_skills.jsonl",
    "rejected_skills.jsonl",
    "metrics.json",
    "sft_episode_skill_train.parquet",
    "sft_episode_skill_val.parquet",
    "sft_episode_skill_all.jsonl",
    "progress.json",
]


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


@dataclass(frozen=True)
class ChatEndpoint:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_completion_tokens: int
    timeout: float
    retries: int
    retry_delay: float
    extra_body: Optional[Dict[str, Any]]


class OpenAITextClient:
    def __init__(self, endpoint: ChatEndpoint):
        self.endpoint = endpoint
        self.client = OpenAI(
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            timeout=endpoint.timeout,
            max_retries=0,
        )

    def complete(self, messages: List[Dict[str, str]]) -> Tuple[str, Optional[str]]:
        last_error: Optional[BaseException] = None
        for attempt in range(max(1, self.endpoint.retries)):
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": self.endpoint.model,
                    "messages": messages,
                    "temperature": self.endpoint.temperature,
                    "max_completion_tokens": self.endpoint.max_completion_tokens,
                    "n": 1,
                }
                if self.endpoint.extra_body:
                    request_kwargs["extra_body"] = self.endpoint.extra_body
                response = self.client.chat.completions.create(**request_kwargs)
                choice = response.choices[0]
                message = choice.message
                content = getattr(message, "content", "") or ""
                reasoning = getattr(message, "reasoning_content", None)
                if reasoning and content and "<think>" not in content.lower():
                    return f"<think>{reasoning}</think>\n{content}", None
                return str(content or reasoning or ""), None
            except Exception as exc:
                last_error = exc
                if attempt < self.endpoint.retries - 1:
                    time.sleep(self.endpoint.retry_delay)
        return "", f"{type(last_error).__name__}: {last_error}"


def load_config_file(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Invalid ALFWorld config file: {path}")
    with open(path, "r", encoding="utf-8") as reader:
        return yaml.safe_load(reader)


def compute_reward(info: Dict[str, Any], multi_modal: bool = False) -> float:
    if multi_modal:
        return 10.0 * float(info["won"]) + float(info["goal_condition_success_rate"])
    return 10.0 * float(info["won"])


class FixedGameFileBatchEnv:
    """A batch ALFWorld text env bound to an explicit list of game files."""

    def __init__(
        self,
        *,
        game_files: Sequence[str],
        alf_config_path: str,
        seed: int,
    ):
        self.game_files = [str(path) for path in game_files]
        if not self.game_files:
            raise ValueError("FixedGameFileBatchEnv requires at least one game file.")

        self.config = load_config_file(alf_config_path)
        self.prev_admissible_commands: List[List[str]] = [[] for _ in self.game_files]
        self.env = self._make_env()
        if hasattr(self.env, "seed"):
            self.env.seed(seed)

    def _make_env(self):
        import textworld
        import textworld.gym
        from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler,
            AlfredInfos,
        )

        domain_randomization = bool(self.config["env"].get("domain_randomization", False))
        wrappers = [AlfredDemangler(shuffle=domain_randomization), AlfredInfos]
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        training_method = self.config["general"]["training_method"]
        if training_method == "dqn":
            max_steps = self.config["rl"]["training"]["max_nb_steps_per_episode"]
        elif training_method == "dagger":
            max_steps = self.config["dagger"]["training"]["max_nb_steps_per_episode"]
        else:
            raise NotImplementedError(f"Unsupported ALFWorld training_method={training_method!r}")

        env_id = textworld.gym.register_games(
            self.game_files,
            request_infos,
            batch_size=len(self.game_files),
            asynchronous=True,
            max_episode_steps=max_steps,
            wrappers=wrappers,
        )
        return textworld.gym.make(env_id)

    @staticmethod
    def _split_infos(infos: Any, batch_size: int) -> List[Dict[str, Any]]:
        if isinstance(infos, list):
            return [dict(info) for info in infos]
        if not isinstance(infos, dict):
            return [{} for _ in range(batch_size)]

        split: List[Dict[str, Any]] = []
        for idx in range(batch_size):
            item: Dict[str, Any] = {}
            for key, value in infos.items():
                if isinstance(value, (list, tuple, np.ndarray)) and len(value) == batch_size:
                    item[key] = value[idx]
                else:
                    item[key] = value
            split.append(item)
        return split

    def reset(self):
        obs, infos = self.env.reset()
        obs_list = list(obs)
        info_list = self._split_infos(infos, len(obs_list))
        self.prev_admissible_commands = [
            list(info.get("admissible_commands", [])) for info in info_list
        ]
        return obs_list, None, info_list

    def step(self, actions: Sequence[str]):
        obs, _scores, dones, infos = self.env.step(list(actions))
        obs_list = list(obs)
        info_list = self._split_infos(infos, len(obs_list))
        self.prev_admissible_commands = [
            list(info.get("admissible_commands", [])) for info in info_list
        ]
        rewards = [compute_reward(info, multi_modal=False) for info in info_list]
        return obs_list, None, rewards, list(dones), info_list

    @property
    def get_admissible_commands(self):
        return self.prev_admissible_commands

    def close(self):
        close = getattr(self.env, "close", None)
        if close is not None:
            close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--alf-config",
        default=str(
            PROJECT_ROOT / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
        ),
    )
    parser.add_argument("--output-dir", default="outputs/alfworld_episode_skill_pipeline")
    parser.add_argument("--tasks-per-type", type=int, default=30)
    parser.add_argument("--rollouts-per-task", type=int, default=8)
    parser.add_argument("--validation-rollouts", type=int, default=8)
    parser.add_argument(
        "--task-batch-size",
        type=int,
        default=4,
        help="Number of different tasks to rollout in the same baseline wave.",
    )
    parser.add_argument(
        "--skill-batch-size",
        type=int,
        default=4,
        help="Number of different candidate skills to validate in the same wave.",
    )
    parser.add_argument(
        "--skill-gen-workers",
        type=int,
        default=128,
        help="Concurrent requests used to generate candidate episode skills.",
    )
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-workers", type=int, default=8)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")

    parser.add_argument("--policy-base-url", default=None)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--policy-model", default=None)
    parser.add_argument("--policy-temperature", type=float, default=0.4)
    parser.add_argument("--policy-max-completion-tokens", type=int, default=512)
    parser.add_argument("--policy-timeout", type=float, default=120.0)
    parser.add_argument("--policy-retries", type=int, default=2)
    parser.add_argument("--policy-retry-delay", type=float, default=1.0)
    parser.add_argument("--policy-extra-body-json", default=None)
    parser.add_argument("--fallback-action", default="look")

    parser.add_argument("--skill-base-url", default=None)
    parser.add_argument("--skill-api-key", default=None)
    parser.add_argument("--skill-model", default=None)
    parser.add_argument("--skill-temperature", type=float, default=0.0)
    parser.add_argument("--skill-max-completion-tokens", type=int, default=1024)
    parser.add_argument("--skill-timeout", type=float, default=120.0)
    parser.add_argument("--skill-retries", type=int, default=5)
    parser.add_argument("--skill-retry-delay", type=float, default=1.0)
    parser.add_argument("--skill-extra-body-json", default=None)

    parser.add_argument("--accept-min-delta-count", type=int, default=1)
    parser.add_argument("--accept-min-delta-rate", type=float, default=0.125)
    parser.add_argument("--sft-val-ratio", type=float, default=0.1)
    return parser.parse_args()


def setup_logging(output_dir: Path, level: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    for noisy_logger in ("httpx", "httpcore", "openai", "openai._base_client"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def resolve_endpoint(
    *,
    prefix: str,
    args: argparse.Namespace,
    default_base_url_env: str,
    default_model_env: str,
    default_model: str,
    temperature: float,
    max_completion_tokens: int,
    timeout: float,
    retries: int,
    retry_delay: float,
    extra_body_json: Optional[str],
) -> ChatEndpoint:
    base_url = getattr(args, f"{prefix}_base_url") or os.environ.get(default_base_url_env)
    if not base_url:
        base_url = os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1"
    api_key = getattr(args, f"{prefix}_api_key") or os.environ.get(
        "OPENAI_API_KEY" if prefix == "policy" else "SKILL_OPENAI_API_KEY"
    )
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
    model = getattr(args, f"{prefix}_model") or os.environ.get(default_model_env)
    if not model:
        model = os.environ.get("OPENAI_MODEL") or default_model
    return ChatEndpoint(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
        extra_body=resolve_extra_body(extra_body_json),
    )


def prepare_output_dir(output_dir: Path, *, resume: bool, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if overwrite:
        for path in existing:
            path.unlink()
        return
    if existing and not resume:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Output files already exist in {output_dir}: {names}. "
            "Use --resume to continue or --overwrite to replace them."
        )


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(payload), file, ensure_ascii=False, indent=2)


def update_progress(
    output_dir: Path,
    *,
    stage: str,
    status: str,
    **details: Any,
) -> None:
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": stage,
        "status": status,
        **details,
    }
    write_json(output_dir / "progress.json", payload)


def log_stage(output_dir: Path, stage: str, status: str = "running", **details: Any) -> None:
    logging.info("========== %s: %s ==========", stage, status)
    if details:
        logging.info("%s details: %s", stage, json.dumps(json_safe(details), ensure_ascii=False))
    update_progress(output_dir, stage=stage, status=status, **details)


def task_type_from_raw(raw: str, path: Optional[str] = None) -> Optional[str]:
    raw = str(raw or "")
    if raw in RAW_TO_TASK_TYPE:
        return RAW_TO_TASK_TYPE[raw]
    text = f"{raw} {path or ''}"
    for task_type in TASK_TYPES:
        if task_type in text:
            return task_type
    if "pick_and_place_simple" in text:
        return "pick_and_place"
    return None


def collect_train_game_files(alf_config_path: str) -> Dict[str, List[Dict[str, Any]]]:
    config = load_config_file(alf_config_path)
    data_path = os.path.expandvars(config["dataset"]["data_path"])
    if not os.path.isdir(data_path):
        logging.warning(
            "ALFWorld train data path does not exist: %s. Check ALFWORLD_DATA or --alf-config.",
            data_path,
        )
    grouped: Dict[str, List[Dict[str, Any]]] = {task_type: [] for task_type in TASK_TYPES}

    for root, _dirs, files in os.walk(data_path, topdown=False):
        if "traj_data.json" not in files:
            continue
        if "movable" in root or "Sliced" in root:
            continue
        traj_path = Path(root) / "traj_data.json"
        game_path = Path(root) / "game.tw-pddl"
        if not game_path.exists():
            continue
        try:
            traj_data = json.loads(traj_path.read_text(encoding="utf-8"))
            game_data = json.loads(game_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.warning("Skipping malformed ALFWorld task under %s: %s", root, exc)
            continue
        if not game_data.get("solvable", False):
            continue
        task_type = task_type_from_raw(traj_data.get("task_type"), str(game_path))
        if task_type not in grouped:
            continue
        grouped[task_type].append(
            {
                "task_type": task_type,
                "task_type_raw": traj_data.get("task_type", ""),
                "game_file": str(game_path),
                "traj_data_path": str(traj_path),
            }
        )

    for records in grouped.values():
        records.sort(key=lambda item: item["game_file"])
    return grouped


def sample_tasks(args: argparse.Namespace, output_dir: Path) -> List[Dict[str, Any]]:
    sampled_path = output_dir / "sampled_tasks.jsonl"
    existing = read_jsonl(sampled_path)
    if existing and args.resume:
        return existing

    grouped = collect_train_game_files(args.alf_config)
    config = load_config_file(args.alf_config)
    train_data_path = os.path.expandvars(config["dataset"]["data_path"])
    rng = random.Random(args.seed)
    sampled: List[Dict[str, Any]] = []
    for task_type in TASK_TYPES:
        pool = list(grouped.get(task_type, []))
        if len(pool) < args.tasks_per_type:
            raise ValueError(
                f"Need {args.tasks_per_type} tasks for {task_type}, found {len(pool)} "
                f"under {train_data_path}. Check ALFWORLD_DATA and the train split files."
            )
        rng.shuffle(pool)
        for idx, item in enumerate(pool[: args.tasks_per_type]):
            record = dict(item)
            record["task_id"] = f"{task_type}_{idx:03d}_{uuid.uuid5(uuid.NAMESPACE_URL, item['game_file']).hex[:10]}"
            record["sample_index_in_type"] = idx
            record["split"] = "train"
            sampled.append(record)

    if args.max_tasks is not None:
        sampled = sampled[: max(0, args.max_tasks)]

    for record in sampled:
        append_jsonl(sampled_path, record)
    logging.info("Sampled %d ALFWorld tasks.", len(sampled))
    return sampled


def build_manager(
    *,
    game_files: Sequence[str],
    alf_config_path: str,
    seed: int,
    history_length: int,
) -> AlfWorldEnvironmentManager:
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    from agent_system.environments.env_package.alfworld.projection import alfworld_projection

    envs = FixedGameFileBatchEnv(
        game_files=game_files,
        alf_config_path=alf_config_path,
        seed=seed,
    )
    config = AttrDict(
        {
            "env": AttrDict(
                {
                    "env_name": "alfworld/AlfredTWEnv",
                    "history_length": history_length,
                    "use_skills_only_memory": False,
                    "use_retrieval_memory": False,
                }
            )
        }
    )
    return AlfWorldEnvironmentManager(envs, alfworld_projection, config)


def make_policy_agent(endpoint: ChatEndpoint, fallback_action: str) -> LocalOpenAIAgent:
    return LocalOpenAIAgent(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        model_name=endpoint.model,
        temperature=endpoint.temperature,
        max_completion_tokens=endpoint.max_completion_tokens,
        timeout=endpoint.timeout,
        retries=endpoint.retries,
        retry_delay=endpoint.retry_delay,
        fallback_action=fallback_action,
        extra_body=endpoint.extra_body,
    )


def infer_task_type_from_gamefile(gamefile: str) -> str:
    detected = task_type_from_raw("", gamefile)
    return detected or "other"


def extract_action_text(model_response: str) -> str:
    text = str(model_response or "")
    lowered = text.lower()
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = lowered.find(start_tag)
    end_idx = lowered.find(end_tag)
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return text[-30:].strip().lower()
    return text[start_idx + len(start_tag) : end_idx].strip().lower()


def extract_standard_alfworld_observation(text: Any) -> str:
    """Return raw ALFWorld feedback from an agent prompt when possible."""
    observation = str(text or "").strip()
    if not observation:
        return ""

    end_markers = (
        "\nYour admissible actions of the current situation are:",
        "Your admissible actions of the current situation are:",
    )
    start_markers = (
        "and your current observation is:",
        "Your current observation is:",
    )
    for start_marker in start_markers:
        start_idx = observation.find(start_marker)
        if start_idx == -1:
            continue
        start_idx += len(start_marker)
        end_idx = len(observation)
        for end_marker in end_markers:
            marker_idx = observation.find(end_marker, start_idx)
            if marker_idx != -1:
                end_idx = min(end_idx, marker_idx)
        return observation[start_idx:end_idx].strip()
    return observation


def list_observations(obs: Dict[str, Any], *, prefer_anchor: bool) -> List[str]:
    key = "anchor" if prefer_anchor and "anchor" in obs else "text"
    return [str(item) for item in obs.get(key, [])]


def coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return coerce_bool(value.reshape(-1)[0].item(), default=default)
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return coerce_bool(value[0], default=default)
    return bool(value)


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    size = max(1, int(size))
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def align_rollout_specs_to_infos(specs, infos):
    """Bind requested rollout metadata to the actual games after TextWorld shuffles them."""
    from collections import defaultdict, deque
    by_game = defaultdict(deque)
    for spec in specs:
        by_game[str(Path(spec["task"]["game_file"]).resolve())].append(spec)
    aligned = []
    for info in infos:
        actual = info.get("extra.gamefile")
        if isinstance(actual, (list, tuple)) and len(actual) == 1:
            actual = actual[0]
        if not isinstance(actual, str):
            raise ValueError("ALFWorld reset did not provide a unique game identity")
        key = str(Path(actual).resolve())
        if not by_game[key]:
            raise ValueError(f"Unexpected or excess game returned by ALFWorld: {key}")
        aligned.append(by_game[key].popleft())
    if len(aligned) != len(specs) or any(by_game.values()):
        raise ValueError("ALFWorld reset did not preserve the requested game multiset")
    return aligned


def run_rollout_specs(
    *,
    specs: Sequence[Dict[str, Any]],
    seed: int,
    args: argparse.Namespace,
    policy_endpoint: ChatEndpoint,
) -> List[Dict[str, Any]]:
    if not specs:
        return []

    batch_size = len(specs)
    manager = build_manager(
        game_files=[spec["task"]["game_file"] for spec in specs],
        alf_config_path=args.alf_config,
        seed=seed,
        history_length=args.history_length,
    )
    agent = make_policy_agent(policy_endpoint, fallback_action=args.fallback_action)
    request_workers = max(1, min(args.request_workers, batch_size))
    trajectories: List[Dict[str, Any]] = []
    try:
        obs, infos = manager.reset({})
        specs = align_rollout_specs_to_infos(specs, infos)
        env_dones = [False] * batch_size
        success_flags = np.zeros(batch_size, dtype=bool)
        task_descriptions = list(getattr(manager, "tasks", [""] * batch_size))
        trajectories = [
            {
                "task_id": spec["task"]["task_id"],
                "task_type": spec["task"]["task_type"],
                "game_file": spec["task"]["game_file"],
                "source_skill_id": spec.get("source_skill_id"),
                "rollout_id": spec.get("rollout_id", env_idx),
                "seed": spec.get("seed", seed + env_idx),
                "episode_skill": spec.get("episode_skill", ""),
                "task_description": task_descriptions[env_idx] if env_idx < len(task_descriptions) else "",
                "initial_info": json_safe(infos[env_idx]),
                "steps": [],
            }
            for env_idx, spec in enumerate(specs)
        ]

        for step_idx in range(args.max_steps):
            current_prompts = list_observations(obs, prefer_anchor=False)
            current_observations = [
                extract_standard_alfworld_observation(item)
                for item in list_observations(obs, prefer_anchor=True)
            ]
            agent_observations = []
            for env_idx, observation_prompt in enumerate(current_prompts):
                env_skill = str(specs[env_idx].get("episode_skill", "")).strip()
                if env_skill:
                    agent_observations.append(
                        build_augmented_observation_text(
                            observation=observation_prompt,
                            episode_skill=env_skill,
                        )
                    )
                else:
                    agent_observations.append(observation_prompt)

            actions, model_responses, format_flags, error_flags = collect_actions_concurrently(
                agent=agent,
                observations=agent_observations,
                env_dones=env_dones,
                done_action="<think>The episode is done.</think><action>look</action>",
                request_workers=request_workers,
                log_responses=False,
                format_checker=has_action_format,
            )

            next_obs, rewards, dones, infos = manager.step(list(actions))
            next_prompts = list_observations(next_obs, prefer_anchor=False)
            next_observations = [
                extract_standard_alfworld_observation(item)
                for item in list_observations(next_obs, prefer_anchor=True)
            ]
            for env_idx in range(batch_size):
                if env_dones[env_idx]:
                    continue
                executed_action = extract_action_text(actions[env_idx])
                trajectories[env_idx]["steps"].append(
                    {
                        "step_idx": step_idx,
                        "observation": current_observations[env_idx],
                        "observation_prompt": current_prompts[env_idx],
                        "skill_augmented_observation": agent_observations[env_idx],
                        "model_response": model_responses[env_idx],
                        "raw_action_text": actions[env_idx],
                        "executed_action": executed_action,
                        "action_valid": coerce_bool(infos[env_idx].get("is_action_valid"), default=True),
                        "reward": json_safe(rewards[env_idx]),
                        "done": bool(dones[env_idx]),
                        "info": json_safe(infos[env_idx]),
                        "next_observation": next_observations[env_idx],
                        "next_observation_prompt": next_prompts[env_idx],
                    }
                )

            obs = next_obs
            for env_idx in range(batch_size):
                if env_dones[env_idx]:
                    continue
                if dones[env_idx]:
                    env_dones[env_idx] = True
                    success_flags[env_idx] = bool(infos[env_idx].get("won", False))
            if all(env_dones):
                break

        for env_idx, trajectory in enumerate(trajectories):
            trajectory["success"] = bool(success_flags[env_idx])
            trajectory["completed"] = bool(env_dones[env_idx])
            trajectory["num_steps"] = len(trajectory["steps"])
            trajectory["final_reward"] = (
                trajectory["steps"][-1]["reward"] if trajectory["steps"] else 0.0
            )
    finally:
        manager.close()
    return trajectories


def run_rollout_batch(
    *,
    task: Dict[str, Any],
    batch_size: int,
    seed: int,
    args: argparse.Namespace,
    policy_endpoint: ChatEndpoint,
    episode_skill: str = "",
    source_skill_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    specs = [
        {
            "task": task,
            "rollout_id": rollout_id,
            "seed": seed + rollout_id,
            "episode_skill": episode_skill,
            "source_skill_id": source_skill_id,
        }
        for rollout_id in range(batch_size)
    ]
    return run_rollout_specs(
        specs=specs,
        seed=seed,
        args=args,
        policy_endpoint=policy_endpoint,
    )


def collect_baseline_rollouts(
    *,
    tasks: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    policy_endpoint: ChatEndpoint,
) -> List[Dict[str, Any]]:
    path = output_dir / "baseline_rollouts.jsonl"
    existing = read_jsonl(path) if args.resume else []
    existing_keys = {
        (record["task_id"], int(record["rollout_id"]))
        for record in existing
    }
    records = list(existing)
    expected_total = len(tasks) * int(args.rollouts_per_task)
    log_stage(
        output_dir,
        "baseline_rollout",
        "running",
        existing_rollouts=len(records),
        expected_rollouts=expected_total,
        task_batch_size=int(args.task_batch_size),
        rollouts_per_task=int(args.rollouts_per_task),
    )

    pending_specs: List[Dict[str, Any]] = []
    for task_idx, task in enumerate(tasks):
        missing_rollout_ids = [
            rollout_id
            for rollout_id in range(args.rollouts_per_task)
            if (task["task_id"], rollout_id) not in existing_keys
        ]
        for rollout_id in missing_rollout_ids:
            pending_specs.append(
                {
                    "task": task,
                    "task_index": task_idx,
                    "rollout_id": rollout_id,
                    "seed": args.seed + task_idx * 1000 + rollout_id,
                }
            )

    if not pending_specs:
        log_stage(
            output_dir,
            "baseline_rollout",
            "complete",
            completed_rollouts=len(records),
            expected_rollouts=expected_total,
        )
        return records

    env_batch_size = max(1, int(args.task_batch_size)) * max(1, int(args.rollouts_per_task))
    total_waves = (len(pending_specs) + env_batch_size - 1) // env_batch_size
    for wave_idx, spec_chunk in enumerate(chunked(pending_specs, env_batch_size)):
        task_ids = sorted({spec["task"]["task_id"] for spec in spec_chunk})
        logging.info(
            "Baseline wave %d/%d: %d envs across %d task(s).",
            wave_idx + 1,
            total_waves,
            len(spec_chunk),
            len(task_ids),
        )
        update_progress(
            output_dir,
            stage="baseline_rollout",
            status="running",
            wave=wave_idx + 1,
            total_waves=total_waves,
            envs_in_wave=len(spec_chunk),
            tasks_in_wave=len(task_ids),
            completed_rollouts=len(records),
            expected_rollouts=expected_total,
        )
        trajectories = run_rollout_specs(
            specs=spec_chunk,
            seed=args.seed + wave_idx * 100_000,
            args=args,
            policy_endpoint=policy_endpoint,
        )
        for trajectory in trajectories:
            trajectory["baseline_key"] = f"{trajectory['task_id']}:{trajectory['rollout_id']}"
            append_jsonl(path, trajectory)
            records.append(trajectory)
        update_progress(
            output_dir,
            stage="baseline_rollout",
            status="running",
            wave=wave_idx + 1,
            total_waves=total_waves,
            completed_rollouts=len(records),
            expected_rollouts=expected_total,
        )
    log_stage(
        output_dir,
        "baseline_rollout",
        "complete",
        completed_rollouts=len(records),
        expected_rollouts=expected_total,
    )
    return records


def trajectory_to_seed_steps(trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = []
    for step in trajectory.get("steps", []):
        raw_observation = extract_standard_alfworld_observation(step.get("observation", ""))
        prompt_observation = (
            step.get("observation_prompt")
            or step.get("skill_augmented_observation")
            or step.get("observation", "")
        )
        step_info = step.get("info", {})
        action_valid = step.get("action_valid")
        if action_valid is None and isinstance(step_info, dict):
            action_valid = step_info.get("is_action_valid")
        steps.append(
            {
                "step_index": int(step.get("step_idx", len(steps))),
                "observation": raw_observation,
                "observation_prompt": prompt_observation,
                "response": step.get("model_response", ""),
                "action_valid": coerce_bool(action_valid, default=True),
                "task_description": trajectory.get("task_description", ""),
            }
        )
    return steps


def normalize_messages(prompt: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = prompt.get("messages", []) if isinstance(prompt, dict) else []
    normalized = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        normalized.append({"role": role, "content": content})
    if not normalized:
        normalized.append({"role": "user", "content": str(prompt)})
    return normalized


def build_candidate_skill_record(
    *,
    trajectory: Dict[str, Any],
    skill_endpoint: ChatEndpoint,
) -> Dict[str, Any]:
    from seed.analysis import SEEDEpisodeAnalyzer

    analyzer = SEEDEpisodeAnalyzer(
        backend="openai",
        max_completion_tokens=skill_endpoint.max_completion_tokens,
        max_step_skills_per_traj=0,
        skill_mode="episode_only",
    )
    skill_client = OpenAITextClient(skill_endpoint)
    skill_id = f"{trajectory['task_id']}:{trajectory['rollout_id']}"
    steps = trajectory_to_seed_steps(trajectory)
    candidate_step_indices = [step["step_index"] for step in steps]
    prompt = analyzer._build_episode_analysis_prompt(
        steps=steps,
        candidate_step_indices=candidate_step_indices,
        analysis_mode="teacher_bootstrap",
        episode_success=1.0 if trajectory.get("success") else 0.0,
        task_description=trajectory.get("task_description", ""),
    )
    raw_output, api_error = skill_client.complete(normalize_messages(prompt))
    parse_ok = False
    parsed: Dict[str, Any] = {"episode_summary": "", "episode_skill": ""}
    parse_error = None
    if raw_output:
        try:
            parsed = analyzer._parse_analysis_response(raw_output)
            parse_ok = bool(str(parsed.get("episode_skill", "")).strip())
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    return {
        "skill_id": skill_id,
        "task_id": trajectory["task_id"],
        "task_type": trajectory["task_type"],
        "game_file": trajectory["game_file"],
        "source_rollout_id": trajectory["rollout_id"],
        "source_success": bool(trajectory.get("success", False)),
        "source_num_steps": int(trajectory.get("num_steps", 0)),
        "task_description": trajectory.get("task_description", ""),
        "analysis_prompt": prompt,
        "llm_raw_output": raw_output,
        "episode_summary": parsed.get("episode_summary", ""),
        "episode_skill": parsed.get("episode_skill", ""),
        "parse_ok": parse_ok,
        "analysis_error": api_error or parse_error,
    }


def build_episode_only_analysis_prompt_from_trajectory(
    trajectory: Dict[str, Any],
    *,
    max_completion_tokens: int,
) -> Dict[str, Any]:
    from seed.analysis import SEEDEpisodeAnalyzer

    analyzer = SEEDEpisodeAnalyzer(
        backend="openai",
        max_completion_tokens=max_completion_tokens,
        max_step_skills_per_traj=0,
        skill_mode="episode_only",
    )
    steps = trajectory_to_seed_steps(trajectory)
    candidate_step_indices = [step["step_index"] for step in steps]
    return analyzer._build_episode_analysis_prompt(
        steps=steps,
        candidate_step_indices=candidate_step_indices,
        analysis_mode="teacher_bootstrap",
        episode_success=1.0 if trajectory.get("success") else 0.0,
        task_description=trajectory.get("task_description", ""),
    )


def generate_candidate_skills(
    *,
    baseline_rollouts: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    skill_endpoint: ChatEndpoint,
) -> List[Dict[str, Any]]:
    path = output_dir / "candidate_skills.jsonl"
    existing = read_jsonl(path) if args.resume else []
    existing_ids = {record["skill_id"] for record in existing}
    records = list(existing)
    rollouts = list(baseline_rollouts)
    if args.max_candidates is not None:
        rollouts = rollouts[: max(0, args.max_candidates)]
    expected_total = len(rollouts)
    log_stage(
        output_dir,
        "skill_generation",
        "running",
        existing_skills=len(records),
        expected_skills=expected_total,
        skill_gen_workers=int(args.skill_gen_workers),
    )
    pending_rollouts = [
        trajectory
        for trajectory in rollouts
        if f"{trajectory['task_id']}:{trajectory['rollout_id']}" not in existing_ids
    ]

    if not pending_rollouts:
        log_stage(
            output_dir,
            "skill_generation",
            "complete",
            completed_skills=len(records),
            expected_skills=expected_total,
        )
        return records

    max_workers = max(1, min(int(args.skill_gen_workers), len(pending_rollouts)))
    logging.info(
        "Generating %d candidate skill(s) with %d worker(s).",
        len(pending_rollouts),
        max_workers,
    )
    completed = 0
    update_progress(
        output_dir,
        stage="skill_generation",
        status="running",
        completed_skills=len(records),
        pending_skills=len(pending_rollouts),
        expected_skills=expected_total,
        skill_gen_workers=max_workers,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_trajectory = {
            executor.submit(
                build_candidate_skill_record,
                trajectory=trajectory,
                skill_endpoint=skill_endpoint,
            ): trajectory
            for trajectory in pending_rollouts
        }
        for future in as_completed(future_to_trajectory):
            trajectory = future_to_trajectory[future]
            skill_id = f"{trajectory['task_id']}:{trajectory['rollout_id']}"
            completed += 1
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    "skill_id": skill_id,
                    "task_id": trajectory["task_id"],
                    "task_type": trajectory["task_type"],
                    "game_file": trajectory["game_file"],
                    "source_rollout_id": trajectory["rollout_id"],
                    "source_success": bool(trajectory.get("success", False)),
                    "source_num_steps": int(trajectory.get("num_steps", 0)),
                    "task_description": trajectory.get("task_description", ""),
                    "analysis_prompt": None,
                    "llm_raw_output": "",
                    "episode_summary": "",
                    "episode_skill": "",
                    "parse_ok": False,
                    "analysis_error": f"{type(exc).__name__}: {exc}",
                }
            logging.info(
                "Generated skill %d/%d: %s parse_ok=%s.",
                completed,
                len(pending_rollouts),
                skill_id,
                record.get("parse_ok"),
            )
            append_jsonl(path, record)
            records.append(record)
            update_progress(
                output_dir,
                stage="skill_generation",
                status="running",
                completed_in_current_run=completed,
                pending_in_current_run=len(pending_rollouts),
                completed_skills=len(records),
                expected_skills=expected_total,
                last_skill_id=skill_id,
                last_parse_ok=record.get("parse_ok"),
            )
    log_stage(
        output_dir,
        "skill_generation",
        "complete",
        completed_skills=len(records),
        expected_skills=expected_total,
        parse_ok_skills=sum(1 for record in records if record.get("parse_ok")),
    )
    return records


def success_count(rollouts: Iterable[Dict[str, Any]]) -> int:
    return sum(1 for rollout in rollouts if bool(rollout.get("success", False)))


def validate_skills(
    *,
    tasks_by_id: Dict[str, Dict[str, Any]],
    baseline_rollouts: Sequence[Dict[str, Any]],
    candidate_skills: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    policy_endpoint: ChatEndpoint,
) -> List[Dict[str, Any]]:
    path = output_dir / "skill_validations.jsonl"
    accepted_path = output_dir / "accepted_skills.jsonl"
    rejected_path = output_dir / "rejected_skills.jsonl"
    existing = read_jsonl(path) if args.resume else []
    existing_ids = {record["skill_id"] for record in existing}
    validations = list(existing)
    log_stage(
        output_dir,
        "skill_validation",
        "running",
        existing_validations=len(validations),
        skill_batch_size=int(args.skill_batch_size),
        validation_rollouts=int(args.validation_rollouts),
    )
    baseline_by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rollout in baseline_rollouts:
        baseline_by_task[rollout["task_id"]].append(rollout)

    candidates = [candidate for candidate in candidate_skills if candidate.get("parse_ok")]
    if args.max_candidates is not None:
        candidates = candidates[: max(0, args.max_candidates)]
    candidates = [candidate for candidate in candidates if candidate["skill_id"] not in existing_ids]

    if not candidates:
        log_stage(
            output_dir,
            "skill_validation",
            "complete",
            completed_validations=len(validations),
            accepted_skills=sum(1 for record in validations if record.get("accepted")),
        )
        return validations

    total_waves = (len(candidates) + max(1, int(args.skill_batch_size)) - 1) // max(1, int(args.skill_batch_size))
    expected_total = len(validations) + len(candidates)
    for wave_idx, candidate_chunk in enumerate(chunked(candidates, max(1, int(args.skill_batch_size)))):
        rollout_specs: List[Dict[str, Any]] = []
        for candidate_idx, candidate in enumerate(candidate_chunk):
            task = tasks_by_id[candidate["task_id"]]
            for rollout_id in range(args.validation_rollouts):
                rollout_specs.append(
                    {
                        "task": task,
                        "rollout_id": rollout_id,
                        "seed": args.seed + 1_000_000 + wave_idx * 100_000 + candidate_idx * 1000 + rollout_id,
                        "episode_skill": str(candidate.get("episode_skill", "")),
                        "source_skill_id": candidate["skill_id"],
                    }
                )

        logging.info(
            "Validation wave %d/%d: %d envs across %d candidate skill(s).",
            wave_idx + 1,
            total_waves,
            len(rollout_specs),
            len(candidate_chunk),
        )
        update_progress(
            output_dir,
            stage="skill_validation",
            status="running",
            wave=wave_idx + 1,
            total_waves=total_waves,
            envs_in_wave=len(rollout_specs),
            skills_in_wave=len(candidate_chunk),
            completed_validations=len(validations),
            expected_validations=expected_total,
            accepted_skills=sum(1 for record in validations if record.get("accepted")),
        )
        wave_rollouts = run_rollout_specs(
            specs=rollout_specs,
            seed=args.seed + 1_000_000 + wave_idx * 100_000,
            args=args,
            policy_endpoint=policy_endpoint,
        )
        rollouts_by_skill: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rollout in wave_rollouts:
            rollouts_by_skill[str(rollout.get("source_skill_id", ""))].append(rollout)

        for candidate in candidate_chunk:
            skill_id = candidate["skill_id"]
            task_id = candidate["task_id"]
            skill_rollouts = rollouts_by_skill.get(skill_id, [])
            baseline_for_task = baseline_by_task.get(task_id, [])
            baseline_count = success_count(baseline_for_task)
            skill_count = success_count(skill_rollouts)
            baseline_total = max(1, len(baseline_for_task))
            skill_total = max(1, len(skill_rollouts))
            baseline_rate = baseline_count / baseline_total
            skill_rate = skill_count / skill_total
            delta_count = skill_count - baseline_count
            delta_rate = skill_rate - baseline_rate
            accepted = (
                delta_count >= args.accept_min_delta_count
                and delta_rate >= args.accept_min_delta_rate
            )
            record = {
                "skill_id": skill_id,
                "task_id": task_id,
                "task_type": candidate["task_type"],
                "episode_skill": candidate.get("episode_skill", ""),
                "episode_summary": candidate.get("episode_summary", ""),
                "baseline_success_count": baseline_count,
                "baseline_total": len(baseline_for_task),
                "baseline_success_rate": baseline_rate,
                "skill_success_count": skill_count,
                "skill_total": len(skill_rollouts),
                "skill_success_rate": skill_rate,
                "delta_success_count": delta_count,
                "delta_success_rate": delta_rate,
                "accepted": accepted,
                "skill_eval_rollouts": skill_rollouts,
                "candidate": candidate,
            }
            append_jsonl(path, record)
            append_jsonl(accepted_path if accepted else rejected_path, record)
            validations.append(record)
        update_progress(
            output_dir,
            stage="skill_validation",
            status="running",
            wave=wave_idx + 1,
            total_waves=total_waves,
            completed_validations=len(validations),
            expected_validations=expected_total,
            accepted_skills=sum(1 for record in validations if record.get("accepted")),
        )
    log_stage(
        output_dir,
        "skill_validation",
        "complete",
        completed_validations=len(validations),
        expected_validations=expected_total,
        accepted_skills=sum(1 for record in validations if record.get("accepted")),
    )
    return validations


def build_sft_exports(
    *,
    validations: Sequence[Dict[str, Any]],
    baseline_rollouts: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    accepted = [record for record in validations if record.get("accepted")]
    baseline_by_skill_id = {
        f"{trajectory['task_id']}:{trajectory['rollout_id']}": trajectory
        for trajectory in baseline_rollouts
    }
    sft_records = []
    for record in accepted:
        candidate = record.get("candidate", {})
        prompt = candidate.get("analysis_prompt", {})
        source_trajectory = baseline_by_skill_id.get(record["skill_id"])
        if source_trajectory is not None:
            prompt = build_episode_only_analysis_prompt_from_trajectory(
                source_trajectory,
                max_completion_tokens=args.skill_max_completion_tokens,
            )
        response_payload = {
            "episode_summary": candidate.get("episode_summary", ""),
            "episode_skill": candidate.get("episode_skill", ""),
        }
        sft_records.append(
            {
                "prompt": normalize_messages(prompt)[-1]["content"],
                "response": json.dumps(response_payload, ensure_ascii=False),
                "skill_id": record["skill_id"],
                "task_id": record["task_id"],
                "task_type": record["task_type"],
                "delta_success_count": record.get("delta_success_count", 0),
                "delta_success_rate": record.get("delta_success_rate", 0.0),
            }
        )

    all_jsonl = output_dir / "sft_episode_skill_all.jsonl"
    if all_jsonl.exists():
        all_jsonl.unlink()
    for record in sft_records:
        append_jsonl(all_jsonl, record)

    if not sft_records:
        logging.warning("No accepted skills; SFT parquet export skipped.")
        return sft_records

    rng = random.Random(args.seed)
    shuffled = list(sft_records)
    rng.shuffle(shuffled)
    val_size = int(round(len(shuffled) * args.sft_val_ratio))
    if len(shuffled) > 1:
        val_size = max(1, min(val_size, len(shuffled) - 1))
    train_records = shuffled[val_size:]
    val_records = shuffled[:val_size]

    try:
        import pandas as pd

        pd.DataFrame(train_records).to_parquet(output_dir / "sft_episode_skill_train.parquet")
        pd.DataFrame(val_records).to_parquet(output_dir / "sft_episode_skill_val.parquet")
    except Exception as exc:
        logging.warning("Could not write parquet SFT exports: %s", exc)
    return sft_records


def write_metrics(
    *,
    tasks: Sequence[Dict[str, Any]],
    baseline_rollouts: Sequence[Dict[str, Any]],
    candidate_skills: Sequence[Dict[str, Any]],
    validations: Sequence[Dict[str, Any]],
    sft_records: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> None:
    accepted = [record for record in validations if record.get("accepted")]
    by_type = {}
    for task_type in TASK_TYPES:
        task_candidates = [record for record in candidate_skills if record.get("task_type") == task_type]
        task_validations = [record for record in validations if record.get("task_type") == task_type]
        task_accepted = [record for record in accepted if record.get("task_type") == task_type]
        by_type[task_type] = {
            "tasks": sum(1 for task in tasks if task.get("task_type") == task_type),
            "baseline_rollouts": sum(1 for rollout in baseline_rollouts if rollout.get("task_type") == task_type),
            "candidate_skills": len(task_candidates),
            "parse_ok": sum(1 for record in task_candidates if record.get("parse_ok")),
            "validated_skills": len(task_validations),
            "accepted_skills": len(task_accepted),
        }

    baseline_successes = [float(bool(record.get("success"))) for record in baseline_rollouts]
    validation_skill_rates = [float(record.get("skill_success_rate", 0.0)) for record in validations]
    validation_delta_rates = [float(record.get("delta_success_rate", 0.0)) for record in validations]
    metrics = {
        "tasks": len(tasks),
        "baseline_rollouts": len(baseline_rollouts),
        "baseline_success_rate": float(np.mean(baseline_successes)) if baseline_successes else 0.0,
        "candidate_skills": len(candidate_skills),
        "parse_ok_skills": sum(1 for record in candidate_skills if record.get("parse_ok")),
        "validated_skills": len(validations),
        "accepted_skills": len(accepted),
        "accepted_rate": len(accepted) / max(1, len(validations)),
        "validation_skill_success_rate_mean": float(np.mean(validation_skill_rates)) if validation_skill_rates else 0.0,
        "validation_delta_success_rate_mean": float(np.mean(validation_delta_rates)) if validation_delta_rates else 0.0,
        "sft_records": len(sft_records),
        "accepted_by_task_type": dict(Counter(record["task_type"] for record in accepted)),
        "by_task_type": by_type,
    }
    write_json(output_dir / "metrics.json", metrics)


def write_run_config(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    policy_endpoint: ChatEndpoint,
    skill_endpoint: ChatEndpoint,
) -> None:
    redacted_args = vars(args).copy()
    for key in ("policy_api_key", "skill_api_key"):
        if redacted_args.get(key):
            redacted_args[key] = "<redacted>"
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "project_root": str(PROJECT_ROOT),
        "argv": sys.argv,
        "args": redacted_args,
        "policy_endpoint": {
            "base_url": policy_endpoint.base_url,
            "api_key": "<redacted>" if policy_endpoint.api_key else None,
            "model": policy_endpoint.model,
            "temperature": policy_endpoint.temperature,
            "max_completion_tokens": policy_endpoint.max_completion_tokens,
        },
        "skill_endpoint": {
            "base_url": skill_endpoint.base_url,
            "api_key": "<redacted>" if skill_endpoint.api_key else None,
            "model": skill_endpoint.model,
            "temperature": skill_endpoint.temperature,
            "max_completion_tokens": skill_endpoint.max_completion_tokens,
        },
    }
    write_json(output_dir / "run_config.json", payload)


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, resume=args.resume, overwrite=args.overwrite)
    setup_logging(output_dir, args.log_level)

    policy_endpoint = resolve_endpoint(
        prefix="policy",
        args=args,
        default_base_url_env="OPENAI_BASE_URL",
        default_model_env="OPENAI_MODEL",
        default_model="Qwen2.5-3B-Instruct",
        temperature=args.policy_temperature,
        max_completion_tokens=args.policy_max_completion_tokens,
        timeout=args.policy_timeout,
        retries=args.policy_retries,
        retry_delay=args.policy_retry_delay,
        extra_body_json=args.policy_extra_body_json,
    )
    skill_endpoint = resolve_endpoint(
        prefix="skill",
        args=args,
        default_base_url_env="SKILL_OPENAI_BASE_URL",
        default_model_env="SKILL_OPENAI_MODEL",
        default_model=policy_endpoint.model,
        temperature=args.skill_temperature,
        max_completion_tokens=args.skill_max_completion_tokens,
        timeout=args.skill_timeout,
        retries=args.skill_retries,
        retry_delay=args.skill_retry_delay,
        extra_body_json=args.skill_extra_body_json,
    )
    write_run_config(
        args=args,
        output_dir=output_dir,
        policy_endpoint=policy_endpoint,
        skill_endpoint=skill_endpoint,
    )

    logging.info("Using policy model %s at %s.", policy_endpoint.model, policy_endpoint.base_url)
    logging.info("Using skill model %s at %s.", skill_endpoint.model, skill_endpoint.base_url)

    log_stage(
        output_dir,
        "task_sampling",
        "running",
        tasks_per_type=int(args.tasks_per_type),
    )
    tasks = sample_tasks(args, output_dir)
    log_stage(
        output_dir,
        "task_sampling",
        "complete",
        sampled_tasks=len(tasks),
    )
    baseline_rollouts = collect_baseline_rollouts(
        tasks=tasks,
        args=args,
        output_dir=output_dir,
        policy_endpoint=policy_endpoint,
    )
    candidate_skills = generate_candidate_skills(
        baseline_rollouts=baseline_rollouts,
        args=args,
        output_dir=output_dir,
        skill_endpoint=skill_endpoint,
    )
    tasks_by_id = {task["task_id"]: task for task in tasks}
    validations = validate_skills(
        tasks_by_id=tasks_by_id,
        baseline_rollouts=baseline_rollouts,
        candidate_skills=candidate_skills,
        args=args,
        output_dir=output_dir,
        policy_endpoint=policy_endpoint,
    )
    log_stage(
        output_dir,
        "sft_export",
        "running",
        validations=len(validations),
        accepted_skills=sum(1 for record in validations if record.get("accepted")),
    )
    sft_records = build_sft_exports(
        validations=validations,
        baseline_rollouts=baseline_rollouts,
        args=args,
        output_dir=output_dir,
    )
    log_stage(
        output_dir,
        "sft_export",
        "complete",
        sft_records=len(sft_records),
    )
    log_stage(output_dir, "metrics", "running")
    write_metrics(
        tasks=tasks,
        baseline_rollouts=baseline_rollouts,
        candidate_skills=candidate_skills,
        validations=validations,
        sft_records=sft_records,
        output_dir=output_dir,
    )
    log_stage(
        output_dir,
        "complete",
        "complete",
        tasks=len(tasks),
        baseline_rollouts=len(baseline_rollouts),
        candidate_skills=len(candidate_skills),
        validations=len(validations),
        sft_records=len(sft_records),
    )
    logging.info("Pipeline complete. Outputs are in %s.", output_dir)


if __name__ == "__main__":
    main()
