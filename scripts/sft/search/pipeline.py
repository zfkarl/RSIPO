#!/usr/bin/env python3
"""Generate episode-skill SFT data from Search QA policy rollouts.

This mirrors the ALFWorld/WebShop flow: sample Search-R1 train examples,
collect baseline rollouts with a local/OpenAI-compatible policy model, ask an
LLM to produce episode-level skills using the current SEED analyzer prompt, and
export every parseable candidate directly as SFT data.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)

from examples.prompt_agent.local_vllm_alfworld import (  # noqa: E402
    AttrDict,
    json_safe,
    load_env_file,
)
from seed.prompting import build_augmented_observation_text  # noqa: E402
from scripts.sft._common.pipeline import (  # noqa: E402
    ChatEndpoint,
    OpenAITextClient,
    append_jsonl,
    coerce_bool,
    log_stage,
    normalize_messages,
    read_jsonl,
    resolve_endpoint,
    setup_logging,
    update_progress,
    write_json,
)


OUTPUT_FILES = [
    "sampled_tasks.jsonl",
    "baseline_rollouts.jsonl",
    "candidate_skills.jsonl",
    "sft_episode_skill_all.jsonl",
    "sft_episode_skill_train.parquet",
    "sft_episode_skill_val.parquet",
    "metrics.json",
    "progress.json",
    "run_config.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output-dir", default="outputs/search_episode_skill_pipeline")
    parser.add_argument("--train-data", default="~/data/searchR1_processed_direct/train.parquet")
    parser.add_argument("--num-tasks", type=int, default=180)
    parser.add_argument("--rollouts-per-task", type=int, default=8)
    parser.add_argument(
        "--task-batch-size",
        type=int,
        default=16,
        help="Number of different QA questions to rollout in the same wave.",
    )
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--request-workers", type=int, default=128)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--regenerate-candidates",
        action="store_true",
        help="Delete existing candidate/SFT outputs before generating candidates.",
    )
    parser.add_argument(
        "--stop-after-baseline-rollouts",
        action="store_true",
        help="Exit successfully after baseline rollouts are complete.",
    )
    parser.add_argument(
        "--stop-after-skill-generation",
        action="store_true",
        help="Exit successfully after candidate skill API generation is complete.",
    )
    parser.add_argument("--log-level", default="INFO")

    parser.add_argument("--search-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--search-topk", type=int, default=3)
    parser.add_argument("--search-timeout", type=float, default=60.0)
    parser.add_argument("--search-log-requests", action="store_true")

    parser.add_argument("--policy-base-url", default=None)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--policy-model", default=None)
    parser.add_argument("--policy-temperature", type=float, default=0.4)
    parser.add_argument("--policy-max-completion-tokens", type=int, default=512)
    parser.add_argument("--policy-timeout", type=float, default=120.0)
    parser.add_argument("--policy-retries", type=int, default=2)
    parser.add_argument("--policy-retry-delay", type=float, default=1.0)
    parser.add_argument("--policy-extra-body-json", default=None)
    parser.add_argument("--fallback-answer", default="I don't know")

    parser.add_argument("--skill-base-url", default=None)
    parser.add_argument("--skill-api-key", default=None)
    parser.add_argument("--skill-model", default=None)
    parser.add_argument("--skill-temperature", type=float, default=0.0)
    parser.add_argument("--skill-max-completion-tokens", type=int, default=1024)
    parser.add_argument("--skill-timeout", type=float, default=120.0)
    parser.add_argument("--skill-retries", type=int, default=5)
    parser.add_argument("--skill-retry-delay", type=float, default=1.0)
    parser.add_argument("--skill-extra-body-json", default=None)
    parser.add_argument("--skill-gen-workers", type=int, default=128)
    parser.add_argument(
        "--skill-prompt-version",
        default="seed",
        choices=("seed",),
        help="Episode-skill prompt version.",
    )
    parser.add_argument("--sft-val-ratio", type=float, default=0.1)
    return parser.parse_args()


def normalize_search_skill_prompt_version(version: object) -> str:
    return str(version or "seed").strip()


def prepare_output_dir(
    output_dir: Path,
    *,
    overwrite: bool,
    resume: bool,
    regenerate_candidates: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if overwrite:
        for path in existing:
            path.unlink()
        return
    if regenerate_candidates:
        for name in (
            "candidate_skills.jsonl",
            "sft_episode_skill_all.jsonl",
            "sft_episode_skill_train.parquet",
            "sft_episode_skill_val.parquet",
            "metrics.json",
        ):
            path = output_dir / name
            if path.exists():
                path.unlink()
        return
    if existing and not resume:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Output files already exist in {output_dir}: {names}. "
            "Use --resume, --overwrite, or --regenerate-candidates."
        )


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    size = max(1, int(size))
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def clean_for_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [clean_for_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): clean_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_for_json(item) for item in value]
    return value


def extract_question(row: Dict[str, Any]) -> str:
    env_kwargs = row.get("env_kwargs")
    if isinstance(env_kwargs, dict) and env_kwargs.get("question"):
        return str(env_kwargs["question"])
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict) and extra_info.get("question"):
        return str(extra_info["question"])
    prompt = row.get("prompt")
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
    return str(row.get("question", ""))


def extract_ground_truth(row: Dict[str, Any]) -> Dict[str, Any]:
    env_kwargs = row.get("env_kwargs")
    if isinstance(env_kwargs, dict) and "ground_truth" in env_kwargs:
        return clean_for_json(env_kwargs["ground_truth"])
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and "ground_truth" in reward_model:
        return clean_for_json(reward_model["ground_truth"])
    return {"target": clean_for_json(row.get("golden_answers", []))}


def sample_tasks(args: argparse.Namespace, output_dir: Path) -> List[Dict[str, Any]]:
    sampled_path = output_dir / "sampled_tasks.jsonl"
    existing = read_jsonl(sampled_path)
    if existing and args.resume:
        return existing

    import pandas as pd

    train_path = Path(args.train_data).expanduser()
    if not train_path.exists():
        raise FileNotFoundError(f"Search train parquet not found: {train_path}")
    df = pd.read_parquet(train_path)
    if len(df) < int(args.num_tasks):
        raise ValueError(f"Need {args.num_tasks} Search examples, found {len(df)} in {train_path}.")

    indices = list(range(len(df)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    selected = indices[: int(args.num_tasks)]
    sampled: List[Dict[str, Any]] = []
    for sample_idx, row_idx in enumerate(selected):
        row = df.iloc[row_idx].to_dict()
        question = extract_question(row)
        ground_truth = extract_ground_truth(row)
        data_source = str(row.get("data_source") or "unknown")
        sampled.append(
            {
                "task_id": f"search_{row_idx:06d}",
                "task_type": "search_qa",
                "row_index": int(row_idx),
                "sample_index": int(sample_idx),
                "split": "train",
                "data_source": data_source,
                "task_description": question,
                "question": question,
                "ground_truth": ground_truth,
                "env_kwargs": {
                    "question": question,
                    "ground_truth": ground_truth,
                    "data_source": data_source,
                },
            }
        )

    if args.max_tasks is not None:
        sampled = sampled[: max(0, int(args.max_tasks))]

    if sampled_path.exists():
        sampled_path.unlink()
    for record in sampled:
        append_jsonl(sampled_path, record)
    logging.info("Sampled %d Search QA train examples.", len(sampled))
    return sampled


class SearchPolicyAgent:
    def __init__(self, *, endpoint: ChatEndpoint, fallback_answer: str):
        self.client = OpenAITextClient(endpoint)
        self.fallback_answer = fallback_answer

    def fallback(self) -> str:
        return (
            "<think>The model call failed, so I will provide a conservative final answer.</think>\n"
            f"<answer>{self.fallback_answer}</answer>"
        )

    def get_action(self, observation: str) -> Tuple[str, bool, Optional[str]]:
        text, error = self.client.complete([{"role": "user", "content": observation}])
        text = str(text or "").strip()
        if error or not text:
            fallback = self.fallback()
            return fallback, has_search_action_format(fallback), error or "empty response"
        return text, has_search_action_format(text), None


def has_search_action_format(text: str) -> bool:
    lowered = str(text or "").lower()
    has_search = "<search>" in lowered and "</search>" in lowered
    has_answer = "<answer>" in lowered and "</answer>" in lowered
    return has_search ^ has_answer


def collect_search_actions_concurrently(
    *,
    agent: SearchPolicyAgent,
    observations: List[str],
    env_dones: List[bool],
    request_workers: int,
) -> Tuple[List[str], List[str], List[float], List[float]]:
    done_action = "<answer>done</answer>"
    actions = [done_action for _ in env_dones]
    model_responses = [done_action for _ in env_dones]
    format_flags: List[float] = []
    api_error_flags: List[float] = []
    active_indices = [idx for idx, done in enumerate(env_dones) if not done]
    if not active_indices:
        return actions, model_responses, format_flags, api_error_flags

    max_workers = max(1, min(int(request_workers), len(active_indices)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(agent.get_action, observations[idx]): idx
            for idx in active_indices
        }
        for future in as_completed(future_to_idx):
            env_idx = future_to_idx[future]
            try:
                action, format_ok, error = future.result()
            except Exception as exc:
                action = agent.fallback()
                format_ok = has_search_action_format(action)
                error = f"{type(exc).__name__}: {exc}"

            actions[env_idx] = action
            model_responses[env_idx] = action
            format_flags.append(float(format_ok))
            api_error_flags.append(float(error is not None))
            if error:
                logging.warning("Search env %d API fallback: %s", env_idx, error)

    return actions, model_responses, format_flags, api_error_flags


def build_manager(*, batch_size: int, seed: int, args: argparse.Namespace) -> Any:
    from agent_system.environments.env_manager import SearchEnvironmentManager
    from agent_system.environments.env_package.search import build_search_envs, search_projection

    config = AttrDict(
        {
            "env": AttrDict(
                {
                    "env_name": "search",
                    "history_length": int(args.history_length),
                    "max_steps": int(args.max_steps),
                    "use_skills_only_memory": False,
                    "use_retrieval_memory": False,
                    "search": AttrDict(
                        {
                            "search_url": args.search_url,
                            "topk": int(args.search_topk),
                            "timeout": float(args.search_timeout),
                            "log_requests": bool(args.search_log_requests),
                        }
                    ),
                }
            )
        }
    )
    envs = build_search_envs(
        seed=int(seed),
        env_num=int(batch_size),
        group_n=1,
        is_train=False,
        env_config=config.env,
    )
    return SearchEnvironmentManager(envs, search_projection, config)


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
    manager = build_manager(batch_size=batch_size, seed=seed, args=args)
    agent = SearchPolicyAgent(endpoint=policy_endpoint, fallback_answer=args.fallback_answer)
    request_workers = max(1, min(int(args.request_workers), batch_size))
    trajectories: List[Dict[str, Any]] = []
    try:
        reset_kwargs = [spec["task"]["env_kwargs"] for spec in specs]
        obs, infos = manager.reset(reset_kwargs)
        env_dones = [False] * batch_size
        success_flags = np.zeros(batch_size, dtype=bool)
        final_scores = np.zeros(batch_size, dtype=float)
        task_descriptions = list(getattr(manager, "tasks", [""] * batch_size))
        trajectories = [
            {
                "task_id": spec["task"]["task_id"],
                "task_type": spec["task"].get("task_type", "search_qa"),
                "row_index": int(spec["task"]["row_index"]),
                "source_skill_id": spec.get("source_skill_id"),
                "rollout_id": int(spec.get("rollout_id", env_idx)),
                "seed": int(spec.get("seed", seed + env_idx)),
                "episode_skill": spec.get("episode_skill", ""),
                "task_description": (
                    task_descriptions[env_idx]
                    if env_idx < len(task_descriptions) and task_descriptions[env_idx]
                    else spec["task"].get("task_description", "")
                ),
                "question": spec["task"].get("question", ""),
                "ground_truth": spec["task"].get("ground_truth", {}),
                "data_source": spec["task"].get("data_source", "unknown"),
                "initial_info": json_safe(infos[env_idx]),
                "steps": [],
            }
            for env_idx, spec in enumerate(specs)
        ]

        for step_idx in range(int(args.max_steps)):
            current_prompts = [str(item) for item in obs.get("text", [])]
            current_observations = [str(item) for item in obs.get("anchor", current_prompts)]
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

            actions, model_responses, _format_flags, _error_flags = collect_search_actions_concurrently(
                agent=agent,
                observations=agent_observations,
                env_dones=env_dones,
                request_workers=request_workers,
            )

            next_obs, rewards, dones, infos = manager.step(list(actions))
            next_prompts = [str(item) for item in next_obs.get("text", [])]
            next_observations = [str(item) for item in next_obs.get("anchor", next_prompts)]

            for env_idx in range(batch_size):
                if env_dones[env_idx]:
                    continue
                reward = float(rewards[env_idx])
                final_scores[env_idx] = reward
                action_valid = coerce_bool(infos[env_idx].get("is_action_valid"), default=False)
                trajectories[env_idx]["steps"].append(
                    {
                        "step_idx": int(step_idx),
                        "observation": current_observations[env_idx],
                        "observation_prompt": current_prompts[env_idx],
                        "skill_augmented_observation": agent_observations[env_idx],
                        "model_response": model_responses[env_idx],
                        "raw_action_text": actions[env_idx],
                        "executed_action": str(infos[env_idx].get("postprocessed_action", actions[env_idx])),
                        "action_valid": action_valid,
                        "reward": reward,
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
                    success_flags[env_idx] = bool(infos[env_idx].get("won", False)) or float(rewards[env_idx]) >= 1.0
            if all(env_dones):
                break

        for env_idx, trajectory in enumerate(trajectories):
            trajectory["success"] = bool(success_flags[env_idx])
            trajectory["completed"] = bool(env_dones[env_idx])
            trajectory["num_steps"] = len(trajectory["steps"])
            trajectory["final_reward"] = (
                trajectory["steps"][-1]["reward"] if trajectory["steps"] else 0.0
            )
            trajectory["final_task_score"] = float(final_scores[env_idx])
    finally:
        manager.close()
    return trajectories


def collect_baseline_rollouts(
    *,
    tasks: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    policy_endpoint: ChatEndpoint,
) -> List[Dict[str, Any]]:
    path = output_dir / "baseline_rollouts.jsonl"
    existing = read_jsonl(path) if args.resume else []
    existing_keys = {(record["task_id"], int(record["rollout_id"])) for record in existing}
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
        for rollout_id in range(int(args.rollouts_per_task)):
            if (task["task_id"], rollout_id) in existing_keys:
                continue
            pending_specs.append(
                {
                    "task": task,
                    "task_index": int(task_idx),
                    "rollout_id": int(rollout_id),
                    "seed": int(args.seed) + task_idx * 1000 + rollout_id,
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
            "Baseline wave %d/%d: %d envs across %d Search QA question(s).",
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
            seed=int(args.seed) + wave_idx * 100_000,
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
    steps: List[Dict[str, Any]] = []
    for step in trajectory.get("steps", []):
        step_info = step.get("info", {})
        action_valid = step.get("action_valid")
        if action_valid is None and isinstance(step_info, dict):
            action_valid = step_info.get("is_action_valid")
        steps.append(
            {
                "step_index": int(step.get("step_idx", len(steps))),
                "observation": str(step.get("observation", "")),
                "observation_prompt": str(
                    step.get("observation_prompt")
                    or step.get("skill_augmented_observation")
                    or step.get("observation", "")
                ),
                "response": str(step.get("model_response", "")),
                "action_valid": coerce_bool(action_valid, default=False),
                "task_description": trajectory.get("task_description", ""),
            }
        )
    return steps


def build_candidate_skill_record(
    *,
    trajectory: Dict[str, Any],
    skill_endpoint: ChatEndpoint,
    skill_prompt_version: str,
) -> Dict[str, Any]:
    from seed.analysis import SEEDEpisodeAnalyzer

    analyzer = SEEDEpisodeAnalyzer(
        backend="openai",
        max_completion_tokens=skill_endpoint.max_completion_tokens,
        max_step_skills_per_traj=0,
        skill_mode="episode_only",
        analysis_prompt_version=skill_prompt_version,
    )
    skill_client = OpenAITextClient(skill_endpoint)
    skill_id = f"{trajectory['task_id']}:{trajectory['rollout_id']}"
    steps = trajectory_to_seed_steps(trajectory)
    prompt = analyzer._build_episode_analysis_prompt(
        steps=steps,
        candidate_step_indices=[step["step_index"] for step in steps],
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
        "task_type": trajectory.get("task_type", "search_qa"),
        "row_index": int(trajectory.get("row_index", -1)),
        "data_source": trajectory.get("data_source", "unknown"),
        "source_rollout_id": int(trajectory["rollout_id"]),
        "source_success": bool(trajectory.get("success", False)),
        "source_num_steps": int(trajectory.get("num_steps", 0)),
        "source_final_task_score": float(trajectory.get("final_task_score", 0.0)),
        "task_description": trajectory.get("task_description", ""),
        "analysis_prompt_version": skill_prompt_version,
        "analysis_prompt": prompt,
        "llm_raw_output": raw_output,
        "episode_summary": str(parsed.get("episode_summary", "")),
        "episode_skill": str(parsed.get("episode_skill", "")),
        "parse_ok": parse_ok,
        "analysis_error": api_error or parse_error,
    }


def generate_candidate_skills(
    *,
    baseline_rollouts: Sequence[Dict[str, Any]],
    output_dir: Path,
    skill_endpoint: ChatEndpoint,
    skill_prompt_version: str,
    max_candidates: Optional[int],
    max_workers: int,
    resume: bool,
) -> List[Dict[str, Any]]:
    path = output_dir / "candidate_skills.jsonl"
    existing = read_jsonl(path) if resume else []
    existing_ids = {record["skill_id"] for record in existing}
    records = list(existing)
    rollouts = list(baseline_rollouts)
    if max_candidates is not None:
        rollouts = rollouts[: max(0, int(max_candidates))]

    pending = [
        trajectory
        for trajectory in rollouts
        if f"{trajectory['task_id']}:{trajectory['rollout_id']}" not in existing_ids
    ]
    expected_total = len(rollouts)
    log_stage(
        output_dir,
        "skill_generation",
        "running",
        existing_skills=len(existing),
        pending_skills=len(pending),
        expected_skills=expected_total,
        skill_prompt_version=skill_prompt_version,
        skill_gen_workers=int(max_workers),
    )
    if not pending:
        log_stage(
            output_dir,
            "skill_generation",
            "complete",
            completed_skills=len(records),
            expected_skills=expected_total,
            parse_ok_skills=sum(1 for record in records if record.get("parse_ok")),
        )
        return records

    worker_count = max(1, min(int(max_workers), len(pending)))
    completed = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_trajectory = {
            executor.submit(
                build_candidate_skill_record,
                trajectory=trajectory,
                skill_endpoint=skill_endpoint,
                skill_prompt_version=skill_prompt_version,
            ): trajectory
            for trajectory in pending
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
                    "task_type": trajectory.get("task_type", "search_qa"),
                    "row_index": int(trajectory.get("row_index", -1)),
                    "data_source": trajectory.get("data_source", "unknown"),
                    "source_rollout_id": int(trajectory["rollout_id"]),
                    "source_success": bool(trajectory.get("success", False)),
                    "source_num_steps": int(trajectory.get("num_steps", 0)),
                    "source_final_task_score": float(trajectory.get("final_task_score", 0.0)),
                    "task_description": trajectory.get("task_description", ""),
                    "analysis_prompt_version": skill_prompt_version,
                    "analysis_prompt": None,
                    "llm_raw_output": "",
                    "episode_summary": "",
                    "episode_skill": "",
                    "parse_ok": False,
                    "analysis_error": f"{type(exc).__name__}: {exc}",
                }
            append_jsonl(path, record)
            records.append(record)
            logging.info(
                "Generated candidate skill %d/%d: %s parse_ok=%s.",
                completed,
                len(pending),
                skill_id,
                record.get("parse_ok"),
            )
            update_progress(
                output_dir,
                stage="skill_generation",
                status="running",
                completed_in_current_run=completed,
                pending_in_current_run=len(pending),
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


def build_sft_exports_from_candidates(
    *,
    candidate_skills: Sequence[Dict[str, Any]],
    output_dir: Path,
    sft_val_ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    sft_records: List[Dict[str, Any]] = []
    for candidate in candidate_skills:
        if not candidate.get("parse_ok"):
            continue
        prompt = candidate.get("analysis_prompt", {})
        messages = prompt.get("messages", []) if isinstance(prompt, dict) else []
        if not messages:
            continue
        analysis_prompt_version = str(candidate.get("analysis_prompt_version", "seed"))
        response_payload = {
            "episode_summary": candidate.get("episode_summary", ""),
            "episode_skill": candidate.get("episode_skill", ""),
        }
        sft_records.append(
            {
                "prompt": str(messages[-1].get("content", "")),
                "response": json.dumps(response_payload, ensure_ascii=False),
                "skill_id": candidate["skill_id"],
                "task_id": candidate["task_id"],
                "task_type": candidate.get("task_type", "search_qa"),
                "row_index": int(candidate.get("row_index", -1)),
                "data_source": candidate.get("data_source", "unknown"),
                "source_success": bool(candidate.get("source_success", False)),
                "source_num_steps": int(candidate.get("source_num_steps", 0)),
                "source_final_task_score": float(candidate.get("source_final_task_score", 0.0)),
                "analysis_prompt_version": analysis_prompt_version,
                "parse_ok": bool(candidate.get("parse_ok")),
            }
        )

    all_jsonl = output_dir / "sft_episode_skill_all.jsonl"
    if all_jsonl.exists():
        all_jsonl.unlink()
    for record in sft_records:
        append_jsonl(all_jsonl, record)

    if not sft_records:
        logging.warning("No parseable candidate skills; SFT parquet export skipped.")
        return sft_records

    shuffled = list(sft_records)
    random.Random(seed).shuffle(shuffled)
    val_size = int(round(len(shuffled) * sft_val_ratio))
    if len(shuffled) > 1:
        val_size = max(1, min(val_size, len(shuffled) - 1))
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]

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
    sft_records: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> None:
    final_scores = [float(record.get("final_task_score", 0.0)) for record in baseline_rollouts]
    payload = {
        "version": "search_no_skill_validation",
        "sampled_tasks": len(tasks),
        "baseline_rollouts": len(baseline_rollouts),
        "candidate_skills": len(candidate_skills),
        "parse_ok_skills": sum(1 for record in candidate_skills if record.get("parse_ok")),
        "parse_error_skills": sum(1 for record in candidate_skills if not record.get("parse_ok")),
        "sft_records": len(sft_records),
        "candidate_by_prompt_version": dict(
            Counter(record.get("analysis_prompt_version", "unknown") for record in candidate_skills)
        ),
        "candidate_by_task_type": dict(Counter(record.get("task_type", "unknown") for record in candidate_skills)),
        "candidate_by_data_source": dict(Counter(record.get("data_source", "unknown") for record in candidate_skills)),
        "sft_by_task_type": dict(Counter(record.get("task_type", "unknown") for record in sft_records)),
        "sft_by_data_source": dict(Counter(record.get("data_source", "unknown") for record in sft_records)),
        "source_success_counts": dict(Counter(str(bool(record.get("source_success", False))) for record in candidate_skills)),
        "baseline_success_rate": (
            float(np.mean([bool(record.get("success", False)) for record in baseline_rollouts]))
            if baseline_rollouts
            else 0.0
        ),
        "baseline_final_task_score_mean": float(np.mean(final_scores)) if final_scores else 0.0,
    }
    write_json(output_dir / "metrics.json", payload)


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
        "version": "search_no_skill_validation",
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
    args.skill_prompt_version = normalize_search_skill_prompt_version(args.skill_prompt_version)
    load_env_file(args.env_file)
    output_dir = Path(args.output_dir)
    prepare_output_dir(
        output_dir,
        overwrite=args.overwrite,
        resume=args.resume,
        regenerate_candidates=args.regenerate_candidates,
    )
    setup_logging(output_dir, args.log_level)

    policy_endpoint = resolve_endpoint(
        prefix="policy",
        args=args,
        default_base_url_env="POLICY_OPENAI_BASE_URL",
        default_model_env="POLICY_OPENAI_MODEL",
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
        default_model="Qwen2.5-3B-Instruct",
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

    log_stage(output_dir, "task_sampling", "running")
    tasks = sample_tasks(args, output_dir)
    log_stage(
        output_dir,
        "task_sampling",
        "complete",
        sampled_tasks=len(tasks),
        rollouts_per_task=int(args.rollouts_per_task),
    )

    baseline_rollouts = collect_baseline_rollouts(
        tasks=tasks,
        args=args,
        output_dir=output_dir,
        policy_endpoint=policy_endpoint,
    )
    if args.stop_after_baseline_rollouts:
        log_stage(
            output_dir,
            "baseline_rollout",
            "complete",
            completed_rollouts=len(baseline_rollouts),
            stopped_before_skill_generation=True,
        )
        logging.info("Baseline rollouts complete; stopping before skill API generation.")
        return

    baseline_rollouts_for_generation = (
        baseline_rollouts[: max(0, int(args.max_candidates))]
        if args.max_candidates is not None
        else baseline_rollouts
    )

    candidate_skills = generate_candidate_skills(
        baseline_rollouts=baseline_rollouts_for_generation,
        output_dir=output_dir,
        skill_endpoint=skill_endpoint,
        skill_prompt_version=args.skill_prompt_version,
        max_candidates=None,
        max_workers=args.skill_gen_workers,
        resume=args.resume and not args.regenerate_candidates,
    )
    if args.stop_after_skill_generation:
        log_stage(
            output_dir,
            "skill_generation",
            "complete",
            completed_skills=len(candidate_skills),
            parse_ok_skills=sum(1 for record in candidate_skills if record.get("parse_ok")),
            stopped_before_sft_export=True,
        )
        logging.info("Candidate skill API generation complete; stopping before SFT export.")
        return

    log_stage(output_dir, "sft_export", "running", candidate_skills=len(candidate_skills))
    sft_records = build_sft_exports_from_candidates(
        candidate_skills=candidate_skills,
        output_dir=output_dir,
        sft_val_ratio=args.sft_val_ratio,
        seed=args.seed,
    )
    log_stage(output_dir, "sft_export", "complete", sft_records=len(sft_records))

    write_metrics(
        tasks=tasks,
        baseline_rollouts=baseline_rollouts_for_generation,
        candidate_skills=candidate_skills,
        sft_records=sft_records,
        output_dir=output_dir,
    )
    log_stage(
        output_dir,
        "complete",
        "complete",
        sampled_tasks=len(tasks),
        baseline_rollouts=len(baseline_rollouts_for_generation),
        candidate_skills=len(candidate_skills),
        parse_ok_skills=sum(1 for record in candidate_skills if record.get("parse_ok")),
        sft_records=len(sft_records),
    )
    logging.info("Search QA episode-skill pipeline complete. Outputs are in %s.", output_dir)


if __name__ == "__main__":
    main()
