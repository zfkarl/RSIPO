import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TASKS = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]

ALFWORLD_RESULT_TASKS = [
    ("Pick", "pick_and_place"),
    ("Look", "look_at_obj_in_light"),
    ("Clean", "pick_clean_then_place_in_recep"),
    ("Heat", "pick_heat_then_place_in_recep"),
    ("Cool", "pick_cool_then_place_in_recep"),
    ("Pick2", "pick_two_obj_and_place"),
]


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def load_env_file(env_file: Optional[str]) -> None:
    if not env_file:
        return

    path = Path(env_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local OpenAI-compatible/vLLM model on ALFWorld."
    )
    parser.add_argument("--env-file", default=".env", help="Optional env file to load first.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL or http://127.0.0.1:8000/v1.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the endpoint. Defaults to OPENAI_API_KEY or EMPTY for local vLLM.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Served model name. Defaults to OPENAI_MODEL or local-alfworld-model.",
    )
    parser.add_argument("--env-num", type=int, default=1, help="Number of parallel ALFWorld envs.")
    parser.add_argument("--test-times", type=int, default=1, help="Evaluation rounds.")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps per episode.")
    parser.add_argument("--history-length", type=int, default=5, help="Prompt history length.")
    parser.add_argument("--seed", type=int, default=1, help="Environment seed.")
    parser.add_argument(
        "--eval-dataset",
        choices=["eval_in_distribution", "eval_out_of_distribution"],
        default="eval_in_distribution",
        help="ALFWorld evaluation split.",
    )
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature.")
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=512,
        help="Maximum completion tokens per API call.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="OpenAI client timeout.")
    parser.add_argument("--retries", type=int, default=2, help="Attempts per model call.")
    parser.add_argument(
        "--request-workers",
        type=int,
        default=None,
        help="Concurrent API requests per environment step. Defaults to env-num.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Delay between model-call retries.",
    )
    parser.add_argument(
        "--fallback-action",
        default="look",
        help="Action used only when the API call itself fails.",
    )
    parser.add_argument(
        "--extra-body-json",
        default=None,
        help="Optional JSON object passed as extra_body to chat.completions.create.",
    )
    parser.add_argument("--log-dir", default="logs/alfworld_local_vllm", help="Log directory.")
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Directory for per-run config JSON files. Defaults to <log-dir>/configs.",
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        help="Directory for per-run result JSON files. Defaults to <log-dir>/results.",
    )
    parser.add_argument(
        "--trajectory-dir",
        default=None,
        help="Directory for per-episode trajectory JSONL files. Defaults to <log-dir>/trajectories.",
    )
    parser.add_argument(
        "--no-save-trajectories",
        action="store_true",
        help="Disable saving per-episode interaction trajectories.",
    )
    parser.add_argument(
        "--log-responses",
        action="store_true",
        help="Log every raw model response. This can make logs large.",
    )
    return parser.parse_args()


def resolve_client_config(args: argparse.Namespace) -> Tuple[str, str, str]:
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1"
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    model_name = args.model_name or os.environ.get("OPENAI_MODEL") or "local-alfworld-model"
    return base_url, api_key, model_name


def build_env(args: argparse.Namespace):
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    from agent_system.environments.env_package.alfworld import (
        alfworld_projection,
        build_alfworld_envs,
    )

    alf_config_path = (
        PROJECT_ROOT
        / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
    )
    env_kwargs = {"eval_dataset": args.eval_dataset}
    resources_per_worker = {"num_cpus": 0.05, "num_gpus": 0.0}
    envs = build_alfworld_envs(
        str(alf_config_path),
        seed=args.seed,
        env_num=args.env_num,
        group_n=1,
        is_train=False,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
    )
    config = AttrDict(
        {
            "env": AttrDict(
                {
                    "env_name": "alfworld/AlfredTWEnv",
                    "history_length": args.history_length,
                    "use_skills_only_memory": False,
                    "use_retrieval_memory": False,
                }
            )
        }
    )
    return AlfWorldEnvironmentManager(envs, alfworld_projection, config)


def coerce_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def extract_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content_text = coerce_content_to_text(getattr(message, "content", None)).strip()
    reasoning = getattr(message, "reasoning_content", None)
    reasoning_text = str(reasoning).strip() if reasoning else ""

    if reasoning_text and content_text and "<think>" not in content_text.lower():
        return f"<think>{reasoning_text}</think>\n{content_text}"
    if content_text:
        return content_text
    return reasoning_text


def has_action_format(text: str) -> bool:
    lowered = text.lower()
    return "<think>" in lowered and "</think>" in lowered and "<action>" in lowered and "</action>" in lowered


def resolve_request_workers(request_workers: Optional[int], env_num: int) -> int:
    if request_workers is None or request_workers <= 0:
        return max(1, env_num)
    return max(1, min(request_workers, env_num))


def collect_actions_concurrently(
    *,
    agent: "LocalOpenAIAgent",
    observations: List[str],
    env_dones: List[bool],
    done_action: str,
    request_workers: int,
    log_responses: bool,
    format_checker: Callable[[str], bool] = has_action_format,
) -> Tuple[List[str], List[str], List[float], List[float]]:
    actions = [done_action for _ in env_dones]
    model_responses = [done_action for _ in env_dones]
    format_flags: List[float] = []
    api_error_flags: List[float] = []
    active_indices = [idx for idx, done in enumerate(env_dones) if not done]
    if not active_indices:
        return actions, model_responses, format_flags, api_error_flags

    max_workers = max(1, min(request_workers, len(active_indices)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(agent.get_action, observations[idx]): idx
            for idx in active_indices
        }
        for future in as_completed(future_to_idx):
            env_idx = future_to_idx[future]
            try:
                action, _, error = future.result()
            except Exception as exc:
                action = (
                    "<think>The local inference request failed, so I will take a safe "
                    f"fallback action.</think><action>{agent.fallback_action}</action>"
                )
                error = f"{type(exc).__name__}: {exc}"

            actions[env_idx] = action
            model_responses[env_idx] = action
            format_flags.append(float(format_checker(action)))
            api_error_flags.append(float(error is not None))

            if log_responses:
                logging.info(
                    "Env %d raw response: %s",
                    env_idx,
                    action.replace("\n", "\\n"),
                )
            if error:
                logging.warning("Env %d API error fallback: %s", env_idx, error)

    return actions, model_responses, format_flags, api_error_flags


class LocalOpenAIAgent:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float,
        max_completion_tokens: int,
        timeout: float,
        retries: int,
        retry_delay: float,
        fallback_action: str,
        extra_body: Optional[Dict[str, Any]],
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.retries = retries
        self.retry_delay = retry_delay
        self.fallback_action = fallback_action
        self.extra_body = extra_body
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    def get_action(self, observation: str) -> Tuple[str, bool, Optional[str]]:
        last_error: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": observation}],
                    "temperature": self.temperature,
                    "max_completion_tokens": self.max_completion_tokens,
                    "n": 1,
                }
                if self.extra_body:
                    request_kwargs["extra_body"] = self.extra_body

                response = self.client.chat.completions.create(**request_kwargs)
                text = extract_response_text(response).strip()
                return text, has_action_format(text), None
            except Exception as exc:
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(self.retry_delay)

        fallback = (
            f"<think>The local inference request failed, so I will take a safe observation action.</think>"
            f"<action>{self.fallback_action}</action>"
        )
        return fallback, False, f"{type(last_error).__name__}: {last_error}"


def sanitize_run_component(value: Any, max_length: int = 120) -> str:
    text = str(value).strip()
    safe = "".join(char if char.isalnum() or char in {".", "_", "-"} else "-" for char in text)
    safe = safe.strip("._-")
    while "--" in safe:
        safe = safe.replace("--", "-")
    if not safe:
        safe = "unknown"
    return safe[:max_length]


def build_run_name(dataset: str, model_name: str, split: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [
        sanitize_run_component(dataset),
        sanitize_run_component(model_name),
        sanitize_run_component(split),
        timestamp,
    ]
    return "_".join(parts)


def setup_logging(log_dir: str, run_name: Optional[str] = None) -> str:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"run_log_{run_name or datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    for noisy_logger in ("httpx", "httpcore", "openai", "openai._base_client"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    return log_path


def resolve_run_config_path(log_dir: str, config_dir: Optional[str], run_name: Optional[str] = None) -> str:
    resolved_dir = config_dir or os.path.join(log_dir, "configs")
    os.makedirs(resolved_dir, exist_ok=True)
    return os.path.join(
        resolved_dir,
        f"run_config_{run_name or datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )


def resolve_result_path(log_dir: str, result_dir: Optional[str], run_name: Optional[str] = None) -> str:
    resolved_dir = result_dir or os.path.join(log_dir, "results")
    os.makedirs(resolved_dir, exist_ok=True)
    return os.path.join(
        resolved_dir,
        f"results_{run_name or datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )


def resolve_trajectory_path(log_dir: str, trajectory_dir: Optional[str], run_name: Optional[str] = None) -> str:
    resolved_dir = trajectory_dir or os.path.join(log_dir, "trajectories")
    os.makedirs(resolved_dir, exist_ok=True)
    return os.path.join(
        resolved_dir,
        f"trajectories_{run_name or datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def write_trajectories(path: str, trajectories: List[Dict[str, Any]]) -> None:
    if not trajectories:
        return
    with open(path, "a", encoding="utf-8") as file:
        for trajectory in trajectories:
            file.write(json.dumps(json_safe(trajectory), ensure_ascii=False) + "\n")


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(json_safe(payload), file, ensure_ascii=False, indent=2)


def metric_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "values": []}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "values": [float(value) for value in values],
    }


SENSITIVE_KEY_PARTS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
RUN_ENV_KEYS = [
    "ENV_FILE",
    "MODEL_PATH",
    "MODEL_NAME",
    "HOST",
    "PORT",
    "BASE_URL",
    "VLLM_BIN",
    "START_VLLM",
    "KEEP_VLLM_ALIVE",
    "VLLM_STARTUP_TIMEOUT",
    "VLLM_LOG_DIR",
    "VLLM_LOG_FILE",
    "TENSOR_PARALLEL_SIZE",
    "DATA_PARALLEL_SIZE",
    "GPU_MEMORY_UTILIZATION",
    "MAX_MODEL_LEN",
    "DTYPE",
    "VLLM_EXTRA_ARGS",
    "CUDA_VISIBLE_DEVICES",
    "ENV_NUM",
    "TEST_TIMES",
    "MAX_STEPS",
    "HISTORY_LENGTH",
    "EVAL_DATASET",
    "GENERALIZATION_LEVEL",
    "SIMPLIFICATIONS_PRESET",
    "SCIWORLD_ENV_STEP_LIMIT",
    "SCIWORLD_JAR_PATH",
    "WEBSHOP_HUMAN_GOALS",
    "WEBSHOP_USE_SMALL",
    "TEMPERATURE",
    "MAX_COMPLETION_TOKENS",
    "REQUEST_WORKERS",
    "EVAL_LOG_DIR",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "OPENAI_API_RETRIES",
    "OPENAI_API_RETRY_DELAY",
]


def redact_value(key: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key_upper = key.upper()
    if any(part in key_upper for part in SENSITIVE_KEY_PARTS):
        return "<redacted>"
    return value


def collect_run_environment() -> Dict[str, str]:
    return {
        key: redact_value(key, os.environ[key])
        for key in RUN_ENV_KEYS
        if key in os.environ
    }


def write_run_config(
    path: str,
    *,
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
    model_name: str,
    request_workers: int,
    log_path: str,
    result_path: Optional[str],
    trajectory_path: Optional[str],
    extra_body: Optional[Dict[str, Any]],
    script_path: Optional[str] = None,
    run_name: Optional[str] = None,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
) -> None:
    args_config = vars(args).copy()
    args_config["api_key"] = redact_value("api_key", args_config.get("api_key"))
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(script_path or __file__).resolve()),
        "project_root": str(PROJECT_ROOT),
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "run_name": run_name,
        "dataset": dataset,
        "split": split,
        "args": json_safe(args_config),
        "resolved": {
            "base_url": base_url,
            "api_key": redact_value("api_key", api_key),
            "model_name": model_name,
            "request_workers": request_workers,
            "log_path": log_path,
            "result_path": result_path,
            "trajectory_path": trajectory_path,
            "extra_body": json_safe(extra_body),
        },
        "environment": collect_run_environment(),
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(json_safe(config), file, ensure_ascii=False, indent=2)


def resolve_extra_body(raw_json: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw_json:
        return None

    value = json.loads(raw_json)
    if not isinstance(value, dict):
        raise ValueError("--extra-body-json must decode to a JSON object.")
    return value


def task_name_from_info(info: Dict[str, Any]) -> str:
    gamefile = str(info.get("extra.gamefile", ""))
    for task in TASKS:
        if task in gamefile:
            return task
    return "other"


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    base_url, api_key, model_name = resolve_client_config(args)
    extra_body = resolve_extra_body(args.extra_body_json)
    dataset_name = "alfworld"
    split_name = args.eval_dataset
    run_name = build_run_name(dataset_name, model_name, split_name)
    log_path = setup_logging(args.log_dir, run_name)
    request_workers = resolve_request_workers(args.request_workers, args.env_num)
    config_path = resolve_run_config_path(args.log_dir, args.config_dir, run_name)
    result_path = resolve_result_path(args.log_dir, args.result_dir, run_name)
    trajectory_path = None
    if not args.no_save_trajectories:
        trajectory_path = resolve_trajectory_path(args.log_dir, args.trajectory_dir, run_name)
    write_run_config(
        config_path,
        args=args,
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        request_workers=request_workers,
        log_path=log_path,
        result_path=result_path,
        trajectory_path=trajectory_path,
        extra_body=extra_body,
        run_name=run_name,
        dataset=dataset_name,
        split=split_name,
    )

    logging.info("Local ALFWorld evaluation log: %s", log_path)
    logging.info("ALFWorld run name: %s", run_name)
    logging.info("ALFWorld run config written to: %s", config_path)
    logging.info("ALFWorld results will be written to: %s", result_path)
    if trajectory_path:
        logging.info("ALFWorld trajectories will be written to: %s", trajectory_path)
    logging.info(
        "env_num=%d test_times=%d max_steps=%d history_length=%d eval_dataset=%s request_workers=%d",
        args.env_num,
        args.test_times,
        args.max_steps,
        args.history_length,
        args.eval_dataset,
        request_workers,
    )

    env_manager = None
    overall_success_rates: List[float] = []
    category_macro_success_rates: List[float] = []
    action_format_rates: List[float] = []
    api_error_rates: List[float] = []
    task_success_history = defaultdict(list)
    round_results: List[Dict[str, Any]] = []

    try:
        env_manager = build_env(args)
        agent = LocalOpenAIAgent(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            timeout=args.timeout,
            retries=args.retries,
            retry_delay=args.retry_delay,
            fallback_action=args.fallback_action,
            extra_body=extra_body,
        )

        for test_idx in range(args.test_times):
            logging.info("========== Start test %d ==========", test_idx)
            start_time = time.time()
            obs, infos = env_manager.reset({})
            env_dones = [False] * args.env_num
            success_flags = np.zeros(args.env_num, dtype=bool)
            task_names = [task_name_from_info(info) for info in infos]
            task_success_cnt = defaultdict(int)
            task_total_cnt = defaultdict(int)
            response_format_flags: List[float] = []
            api_error_flags: List[float] = []
            trajectories = [
                {
                    "test_idx": test_idx,
                    "env_idx": env_idx,
                    "task": env_manager.tasks[env_idx] if hasattr(env_manager, "tasks") else "",
                    "task_type": task_names[env_idx],
                    "initial_info": json_safe(infos[env_idx]),
                    "steps": [],
                }
                for env_idx in range(args.env_num)
            ]

            for step_idx in range(args.max_steps):
                logging.info(
                    "Step %d | Dones %d/%d | Current SR %.4f",
                    step_idx,
                    int(np.sum(env_dones)),
                    args.env_num,
                    float(success_flags.mean()),
                )

                current_observations = list(obs["text"])
                actions, model_responses, step_format_flags, step_error_flags = collect_actions_concurrently(
                    agent=agent,
                    observations=obs["text"],
                    env_dones=env_dones,
                    done_action="<think>The episode is done.</think><action>look</action>",
                    request_workers=request_workers,
                    log_responses=args.log_responses,
                    format_checker=has_action_format,
                )
                response_format_flags.extend(step_format_flags)
                api_error_flags.extend(step_error_flags)

                actions_for_env = list(actions)
                next_obs, rewards, dones, infos = env_manager.step(actions_for_env)

                for env_idx in range(args.env_num):
                    if env_dones[env_idx]:
                        continue
                    trajectories[env_idx]["steps"].append(
                        {
                            "step_idx": step_idx,
                            "observation": current_observations[env_idx],
                            "model_response": model_responses[env_idx],
                            "raw_action_text": actions[env_idx],
                            "executed_action": actions_for_env[env_idx],
                            "reward": json_safe(rewards[env_idx]),
                            "done": bool(dones[env_idx]),
                            "info": json_safe(infos[env_idx]),
                            "next_observation": next_obs["text"][env_idx],
                        }
                    )

                obs = next_obs

                for env_idx in range(args.env_num):
                    if env_dones[env_idx]:
                        continue
                    if dones[env_idx]:
                        env_dones[env_idx] = True
                        won = bool(infos[env_idx].get("won", False))
                        success_flags[env_idx] = won
                        if task_names[env_idx] == "other":
                            task_names[env_idx] = task_name_from_info(infos[env_idx])

                if all(env_dones):
                    logging.info("All environments finished early.")
                    break

            round_success_rate = float(success_flags.mean())
            round_format_rate = (
                float(np.mean(response_format_flags)) if response_format_flags else 0.0
            )
            round_api_error_rate = float(np.mean(api_error_flags)) if api_error_flags else 0.0
            overall_success_rates.append(round_success_rate)
            action_format_rates.append(round_format_rate)
            api_error_rates.append(round_api_error_rate)

            for env_idx, task_name in enumerate(task_names):
                task_total_cnt[task_name] += 1
                if success_flags[env_idx]:
                    task_success_cnt[task_name] += 1
                trajectories[env_idx]["task_type"] = task_name
                trajectories[env_idx]["success"] = bool(success_flags[env_idx])
                trajectories[env_idx]["completed"] = bool(env_dones[env_idx])
                trajectories[env_idx]["num_steps"] = len(trajectories[env_idx]["steps"])

            category_rates = [
                task_success_cnt[task] / task_total_cnt[task]
                for task in TASKS + ["other"]
                if task_total_cnt.get(task, 0) > 0
            ]
            round_category_macro_success = (
                float(np.mean(category_rates)) if category_rates else 0.0
            )
            category_macro_success_rates.append(round_category_macro_success)
            round_subsets = []
            for display_name, task_key in ALFWORLD_RESULT_TASKS:
                total = task_total_cnt.get(task_key, 0)
                success_count = task_success_cnt.get(task_key, 0)
                round_subsets.append(
                    {
                        "name": display_name,
                        "task_key": task_key,
                        "success_rate": (success_count / total) if total > 0 else None,
                        "success_count": success_count,
                        "total": total,
                    }
                )
            if task_total_cnt.get("other", 0) > 0:
                total = task_total_cnt["other"]
                success_count = task_success_cnt.get("other", 0)
                round_subsets.append(
                    {
                        "name": "Other",
                        "task_key": "other",
                        "success_rate": success_count / total,
                        "success_count": success_count,
                        "total": total,
                    }
                )
            round_results.append(
                {
                    "test_idx": test_idx,
                    "overall_success": round_success_rate,
                    "category_macro_success": round_category_macro_success,
                    "action_format_valid": round_format_rate,
                    "api_error_fallback": round_api_error_rate,
                    "subsets": round_subsets,
                    "time_elapsed_sec": time.time() - start_time,
                }
            )

            logging.info("Test %d overall success: %.4f", test_idx, round_success_rate)
            logging.info(
                "Test %d category-macro success: %.4f",
                test_idx,
                round_category_macro_success,
            )
            logging.info("Test %d action format valid rate: %.4f", test_idx, round_format_rate)
            logging.info("Test %d API error fallback rate: %.4f", test_idx, round_api_error_rate)
            for display_name, task in ALFWORLD_RESULT_TASKS:
                if task_total_cnt.get(task, 0) > 0:
                    rate = task_success_cnt[task] / task_total_cnt[task]
                    task_success_history[task].append(rate)
                    logging.info(
                        "    %-8s %-35s: %.4f (%d/%d)",
                        display_name,
                        task,
                        rate,
                        task_success_cnt[task],
                        task_total_cnt[task],
                    )
            if task_total_cnt.get("other", 0) > 0:
                rate = task_success_cnt["other"] / task_total_cnt["other"]
                task_success_history["other"].append(rate)
                logging.info(
                    "    %-8s %-35s: %.4f (%d/%d)",
                    "Other",
                    "other",
                    rate,
                    task_success_cnt["other"],
                    task_total_cnt["other"],
                )
            logging.info("Test %d time elapsed: %.2fs", test_idx, time.time() - start_time)
            if trajectory_path:
                write_trajectories(trajectory_path, trajectories)

        logging.info("=============== Final Summary ===============")
        logging.info(
            "Total tests: %d | Envs / test: %d | Total envs: %d",
            args.test_times,
            args.env_num,
            args.env_num * args.test_times,
        )
        logging.info(
            "Overall success avg +/- std: %.4f +/- %.4f",
            float(np.mean(overall_success_rates)) if overall_success_rates else 0.0,
            float(np.std(overall_success_rates)) if overall_success_rates else 0.0,
        )
        logging.info(
            "Category-macro success avg +/- std: %.4f +/- %.4f",
            float(np.mean(category_macro_success_rates)) if category_macro_success_rates else 0.0,
            float(np.std(category_macro_success_rates)) if category_macro_success_rates else 0.0,
        )
        logging.info(
            "Action format valid avg +/- std: %.4f +/- %.4f",
            float(np.mean(action_format_rates)) if action_format_rates else 0.0,
            float(np.std(action_format_rates)) if action_format_rates else 0.0,
        )
        logging.info(
            "API error fallback avg +/- std: %.4f +/- %.4f",
            float(np.mean(api_error_rates)) if api_error_rates else 0.0,
            float(np.std(api_error_rates)) if api_error_rates else 0.0,
        )
        subset_summary = []
        for display_name, task in ALFWORLD_RESULT_TASKS:
            values = task_success_history.get(task, [])
            summary = metric_stats(values)
            subset_summary.append(
                {
                    "name": display_name,
                    "task_key": task,
                    "success": summary,
                }
            )
            if values:
                logging.info(
                    "%-8s %-35s: %.4f +/- %.4f",
                    display_name,
                    task,
                    summary["mean"],
                    summary["std"],
                )
        if task_success_history.get("other"):
            summary = metric_stats(task_success_history["other"])
            subset_summary.append(
                {
                    "name": "Other",
                    "task_key": "other",
                    "success": summary,
                }
            )
            logging.info(
                "%-8s %-35s: %.4f +/- %.4f",
                "Other",
                "other",
                summary["mean"],
                summary["std"],
            )

        result_payload = {
            "run_name": run_name,
            "dataset": dataset_name,
            "model_name": model_name,
            "split": split_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "paths": {
                "log": log_path,
                "config": config_path,
                "results": result_path,
                "trajectories": trajectory_path,
            },
            "total_tests": args.test_times,
            "envs_per_test": args.env_num,
            "total_envs": args.env_num * args.test_times,
            "summary": {
                "overall_success": metric_stats(overall_success_rates),
                "category_macro_success": metric_stats(category_macro_success_rates),
                "action_format_valid": metric_stats(action_format_rates),
                "api_error_fallback": metric_stats(api_error_rates),
            },
            "subsets": subset_summary,
            "rounds": round_results,
        }
        write_json(result_path, result_payload)
        logging.info("ALFWorld results written to: %s", result_path)
    finally:
        if env_manager is not None:
            env_manager.close()


if __name__ == "__main__":
    main()
