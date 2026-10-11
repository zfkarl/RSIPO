import ast
import base64
import io
import json
import logging
import mimetypes
import os
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from gigpo import core_gigpo
from seed.prompting import validate_skill_mode
from utils import build_prompt_dict, chat_completion_with_retry, create_openai_client


logger = logging.getLogger(__name__)


_TASK_DESCRIPTION_PATTERNS = (
    re.compile(r"Your task is to:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Your task is:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Your current task is:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Your question:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Your goal is to\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"You are [^\n]*?\bhelping to\s*(.+?)(?:\n|$)", re.IGNORECASE),
)

_ANALYSIS_PROMPT_VERSIONS = {
    "seed",
    "seed_visual",
}


def validate_analysis_prompt_version(version: object) -> str:
    value = str(version or "seed").strip()
    if value not in _ANALYSIS_PROMPT_VERSIONS:
        raise ValueError(
            f"Unsupported SEED analysis_prompt_version: {value!r}. "
            f"Expected one of {sorted(_ANALYSIS_PROMPT_VERSIONS)}."
        )
    return value


def _clean_task_description(task_description: object) -> str:
    task = " ".join(str(task_description or "").split())
    return task.strip()


def _extract_task_description_from_text(text: object) -> str:
    text = str(text or "")
    for pattern in _TASK_DESCRIPTION_PATTERNS:
        match = pattern.search(text)
        if match:
            task_description = _clean_task_description(match.group(1))
            if task_description:
                return task_description
    return ""


def build_traj_step_indices(traj_index: Sequence) -> np.ndarray:
    """Return per-sample step indices in trajectory order."""
    counters: Dict[object, int] = defaultdict(int)
    step_indices = []
    for traj_uid in traj_index:
        step_indices.append(counters[traj_uid])
        counters[traj_uid] += 1
    return np.asarray(step_indices, dtype=np.int64)


def select_critical_steps_by_stats(
    step_rewards: torch.Tensor,
    anchor_obs: np.ndarray,
    index: np.ndarray,
    traj_index: np.ndarray,
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
    min_group_size: int = 2,
    var_quantile: float = 0.75,
    topk_per_traj: int = 1,
    below_group_mean_only: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Select critical steps using reward statistics of each observation group.

    A step becomes a candidate when its observation group has sufficiently high
    return variance. Candidate steps are ranked by how much they underperform
    the group mean (or deviate from it), and we keep the top-k per trajectory.
    """
    if topk_per_traj <= 0 or len(step_rewards) == 0:
        return np.zeros(len(step_rewards), dtype=bool), {
            "num_groups": 0.0,
            "eligible_groups": 0.0,
            "selected_steps": 0.0,
        }

    step_group_uids = core_gigpo.build_step_group(
        anchor_obs=anchor_obs,
        index=index,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
    )
    step_rewards_np = step_rewards.detach().cpu().numpy().astype(np.float32)
    step_indices = build_traj_step_indices(traj_index)

    group_to_indices: Dict[object, List[int]] = defaultdict(list)
    for sample_idx, group_uid in enumerate(step_group_uids):
        group_to_indices[group_uid].append(sample_idx)

    group_stats = {}
    eligible_group_vars: List[float] = []
    for group_uid, sample_indices in group_to_indices.items():
        rewards = step_rewards_np[sample_indices]
        group_mean = float(np.mean(rewards))
        group_var = float(np.var(rewards))
        group_stats[group_uid] = {
            "mean": group_mean,
            "var": group_var,
            "size": float(len(sample_indices)),
        }
        if len(sample_indices) >= min_group_size:
            eligible_group_vars.append(group_var)

    if eligible_group_vars:
        clipped_quantile = min(max(var_quantile, 0.0), 1.0)
        var_cutoff = float(np.quantile(np.asarray(eligible_group_vars, dtype=np.float32), clipped_quantile))
    else:
        var_cutoff = float("inf")

    sample_scores = np.zeros(len(step_rewards_np), dtype=np.float32)
    candidate_mask = np.zeros(len(step_rewards_np), dtype=bool)
    for sample_idx, group_uid in enumerate(step_group_uids):
        group_stat = group_stats[group_uid]
        group_mean = group_stat["mean"]
        group_var = group_stat["var"]
        group_size = int(group_stat["size"])
        if group_size < min_group_size or group_var < var_cutoff or group_var <= 0.0:
            continue

        reward = step_rewards_np[sample_idx]
        if below_group_mean_only:
            deviation = max(group_mean - reward, 0.0)
        else:
            deviation = abs(reward - group_mean)
        if deviation <= 0.0 and below_group_mean_only:
            continue

        candidate_mask[sample_idx] = True
        sample_scores[sample_idx] = group_var * (deviation + 1e-6)

    critical_mask = np.zeros(len(step_rewards_np), dtype=bool)
    traj_to_candidates: Dict[object, List[int]] = defaultdict(list)
    for sample_idx, is_candidate in enumerate(candidate_mask):
        if is_candidate:
            traj_to_candidates[traj_index[sample_idx]].append(sample_idx)

    for traj_uid, sample_indices in traj_to_candidates.items():
        ranked = sorted(
            sample_indices,
            key=lambda i: (sample_scores[i], step_indices[i]),
            reverse=True,
        )
        for sample_idx in ranked[:topk_per_traj]:
            critical_mask[sample_idx] = True

    metrics = {
        "num_groups": float(len(group_to_indices)),
        "eligible_groups": float(sum(
            1
            for group_stat in group_stats.values()
            if int(group_stat["size"]) >= min_group_size and group_stat["var"] >= var_cutoff and group_stat["var"] > 0.0
        )),
        "selected_steps": float(critical_mask.sum()),
        "variance_cutoff": 0.0 if not np.isfinite(var_cutoff) else float(var_cutoff),
    }
    return critical_mask, metrics


def build_episode_records(
    tokenizer,
    obs_texts: Sequence,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    traj_index: Sequence,
    step_indices: Sequence,
    step_rewards: Optional[torch.Tensor] = None,
    obs_raws: Optional[Sequence] = None,
    obs_images: Optional[Sequence] = None,
    action_valids: Optional[Sequence] = None,
) -> Dict[object, List[Dict[str, object]]]:
    """Decode trajectories into step records for analysis.

    `observation` prefers raw text feedback from the environment when
    available (for example `anchor_obs` in rollout batches). The prompt-form
    observation is preserved in `observation_prompt` for debugging.
    """
    if step_rewards is not None:
        reward_np = step_rewards.detach().cpu().numpy().astype(np.float32)
    else:
        reward_np = None

    episodes: Dict[object, List[Dict[str, object]]] = defaultdict(list)
    responses_cpu = responses.detach().cpu()
    response_mask_cpu = response_mask.detach().cpu()
    for sample_idx, traj_uid in enumerate(traj_index):
        valid_len = int(response_mask_cpu[sample_idx].sum().item())
        response_text = tokenizer.decode(
            responses_cpu[sample_idx][:valid_len],
            skip_special_tokens=True,
        )
        prompt_observation = str(obs_texts[sample_idx])
        raw_observation = prompt_observation
        if obs_raws is not None:
            candidate_raw_observation = obs_raws[sample_idx]
            if isinstance(candidate_raw_observation, (str, np.str_)):
                raw_observation = str(candidate_raw_observation)
        step_record = {
            "step_index": int(step_indices[sample_idx]),
            "observation": raw_observation,
            "observation_prompt": prompt_observation,
            "response": response_text,
        }
        if obs_images is not None:
            try:
                observation_image = obs_images[sample_idx]
            except Exception:
                observation_image = None
            if observation_image is not None:
                step_record["observation_image"] = observation_image
                step_record["has_observation_image"] = True
        task_description = (
            _extract_task_description_from_text(prompt_observation)
            or _extract_task_description_from_text(raw_observation)
        )
        if task_description:
            step_record["task_description"] = task_description
        if reward_np is not None:
            step_record["step_reward"] = float(reward_np[sample_idx])
        if action_valids is not None:
            step_record["action_valid"] = bool(action_valids[sample_idx])
        episodes[traj_uid].append(step_record)

    for traj_uid in episodes:
        episodes[traj_uid].sort(key=lambda step: step["step_index"])
    return episodes


class SEEDEpisodeAnalyzer:
    """
    Analyze trajectories for reusable episode-level and critical-step skills.

    The analyzer uses an OpenAI-compatible JSON analysis backend or the
    on-policy rollout model when the trainer selects ``policy_vllm``.
    ``azure`` remains a legacy alias for ``openai``.
    """

    def __init__(
        self,
        backend: str = "openai",
        max_completion_tokens: int = 4096,
        max_step_skills_per_traj: int = 1,
        skill_mode: str = "episode_step",
        analysis_prompt_version: str = "seed",
        include_episode_summary: bool = True,
    ):
        self.requested_backend = backend
        if backend == "azure":
            logger.warning("SEED backend='azure' is deprecated; using the OpenAI-compatible backend instead.")
            backend = "openai"
        elif backend not in {"openai", "policy_vllm"}:
            raise ValueError(f"Unsupported SEED backend: {backend}")

        self.backend = backend
        self.max_completion_tokens = max_completion_tokens
        self.max_step_skills_per_traj = max(int(max_step_skills_per_traj), 0)
        self.skill_mode = validate_skill_mode(skill_mode)
        self.analysis_prompt_version = validate_analysis_prompt_version(analysis_prompt_version)
        self.include_episode_summary = bool(include_episode_summary)
        self.client = None
        self.model = os.environ.get("OPENAI_MODEL", "gemini-2.5-flash")

        logger.info(
            "Initialized SEEDEpisodeAnalyzer with requested_backend=%s, resolved_backend=%s, model=%s, max_completion_tokens=%s, max_step_skills_per_traj=%s, skill_mode=%s, analysis_prompt_version=%s, include_episode_summary=%s",
            self.requested_backend,
            self.backend,
            self.model,
            self.max_completion_tokens,
            self.max_step_skills_per_traj,
            self.skill_mode,
            self.analysis_prompt_version,
            self.include_episode_summary,
        )

    def _get_openai_client(self):
        if self.backend != "openai":
            raise ValueError(f"SEED backend {self.backend!r} does not use an OpenAI client.")
        if self.client is None:
            self.client = create_openai_client(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )
        return self.client

    def analyze_episode(
        self,
        steps: List[Dict[str, object]],
        candidate_step_indices: Optional[Sequence[int]] = None,
        analysis_mode: str = "teacher_bootstrap",
        episode_success: Optional[float] = None,
        task_description: Optional[str] = None,
    ) -> Dict[str, object]:
        if self.backend != "openai":
            raise ValueError(f"analyze_episode is only implemented for openai, got {self.backend!r}.")
        return self._analyze_episode_with_openai(
            steps=steps,
            candidate_step_indices=candidate_step_indices,
            task_description=task_description,
            analysis_mode=analysis_mode,
            episode_success=episode_success,
        )

    def _analysis_has_required_skill(self, parsed: Dict[str, object]) -> bool:
        if self.skill_mode == "step_only":
            step_skills = parsed.get("step_skills")
            if not isinstance(step_skills, dict):
                return False
            return any(self._is_nontrivial_skill_text(skill) for skill in step_skills.values())
        return self._is_nontrivial_skill_text(parsed.get("episode_skill", ""))

    @staticmethod
    def _is_nontrivial_skill_text(value: object) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        if text in {"...", "…"} or re.fullmatch(r"[\s.。…,_-]+", text):
            return False
        return len(text.split()) >= 5

    @staticmethod
    def _extract_step_image(step: Dict[str, object]) -> Optional[Any]:
        if step.get("observation_image") is not None:
            return step.get("observation_image")
        images = step.get("images")
        if isinstance(images, list) and images:
            first = images[0]
            if isinstance(first, dict):
                return first.get("image")
            return first
        return None

    @staticmethod
    def _image_to_data_url(image: Any) -> str:
        if isinstance(image, dict):
            image = image.get("image") or image.get("url") or image.get("path")
        if isinstance(image, (str, os.PathLike)):
            image_text = str(image)
            if image_text.startswith(("data:", "http://", "https://")):
                return image_text
            path = Path(image_text).expanduser()
            mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
            payload = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime_type};base64,{payload}"
        if isinstance(image, bytes):
            payload = base64.b64encode(image).decode("ascii")
            return f"data:image/png;base64,{payload}"
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        if isinstance(image, np.ndarray):
            from PIL import Image

            array = image
            if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
                array = np.moveaxis(array, 0, -1)
            if np.issubdtype(array.dtype, np.floating):
                scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
                array = np.clip(array * scale, 0, 255).astype(np.uint8)
            elif array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            image = Image.fromarray(array)
        if hasattr(image, "save"):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            payload = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/png;base64,{payload}"
        raise TypeError(f"Unsupported image value for OpenAI multimodal prompt: {type(image)!r}")

    def _build_openai_prompt_with_images(
        self,
        prompt: Dict[str, Any],
        steps: List[Dict[str, object]],
    ) -> Dict[str, Any]:
        if self.analysis_prompt_version != "seed_visual":
            return prompt
        messages = prompt.get("messages", []) if isinstance(prompt, dict) else []
        placeholder_count = sum(str(message.get("content", "")).count("<image>") for message in messages)
        if placeholder_count == 0:
            return prompt

        images = [
            self._extract_step_image(step)
            for step in steps
            if self._extract_step_image(step) is not None
        ]
        if len(images) != placeholder_count:
            raise ValueError(
                f"OpenAI multimodal SEED prompt has {placeholder_count} image placeholder(s) "
                f"but {len(images)} image(s)."
            )

        image_urls = [self._image_to_data_url(image) for image in images]
        image_index = 0
        prompt_with_images = deepcopy(prompt)
        prompt_with_images["messages"] = []
        for message in messages:
            copied = dict(deepcopy(message))
            content = copied.get("content")
            if isinstance(content, str) and "<image>" in content:
                parts = []
                segments = content.split("<image>")
                for segment_index, segment in enumerate(segments):
                    if segment:
                        parts.append({"type": "text", "text": segment})
                    if segment_index + 1 < len(segments):
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": image_urls[image_index]},
                            }
                        )
                        image_index += 1
                copied["content"] = parts
            prompt_with_images["messages"].append(copied)
        return prompt_with_images

    def _analyze_episode_with_openai(
        self,
        steps: List[Dict[str, object]],
        candidate_step_indices: Optional[Sequence[int]] = None,
        analysis_mode: str = "teacher_bootstrap",
        episode_success: Optional[float] = None,
        task_description: Optional[str] = None,
    ) -> Dict[str, object]:
        candidate_list = (
            [int(step["step_index"]) for step in steps]
            if candidate_step_indices is None
            else [int(idx) for idx in candidate_step_indices]
        )
        prompt = self._build_episode_analysis_prompt(
            steps=steps,
            candidate_step_indices=candidate_list,
            task_description=task_description,
            analysis_mode=analysis_mode,
            episode_success=episode_success,
        )
        parse_attempts = self._get_parse_attempts()
        content = ""
        last_error: Optional[BaseException] = None
        for parse_attempt in range(parse_attempts):
            prompt_for_attempt = (
                prompt
                if parse_attempt == 0
                else self._build_json_retry_prompt(
                    original_prompt=prompt,
                    invalid_response=content,
                    error=last_error,
                )
            )
            content = chat_completion_with_retry(
                client=self._get_openai_client(),
                model=self.model,
                prompt=self._build_openai_prompt_with_images(prompt_for_attempt, steps),
                retries=max(1, int(os.environ.get("OPENAI_API_RETRIES", "5"))),
                retry_delay=float(os.environ.get("OPENAI_API_RETRY_DELAY", "1.0")),
                max_completion_tokens=self.max_completion_tokens,
            )
            try:
                parsed = self._parse_analysis_response(content)
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "SEED analyzer JSON parse failed on attempt %s/%s: %s",
                    parse_attempt + 1,
                    parse_attempts,
                    exc,
                )
                continue
            if not self._analysis_has_required_skill(parsed):
                missing_field = "step_skills" if self.skill_mode == "step_only" else "episode_skill"
                last_error = ValueError(f"SEED analyzer response missing required field: {missing_field}")
                logger.warning(
                    "SEED analyzer JSON parse failed on attempt %s/%s: %s",
                    parse_attempt + 1,
                    parse_attempts,
                    last_error,
                )
                continue

            candidate_step_set = set(candidate_list)
            filtered_step_skills = {
                step_idx: skill
                for step_idx, skill in parsed.get("step_skills", {}).items()
                if step_idx in candidate_step_set
            }
            parsed["step_skills"] = dict(
                list(filtered_step_skills.items())[: self.max_step_skills_per_traj]
            )
            parsed["analysis_backend_requested"] = self.requested_backend
            parsed["analysis_backend_used"] = "openai"
            parsed["analysis_error"] = None
            parsed["analysis_mode"] = analysis_mode
            parsed["analysis_prompt_version"] = self.analysis_prompt_version
            parsed["task_description"] = task_description or self._infer_task_description(steps)
            parsed["llm_prompt"] = prompt_for_attempt
            parsed["llm_raw_output"] = content
            return parsed

        logger.warning(
            "SEED analyzer failed to produce valid JSON after %s attempt(s); skipping teacher skill for this trajectory: %s",
            parse_attempts,
            last_error,
        )
        return {
            "episode_summary": "",
            "episode_skill": "",
            "step_skills": {},
            "analysis_backend_requested": self.requested_backend,
            "analysis_backend_used": "openai",
            "analysis_error": str(last_error) if last_error is not None else "unknown JSON parse error",
            "analysis_mode": analysis_mode,
            "analysis_prompt_version": self.analysis_prompt_version,
            "task_description": task_description or self._infer_task_description(steps),
            "llm_prompt": prompt,
            "llm_raw_output": content,
        }

    def _get_parse_attempts(self) -> int:
        raw_attempts = os.environ.get("SEED_ANALYSIS_PARSE_RETRIES", "2")
        try:
            return max(1, int(raw_attempts))
        except (TypeError, ValueError):
            logger.warning("Invalid SEED_ANALYSIS_PARSE_RETRIES=%r; using 2.", raw_attempts)
            return 2

    def _build_json_retry_prompt(
        self,
        original_prompt: Dict[str, Any],
        invalid_response: str,
        error: Optional[BaseException],
    ) -> Dict[str, Any]:
        messages = original_prompt.get("messages", []) if isinstance(original_prompt, dict) else []
        original_user_prompt = ""
        for message in messages:
            if message.get("role") == "user":
                original_user_prompt = str(message.get("content", ""))
                break
        invalid_preview = str(invalid_response or "")[-2000:]
        if self.skill_mode == "step_only":
            schema_text = """{
  "episode_summary": "string",
  "step_skills": {}
}"""
        elif self.skill_mode == "episode_only":
            schema_text = """{
  "episode_summary": "string",
  "episode_skill": "string"
}"""
        else:
            schema_text = """{
  "episode_summary": "string",
  "episode_skill": "string",
  "step_skills": {}
}"""
        if not self.include_episode_summary:
            if self.skill_mode == "step_only":
                schema_text = """{
  "step_skills": {}
}"""
            elif self.skill_mode == "episode_only":
                schema_text = """{
  "episode_skill": "string"
}"""
            else:
                schema_text = """{
  "episode_skill": "string",
  "step_skills": {}
}"""
        retry_prompt = f"""{original_user_prompt}

The previous answer could not be parsed as the required JSON object.
Parse error: {error}

Previous answer:
{invalid_preview}

Return ONLY one valid JSON object with this exact shape:
{schema_text}
"""
        return build_prompt_dict(user_prompt=retry_prompt)

    def _infer_task_description(self, steps: List[Dict[str, object]]) -> str:
        for step in steps:
            task_description = _clean_task_description(step.get("task_description", ""))
            if task_description:
                return task_description

        for step in steps:
            for field_name in ("observation_prompt", "observation"):
                task_description = _extract_task_description_from_text(step.get(field_name, ""))
                if task_description:
                    return task_description
        return ""

    def _build_default_episode_analysis_prompt(
        self,
        steps: List[Dict[str, object]],
        candidate_step_indices: Sequence[int],
        analysis_mode: str = "teacher_bootstrap",
        episode_success: Optional[float] = None,
        task_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        task_description = _clean_task_description(task_description) or self._infer_task_description(steps)
        max_skill_count = min(self.max_step_skills_per_traj, len(candidate_step_indices))

        outcome_label = "unknown"
        if episode_success is not None:
            try:
                outcome_label = "success" if float(episode_success) >= 1.0 else "failure"
            except (TypeError, ValueError):
                outcome_label = "unknown"

        if outcome_label == "success":
            episode_skill_instruction = (
                "Write one episode_skill that extracts the successful trajectory into workflow: "
                "the core decision rule and action ordering that made this trajectory work. "
                # "Phrase it as a general skill pattern, not as instructions for this exact task."
            )
        elif outcome_label == "failure":
            episode_skill_instruction = (
                "Write one episode_skill that extracts the failed trajectory into avoidance rules: "
                "the core mistake and warning signs that agent should avoid. "
                # "Phrase it as a general avoidance skill pattern, not as a one-off lesson tied to this exact task."
            )
        else:
            episode_skill_instruction = (
                "Write one episode_skill distilled from the trajectory. If it succeeded, extract "
                "the successful workflow; if it failed, extract the avoidance rules. "
            )

        summary_instruction = "Write a concise episode_summary."
        selection_instruction = (
            f"Provide concise, action-oriented decision skills for at most {max_skill_count} critical step(s) "
            "from the candidate set as entries in step_skills; use the full episode to infer the skill, "
            "but phrase each skill as advice the policy can act on at that step."
        )
        if self.skill_mode == "step_only":
            if not self.include_episode_summary:
                prompt_text = f"""Analyze the following agent episode and return ONLY valid JSON.

You need to complete one field:
1. {selection_instruction}

Important constraints:
- Step indexing is 0-based: step 0 is the first step of the trajectory.
- Use the full episode context to identify what each critical step should have done better.
- Each step_skills value should be one short imperative sentence for the policy at that step.
- Write step_skills as policy-facing skills, not as retrospective explanation of the trajectory.
- Return only this top-level field: step_skills.
- The chosen steps are exactly the keys present in step_skills.

Return format:
{{
  "step_skills": {{
    "0": "skill for step 0",
    "2": "skill for step 2"
  }}
}}

Episode context:
- Task description: {task_description or "(not available)"}
- episode_success: {outcome_label}
- Candidate step indices: {list(candidate_step_indices)}
- Interaction trajectory: {self._format_episode_steps(steps)}
"""
                return build_prompt_dict(user_prompt=prompt_text)

            prompt_text = f"""Analyze the following agent episode and return ONLY valid JSON.

You need to complete two fields:
1. {summary_instruction}
2. {selection_instruction}

Important constraints:
- Step indexing is 0-based: step 0 is the first step of the trajectory.
- Use the full episode context to identify what each critical step should have done better.
- Each step_skills value should be one short imperative sentence for the policy at that step.
- Write step_skills as policy-facing skills, not as retrospective explanation of the trajectory.
- Return only these top-level fields: episode_summary, step_skills.
- The chosen steps are exactly the keys present in step_skills.

Return format:
{{
  "episode_summary": "string",
  "step_skills": {{
    "0": "skill for step 0",
    "2": "skill for step 2"
  }}
}}

Episode context:
- Task description: {task_description or "(not available)"}
- episode_success: {outcome_label}
- Candidate step indices: {list(candidate_step_indices)}
- Interaction trajectory: {self._format_episode_steps(steps)}
"""
            return build_prompt_dict(user_prompt=prompt_text)

        if self.skill_mode == "episode_only":
            if not self.include_episode_summary:
                prompt_text = f"""Analyze the following agent episode and return ONLY valid JSON.

You need to complete one field:
1. {episode_skill_instruction}

Important constraints:
- Write episode_skill as one short policy-facing skill, not as retrospective explanation of the trajectory.
- Return only this top-level field: episode_skill.

Return format:
{{
  "episode_skill": "string"
}}

Episode context:
- Task description: {task_description or "(not available)"}
- episode_success: {outcome_label}
- Interaction trajectory: {self._format_episode_steps(steps)}
"""
                return build_prompt_dict(user_prompt=prompt_text)

            prompt_text = f"""Analyze the following agent episode and return ONLY valid JSON.

You need to complete two fields:
1. {summary_instruction}
2. {episode_skill_instruction}

Important constraints:
- Write episode_skill as one short policy-facing skill, not as retrospective explanation of the trajectory.
- Return only these top-level fields: episode_summary, episode_skill.

Return format:
{{
  "episode_summary": "string",
  "episode_skill": "string"
}}

Episode context:
- Task description: {task_description or "(not available)"}
- episode_success: {outcome_label}
- Interaction trajectory: {self._format_episode_steps(steps)}
"""
            return build_prompt_dict(user_prompt=prompt_text)

        if not self.include_episode_summary:
            prompt_text = f"""Analyze the following agent episode and return ONLY valid JSON.

You need to complete two fields:
1. {episode_skill_instruction}
2. {selection_instruction}

Important constraints:
- Step indexing is 0-based: step 0 is the first step of the trajectory.
- Use the full episode context to identify what each critical step should have done better.
- Each step_skills value should be one short imperative sentence for the policy at that step.
- Write episode_skill and step_skills as policy-facing skills, not as retrospective explanation of the trajectory.
- Return only these top-level fields: episode_skill, step_skills.
- The chosen steps are exactly the keys present in step_skills.

Return format:
{{
  "episode_skill": "string",
  "step_skills": {{
    "0": "skill for step 0",
    "2": "skill for step 2"
  }}
}}

Episode context:
- Task description: {task_description or "(not available)"}
- episode_success: {outcome_label}
- Candidate step indices: {list(candidate_step_indices)}
- Interaction trajectory: {self._format_episode_steps(steps)}
"""
            return build_prompt_dict(user_prompt=prompt_text)

        prompt_text = f"""Analyze the following agent episode and return ONLY valid JSON.

You need to complete all three fields:
1. {summary_instruction}
2. {episode_skill_instruction}
3. {selection_instruction}

Important constraints:
- Step indexing is 0-based: step 0 is the first step of the trajectory.
- Use the full episode context to identify what each critical step should have done better.
- Each step_skills value should be one short imperative sentence for the policy at that step.
- Write step_skills as policy-facing skills, not as retrospective explanation of the trajectory.
- Return only these top-level fields: episode_summary, episode_skill, step_skills.
- The chosen steps are exactly the keys present in step_skills.

Return format:
{{
  "episode_summary": "string",
  "episode_skill": "string",
  "step_skills": {{
    "0": "skill for step 0",
    "2": "skill for step 2"
  }}
}}

Episode context:
- Task description: {task_description or "(not available)"}
- episode_success: {outcome_label}
- Candidate step indices: {list(candidate_step_indices)}
- Interaction trajectory: {self._format_episode_steps(steps)}
"""
        return build_prompt_dict(user_prompt=prompt_text)

    def _build_seed_visual_episode_analysis_prompt(
        self,
        steps: List[Dict[str, object]],
        candidate_step_indices: Sequence[int],
        analysis_mode: str = "teacher_bootstrap",
        episode_success: Optional[float] = None,
        task_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        for step in steps:
            if not step.get("has_observation_image"):
                raise ValueError("seed_visual analysis requires an observation image for every episode step.")
        image_steps = []
        for step in steps:
            image_step = dict(step)
            image_step["observation"] = "<image>"
            image_step["response"] = str(step["response"]).replace("<image>", "[image]")
            image_steps.append(image_step)
        return self._build_default_episode_analysis_prompt(
            steps=image_steps,
            candidate_step_indices=candidate_step_indices,
            analysis_mode=analysis_mode,
            episode_success=episode_success,
            task_description=task_description,
        )

    def _build_episode_analysis_prompt(
        self,
        steps: List[Dict[str, object]],
        candidate_step_indices: Sequence[int],
        analysis_mode: str = "teacher_bootstrap",
        episode_success: Optional[float] = None,
        task_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.analysis_prompt_version == "seed_visual":
            return self._build_seed_visual_episode_analysis_prompt(
                steps=steps,
                candidate_step_indices=candidate_step_indices,
                analysis_mode=analysis_mode,
                episode_success=episode_success,
                task_description=task_description,
            )
        return self._build_default_episode_analysis_prompt(
            steps=steps,
            candidate_step_indices=candidate_step_indices,
            analysis_mode=analysis_mode,
            episode_success=episode_success,
            task_description=task_description,
        )

    def _parse_analysis_response(self, response: str) -> Dict[str, object]:
        parsed = self._extract_json_object(response)
        step_skills_raw = parsed.get("step_skills", {}) or {}
        step_skills = {}
        if isinstance(step_skills_raw, dict):
            for step_idx, skill in step_skills_raw.items():
                try:
                    step_skills[int(step_idx)] = str(skill)
                except (TypeError, ValueError):
                    continue
        if self.skill_mode == "episode_only":
            step_skills = {}
        return {
            "episode_summary": str(parsed.get("episode_summary", "")) if self.include_episode_summary else "",
            "episode_skill": "" if self.skill_mode == "step_only" else str(parsed.get("episode_skill", "")),
            "step_skills": step_skills,
        }

    def _extract_json_object(self, response: str) -> Dict[str, object]:
        response_text = str(response or "")
        candidate_texts = self._candidate_json_texts(response_text)
        decoded_objects: List[Dict[str, object]] = []
        saw_object_start = False
        last_error: Optional[BaseException] = None

        for text in candidate_texts:
            if "{" in text:
                saw_object_start = True
            for variant in self._json_text_variants(text):
                objects, error = self._decode_json_objects(variant)
                decoded_objects.extend(objects)
                if error is not None:
                    last_error = error

        preferred_field = "step_skills" if self.skill_mode == "step_only" else "episode_skill"
        for obj in decoded_objects:
            if preferred_field in obj:
                return obj
        if decoded_objects:
            return decoded_objects[0]
        if not saw_object_start:
            raise ValueError("No JSON object found in SEED analyzer response.")
        raise ValueError(f"Could not parse JSON object in SEED analyzer response: {last_error}") from last_error

    def _candidate_json_texts(self, response_text: str) -> List[str]:
        fenced_blocks = re.findall(
            r"```(?:json|JSON)?\s*(.*?)```",
            response_text,
            flags=re.DOTALL,
        )
        candidates = [block.strip() for block in fenced_blocks if block.strip()]
        candidates.append(response_text.strip())
        return candidates

    def _json_text_variants(self, text: str) -> List[str]:
        repaired_trailing_commas = re.sub(r",\s*([}\]])", r"\1", text)
        variants = [text] if repaired_trailing_commas == text else [text, repaired_trailing_commas]
        # RSIPO: a small policy sometimes stops right after the last field, before the object's closing brace.
        missing = text.count("{") - text.count("}")
        if 0 < missing <= 2 and text.rstrip().endswith('"'):
            variants.append(text.rstrip() + "\n" + "}" * missing)
        return variants

    def _decode_json_objects(self, text: str) -> Tuple[List[Dict[str, object]], Optional[BaseException]]:
        objects: List[Dict[str, object]] = []
        last_error: Optional[BaseException] = None
        decoder = json.JSONDecoder(strict=False)

        for idx, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text, idx)
            except (json.JSONDecodeError, RecursionError) as exc:  # RSIPO: degenerate output nests thousands deep
                last_error = exc
                continue
            if isinstance(parsed, dict):
                objects.append(parsed)

        if objects:
            return objects, last_error

        for json_slice in self._balanced_object_slices(text):
            try:
                parsed = ast.literal_eval(json_slice)
            except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
                last_error = exc
                continue
            if isinstance(parsed, dict):
                objects.append(parsed)
        return objects, last_error

    def _balanced_object_slices(self, text: str) -> List[str]:
        slices = []
        start: Optional[int] = None
        depth = 0
        in_string = False
        quote_char = ""
        escape = False

        for idx, char in enumerate(text):
            if escape:
                escape = False
                continue
            if in_string:
                if char == "\\":
                    escape = True
                elif char == quote_char:
                    in_string = False
                continue
            if char in {"'", '"'}:
                in_string = True
                quote_char = char
                continue
            if char == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif char == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    slices.append(text[start:idx + 1])
                    start = None
        return slices

    def _format_episode_steps(self, steps: List[Dict[str, object]]) -> str:
        step_lines = []
        for step in steps:
            action_valid_str = ""
            if "action_valid" in step:
                action_valid_str = f"\nAction valid: {str(bool(step['action_valid'])).lower()}"
            step_lines.append(
                f"Step {step['step_index']}\n"
                f"Observation: {step['observation']}\n"
                f"Response: {step['response']}{action_valid_str}\n"
            )
        return "".join(step_lines)
