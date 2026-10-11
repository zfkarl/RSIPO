import re
from typing import Any, Sequence, Tuple


SKILL_MODES = ("episode_step", "step_only", "episode_only")
SKILL_TEACHER_MODES = ("step_priority", "additive")


def validate_skill_mode(mode: str) -> str:
    mode = str(mode or "episode_step")
    if mode not in SKILL_MODES:
        raise ValueError(f"Unsupported SEED skill_mode: {mode}")
    return mode


def select_skill_teacher_sources(
    *,
    step_skill: str,
    episode_skill_enabled: bool,
    step_skill_enabled: bool,
    mode: str = "step_priority",
    skill_mode: str = "episode_step",
) -> Tuple[bool, bool]:
    """Return whether to score the episode-skill and step-skill prompts."""
    skill_mode = validate_skill_mode(skill_mode)
    if mode not in SKILL_TEACHER_MODES:
        raise ValueError(f"Unsupported SEED skill_teacher_mode: {mode}")

    use_step_skill = bool(str(step_skill).strip()) and step_skill_enabled
    if skill_mode == "step_only":
        return False, use_step_skill
    if skill_mode == "episode_only":
        return episode_skill_enabled, False

    use_episode_skill = episode_skill_enabled and (mode == "additive" or not use_step_skill)
    return use_episode_skill, use_step_skill


def build_augmented_observation_text(
    *,
    observation: str,
    episode_skill: str = "",
    step_skill: str = "",
) -> str:
    """Insert episode-level and optional critical-step teacher context into a prompt."""

    def _format_skill_section(title: str, instruction: str, body: Any) -> str:
        body_text = str(body).strip()
        if not body_text:
            return ""
        return f"**{title}**\n{instruction}:\n[{body_text}]"

    def _insert_before_anchor(prompt_text: str, sections: Sequence[str], anchors: Sequence[str]) -> str:
        merged_sections = [str(section).strip() for section in sections if str(section or "").strip()]
        if not merged_sections:
            return prompt_text

        insertion_block = "\n\n".join(merged_sections)
        for anchor in anchors:
            anchor_idx = prompt_text.find(anchor)
            if anchor_idx == -1:
                continue

            prefix = prompt_text[:anchor_idx].rstrip()
            suffix = prompt_text[anchor_idx:].lstrip()
            if prefix and suffix:
                return f"{prefix}\n\n{insertion_block}\n\n{suffix}"
            if prefix:
                return f"{prefix}\n\n{insertion_block}"
            return f"{insertion_block}\n\n{suffix}"

        return f"{prompt_text}\n\n{insertion_block}" if prompt_text else insertion_block

    def _insert_after_task_description(prompt_text: str, sections: Sequence[str]) -> str:
        merged_sections = [str(section).strip() for section in sections if str(section or "").strip()]
        if not merged_sections:
            return prompt_text

        insertion_block = "\n\n".join(merged_sections)
        task_line_patterns = (
            r"^.*\bYour task is to:\s*.*(?:\n|$)",
            r"^.*\bYour current task is:\s*.*(?:\n|$)",
            r"^.*\bYour task is:\s*.*(?:\n|$)",
            r"^.*\bYour question:\s*.*(?:\n|$)",
            r"^.*\bYour goal is(?: to)?\b[:\s].*(?:\n|$)",
        )
        for pattern in task_line_patterns:
            match = re.search(pattern, prompt_text, flags=re.MULTILINE)
            if match is None:
                continue

            prefix = prompt_text[: match.end()].rstrip()
            suffix = prompt_text[match.end() :].lstrip()
            if prefix and suffix:
                return f"{prefix}\n\n{insertion_block}\n\n{suffix}"
            if prefix:
                return f"{prefix}\n\n{insertion_block}"
            return f"{insertion_block}\n\n{suffix}"

        return _insert_before_anchor(
            prompt_text,
            merged_sections,
            (
                "Now it's your turn to",
                "Now it's your turn",
            ),
        )

    episode_section = _format_skill_section(
        "Episode-Level Skill",
        (
            "Refer to this episode-level skill when deciding what action "
            "to take in the current episode"
        ),
        episode_skill,
    )
    step_section = _format_skill_section(
        "Critical-Step Skill",
        "Use this current-step skill for this decision only",
        step_skill,
    )
    prompt_text = _insert_after_task_description(str(observation).strip(), [episode_section])
    return _insert_before_anchor(
        prompt_text,
        [step_section],
        (
            "Now it's your turn to",
            "Now it's your turn",
        ),
    )
