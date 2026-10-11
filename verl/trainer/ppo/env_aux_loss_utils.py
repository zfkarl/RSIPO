from typing import Any, Dict, List, Sequence, Tuple


def format_history_pairs(
    history_pairs: Sequence[Tuple[str, str]],
    history_start_step: int,
    total_previous_steps: int,
) -> str:
    if total_previous_steps == 0:
        return "You have not taken any previous steps before this action.\n"

    lines = [
        f'[Observation {history_start_step + idx}: "{obs}", Action {history_start_step + idx}: "{act}"]'
        for idx, (obs, act) in enumerate(history_pairs)
    ]
    return (
        f"Prior to this step, you have already taken {total_previous_steps} step(s). "
        "Below are the history observations and the corresponding actions you took:\n"
        + "\n".join(lines)
        + "\n"
    )


def format_available_actions(actions: Sequence[Any]) -> str:
    if len(actions) == 0:
        return (
            "The environment did not provide any admissible actions for this step; "
            "reason about plausible actions based on the observations.\n"
        )

    choices = "\n".join(f"- {act}" for act in actions)
    return (
        "You must select exactly one of the following admissible actions to explain the transition:\n"
        f"{choices}\n"
    )


def create_state_prediction_messages(
    history_pairs: List[Tuple[str, str]],
    current_obs: str,
    action: str,
    next_obs: str,
    step_number: int,
    history_start_step: int,
    task: str = None,
    system_prompt: str = "You are an expert agent operating in an interactive environment.",
) -> List[Dict[str, str]]:
    total_previous_steps = step_number - 1
    history_text = format_history_pairs(history_pairs, history_start_step, total_previous_steps)
    task_text = f"Your task is: {task}\n" if task else ""

    user_content = (
        f"{system_prompt}\n"
        f"{task_text}"
        f"{history_text}"
        f"You are now at step {step_number} and your current observation is: {current_obs}\n"
        f"You take the action: {action}.\n\n"
        "Please predict the observation after taking this action.\n"
        "Present your prediction within <observation> </observation> tags."
    )
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": f"<observation>{next_obs}</observation>"},
    ]


def create_search_inverse_dynamics_messages(
    history_pairs: List[Tuple[str, str]],
    current_obs: str,
    next_obs: str,
    action: str,
    step_number: int,
    history_start_step: int,
    task: str = None,
    system_prompt: str = "You are an expert agent operating in an interactive environment.",
) -> List[Dict[str, str]]:
    total_previous_steps = step_number - 1
    history_text = format_history_pairs(history_pairs, history_start_step, total_previous_steps)
    task_text = f"Your task is: {task}\n" if task else ""

    user_content = (
        f"{system_prompt}\n"
        f"{task_text}"
        f"{history_text}"
        f"You are now at step {step_number}.\n"
        f"Your current observation is: {current_obs}\n"
        f"In the next timestep, the environment observation becomes: {next_obs}\n\n"
        "What search query did you issue to cause this transition?\n"
        "Present your search query within <search> </search> tags."
    )
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": f"<search>{action}</search>"},
    ]


def create_inverse_dynamics_messages(
    history_pairs: List[Tuple[str, str]],
    current_obs: str,
    next_obs: str,
    action: str,
    admissible_actions: List[str],
    step_number: int,
    history_start_step: int,
    task: str = None,
    system_prompt: str = "You are an expert agent operating in an interactive environment.",
) -> List[Dict[str, str]]:
    total_previous_steps = step_number - 1
    history_text = format_history_pairs(history_pairs, history_start_step, total_previous_steps)

    if action not in admissible_actions:
        admissible_actions = list(admissible_actions) + [action]

    import random

    random.shuffle(admissible_actions)
    available_actions_text = format_available_actions(admissible_actions)
    task_text = f"Your task is: {task}\n" if task else ""

    user_content = (
        f"{system_prompt}\n"
        f"{task_text}"
        f"{history_text}"
        f"You are now at step {step_number}.\n"
        f"Your current observation is: {current_obs}\n"
        f"In the next timestep, the environment observation becomes: {next_obs}\n\n"
        f"{available_actions_text}"
        "Choose the single admissible action that best explains this transition.\n"
        "Present your chosen action within <action> </action> tags."
    )
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": f"<action>{action}</action>"},
    ]
