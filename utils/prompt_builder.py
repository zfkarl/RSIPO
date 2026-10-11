from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

PromptInput = Union[str, Mapping[str, Any], Sequence[Mapping[str, Any]]]
Message = Dict[str, Any]


def build_message(role: str, content: Any, **extra_fields: Any) -> Message:
    """Build one OpenAI-format message dict."""
    if not role:
        raise ValueError("role must be a non-empty string.")
    return {"role": role, "content": content, **extra_fields}


def build_messages(
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    assistant_prompt: Optional[str] = None,
    history: Optional[Iterable[Mapping[str, Any]]] = None,
) -> List[Message]:
    """Build a messages list in OpenAI chat format."""
    messages: List[Message] = []
    if system_prompt is not None:
        messages.append(build_message("system", system_prompt))
    if history is not None:
        for message in history:
            messages.append(_copy_message(message))
    if user_prompt is not None:
        messages.append(build_message("user", user_prompt))
    if assistant_prompt is not None:
        messages.append(build_message("assistant", assistant_prompt))
    return messages


def build_prompt_dict(
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    assistant_prompt: Optional[str] = None,
    history: Optional[Iterable[Mapping[str, Any]]] = None,
    **extra_fields: Any,
) -> Dict[str, Any]:
    """Build a dict prompt with a top-level ``messages`` field."""
    return {
        "messages": build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            assistant_prompt=assistant_prompt,
            history=history,
        ),
        **extra_fields,
    }


def build_chat_payload(
    model: str,
    prompt: Optional[PromptInput] = None,
    **request_kwargs: Any,
) -> Dict[str, Any]:
    """Build a request payload for ``client.chat.completions.create``."""
    payload = {"model": model}
    if prompt is not None:
        payload["messages"] = normalize_messages(prompt)
    payload.update(request_kwargs)
    return payload


def normalize_messages(prompt: PromptInput) -> List[Message]:
    """
    Normalize a string / message dict / prompt dict / message list into ``messages``.

    Supported inputs:
    - ``"hello"`` -> ``[{"role": "user", "content": "hello"}]``
    - ``{"role": "user", "content": "hello"}``
    - ``{"messages": [...]}``
    - ``[{"role": "...", "content": ...}, ...]``
    """
    if isinstance(prompt, str):
        return [build_message("user", prompt)]

    if isinstance(prompt, Mapping):
        if "messages" in prompt:
            messages = prompt["messages"]
            if not isinstance(messages, Sequence):
                raise ValueError("prompt['messages'] must be a sequence of message dicts.")
            return [_copy_message(message) for message in messages]
        if "role" in prompt and "content" in prompt:
            return [_copy_message(prompt)]
        raise ValueError("Prompt dict must contain either 'messages' or both 'role' and 'content'.")

    if isinstance(prompt, Sequence):
        return [_copy_message(message) for message in prompt]

    raise TypeError(f"Unsupported prompt type: {type(prompt)!r}")


def _copy_message(message: Mapping[str, Any]) -> Message:
    if not isinstance(message, Mapping):
        raise TypeError(f"Each message must be a mapping, got {type(message)!r}")
    if "role" not in message or "content" not in message:
        raise ValueError("Each message must contain 'role' and 'content'.")
    return dict(deepcopy(message))
