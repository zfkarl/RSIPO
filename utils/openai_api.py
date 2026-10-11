from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

from openai import OpenAI

from .prompt_builder import build_messages, normalize_messages

MessageLike = Mapping[str, Any]


def create_openai_client(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    timeout: Optional[float] = None,
    **client_kwargs: Any,
) -> OpenAI:
    """Create an OpenAI-compatible client from args or environment variables."""
    resolved_timeout = timeout if timeout is not None else client_kwargs.pop("timeout", None)
    resolved_max_retries = client_kwargs.pop("max_retries", 0)

    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    openai_kwargs: Dict[str, Any] = {
        "max_retries": resolved_max_retries,
    }
    if resolved_api_key is not None:
        openai_kwargs["api_key"] = resolved_api_key
    if resolved_base_url:
        openai_kwargs["base_url"] = resolved_base_url
    if organization:
        openai_kwargs["organization"] = organization
    if resolved_timeout is not None:
        openai_kwargs["timeout"] = resolved_timeout
    openai_kwargs.update(client_kwargs)
    return OpenAI(**openai_kwargs)


def chat_completion_with_retry(
    *,
    client: OpenAI,
    model: str,
    prompt: Optional[Any] = None,
    messages: Optional[Sequence[MessageLike]] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    assistant_prompt: Optional[str] = None,
    history: Optional[Sequence[MessageLike]] = None,
    retries: int = 5,
    retry_delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 30.0,
    jitter: float = 0.0,
    temperature: Optional[float] = None,
    max_completion_tokens: Optional[int] = None,
    require_nonempty_text: bool = True,
    return_response: bool = False,
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
    sleep_fn=time.sleep,
    **request_kwargs: Any,
) -> Any:
    """
    Call ``client.chat.completions.create`` with retry.

    ``retries`` follows the same convention as ``range(retries)``:
    it is the total number of attempts, not "extra retries".
    """
    if retries <= 0:
        raise ValueError("retries must be a positive integer.")

    normalized_messages = _resolve_messages(
        prompt=prompt,
        messages=messages,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        assistant_prompt=assistant_prompt,
        history=history,
    )

    request_payload: Dict[str, Any] = {
        "model": model,
        "messages": normalized_messages,
    }
    if temperature is not None:
        request_payload["temperature"] = temperature
    if max_completion_tokens is not None:
        request_payload["max_completion_tokens"] = max_completion_tokens
    request_payload.update(request_kwargs)

    last_error: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**request_payload)
            if return_response:
                return response

            text = extract_message_text(response)
            if require_nonempty_text and not text.strip():
                raise ValueError("Empty response from model.")
            return text
        except retry_on as exc:
            last_error = exc
            if attempt == retries - 1:
                break

            delay = min(max_delay, retry_delay * (backoff ** attempt))
            if jitter > 0:
                delay += random.uniform(0.0, jitter)
            sleep_fn(delay)

    raise RuntimeError(f"OpenAI chat completion failed after {retries} attempts.") from last_error


def extract_message_text(response: Any) -> str:
    """Extract text from a chat completion response."""
    choices = getattr(response, "choices", None)
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    reasoning = getattr(message, "reasoning_content", None)
    content = getattr(message, "content", None)

    parts = []
    if reasoning:
        parts.append(str(reasoning))
    content_text = _coerce_content_to_text(content)
    if content_text:
        parts.append(content_text)
    return "\n".join(parts)


def _resolve_messages(
    *,
    prompt: Optional[Any],
    messages: Optional[Sequence[MessageLike]],
    system_prompt: Optional[str],
    user_prompt: Optional[str],
    assistant_prompt: Optional[str],
    history: Optional[Sequence[MessageLike]],
) -> List[Dict[str, Any]]:
    if messages is not None:
        return normalize_messages(messages)
    if prompt is not None:
        return normalize_messages(prompt)

    resolved_messages = build_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        assistant_prompt=assistant_prompt,
        history=history,
    )
    if not resolved_messages:
        raise ValueError("No prompt or messages were provided.")
    return resolved_messages


def _coerce_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        text_parts = []
        for item in content:
            if isinstance(item, Mapping):
                if item.get("type") == "text" and "text" in item:
                    text_parts.append(str(item["text"]))
                elif "content" in item:
                    text_parts.append(str(item["content"]))
            else:
                text_parts.append(str(item))
        return "\n".join(part for part in text_parts if part)
    return str(content)
