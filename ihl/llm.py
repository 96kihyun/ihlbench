"""OpenAI-compatible chat completions client, standard library only."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


def _request(url: str, *, body: bytes | None, api_key: str | None, timeout: int) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST" if body else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float,
    max_tokens: int,
    tool_choice: str = "required",
    seed: int | None = None,
    api_key: str | None = None,
    timeout: int = 300,
    retries: int = 3,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Omitted rather than sent empty: a server that sees a tool list forces a
    # tool call, which would break the free-text turns.
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    if seed is not None:
        payload["seed"] = seed
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = f"{base_url.rstrip('/')}/chat/completions"
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            return _request(url, body=body, api_key=api_key, timeout=timeout)
        except urllib.error.HTTPError as e:
            last_err = RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:800]}")
            if e.code < 500:
                break
            time.sleep(2**attempt)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(2**attempt)
    raise RuntimeError(f"chat_completion failed after {retries} tries: {last_err}")


def server_info(base_url: str, api_key: str | None = None) -> dict[str, Any]:
    """What the server says it is serving (model path, adapter), for the manifest."""
    try:
        return _request(f"{base_url.rstrip('/')}/models", body=None, api_key=api_key, timeout=10)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def assistant_message(raw: dict[str, Any]) -> dict[str, Any]:
    return raw["choices"][0]["message"]


def tool_result_messages(message: dict[str, Any], content: str) -> list[dict[str, Any]]:
    """One neutral tool result per call, so the next turn is a valid request."""
    return [
        {"role": "tool", "tool_call_id": call["id"], "content": content}
        for call in message.get("tool_calls") or []
        if call.get("id") is not None
    ]
