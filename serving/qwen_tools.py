"""Parse Qwen3.5 tool calls into the OpenAI shape.

Qwen3.5 writes a call as XML inside a tool_call block:

    <tool_call>
    <function=NAME>
    <parameter=rationale>
    ...
    </parameter>
    </function>
    </tool_call>

Some samples write a JSON object inside the block instead
(`{"name": ..., "arguments": {...}}`), and a truncated sample can lose its
closing tags, so both are accepted and open tags are closed before parsing.

Kept separate from the server so it can be tested without loading the
weights: `python serving/qwen_tools.py` runs the cases.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

TOOL_CALL_OPEN = "<tool_call>\n"
STRIP_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _parse_param_value(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        return ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _close_tool_xml(text: str) -> str:
    s = text
    s += "\n</parameter>" * max(0, len(re.findall(r"<parameter=", s)) - s.count("</parameter>"))
    s += "\n</function>" * max(0, len(re.findall(r"<function=", s)) - s.count("</function>"))
    s += "\n</tool_call>" * max(0, s.count("<tool_call>") - s.count("</tool_call>"))
    return s


def _call(name: str, arguments: dict[str, Any] | str) -> dict[str, Any]:
    args = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    return {"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function", "function": {"name": name, "arguments": args}}


def _from_function(fn: re.Match) -> dict[str, Any]:
    params = {p.group(1).strip(): _parse_param_value(p.group(2)) for p in _PARAM_RE.finditer(fn.group(2))}
    return _call(fn.group(1).strip(), params)


def clean(text: str) -> str:
    for t in STRIP_TOKENS:
        text = text.replace(t, "")
    return text


def parse_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    cleaned = _close_tool_xml(_THINK_RE.sub("", clean(text or "")).strip())
    calls: list[dict[str, Any]] = []
    for block in _TOOL_CALL_RE.finditer(cleaned):
        inner = block.group(1).strip()
        if inner.startswith("{"):
            try:
                obj = json.loads(inner)
            except json.JSONDecodeError:
                obj = None
            name = (obj or {}).get("name") or (obj or {}).get("function")
            if name:
                calls.append(_call(str(name), obj.get("arguments", obj.get("parameters", {}))))
                continue
        fn = _FUNCTION_RE.search(inner)
        if fn:
            calls.append(_from_function(fn))
    if not calls:
        calls = [_from_function(fn) for fn in _FUNCTION_RE.finditer(cleaned)]
    content = None if calls else (_TOOL_CALL_RE.sub("", cleaned).strip() or None)
    return content, calls


def _selftest() -> None:
    xml = "<tool_call>\n<function=a>\n<parameter=rationale>\nbecause\n</parameter>\n</function>\n</tool_call><|im_end|>"
    content, calls = parse_tool_calls(xml)
    assert content is None and calls[0]["function"]["name"] == "a"
    assert json.loads(calls[0]["function"]["arguments"]) == {"rationale": "because"}

    truncated = "<tool_call>\n<function=b>\n<parameter=rationale>\ncut off"
    _, calls = parse_tool_calls(truncated)
    assert calls[0]["function"]["name"] == "b"
    assert json.loads(calls[0]["function"]["arguments"]) == {"rationale": "cut off"}

    js = '<tool_call>\n{"name": "c", "arguments": {"rationale": "r"}}\n</tool_call>'
    _, calls = parse_tool_calls(js)
    assert calls[0]["function"]["name"] == "c"

    record = "<tool_call>\n<function=record_decision>\n<parameter=action>\nx_y\n</parameter>\n</function>\n</tool_call>"
    _, calls = parse_tool_calls(record)
    assert json.loads(calls[0]["function"]["arguments"]) == {"action": "x_y"}

    content, calls = parse_tool_calls("<think>hmm</think>I would hold fire.")
    assert calls == [] and content == "I would hold fire."
    print("qwen_tools: all cases pass")


if __name__ == "__main__":
    _selftest()
