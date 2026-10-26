"""Family-specific DPO completions. Prompt messages stay the same."""

from __future__ import annotations

import json


def prompt_messages(system: str, user: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def qwen_tool_call(name: str, rationale: str) -> str:
    return (
        f"<tool_call>\n<function={name}>\n"
        f"<parameter=rationale>\n{rationale}\n</parameter>\n"
        f"</function>\n</tool_call>"
    )


def gemma_tool_call(name: str, rationale: str) -> str:
    # Native Gemma-4 call. Markers are special tokens; encode without
    # add_special_tokens so the tokenizer can still map the literals.
    return f'<|tool_call>call:{name}{{rationale:<|"|>{rationale}<|"|>}}<tool_call|>'


def assistant_tool_message(name: str, rationale: str, call_id: str = "call00001") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": {"rationale": rationale},
                },
            }
        ],
    }


def mistral_control_id(tokenizer, name: str) -> int:
    tid = tokenizer.convert_tokens_to_ids(name)
    if isinstance(tid, int) and tid != getattr(tokenizer, "unk_token_id", None):
        return tid
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None:
        inner = getattr(inner, "instruct_tokenizer", inner)
        inner = getattr(inner, "tokenizer", inner)
        for attr in ("get_special_token", "get_control_token"):
            getter = getattr(inner, attr, None)
            if getter is None:
                continue
            try:
                return int(getter(name))
            except Exception:
                continue
    raise RuntimeError(f"cannot resolve control token {name!r}")


def encode_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def encode_completion(tokenizer, family: str, name: str, rationale: str) -> list[int]:
    if family == "gemma":
        ids = encode_ids(tokenizer, gemma_tool_call(name, rationale))
    elif family == "ministral":
        tc = mistral_control_id(tokenizer, "[TOOL_CALLS]")
        args = mistral_control_id(tokenizer, "[ARGS]")
        body = json.dumps({"rationale": rationale}, ensure_ascii=False)
        ids = [tc] + encode_ids(tokenizer, name) + [args] + encode_ids(tokenizer, body)
    else:
        ids = encode_ids(tokenizer, qwen_tool_call(name, rationale))
    eos = tokenizer.eos_token_id
    if eos is not None and (not ids or ids[-1] != eos):
        ids = ids + [eos]
    return ids


def apply_prompt_template(processor, messages, tools, family: str) -> str | list[int]:
    kwargs = {"tokenize": False, "add_generation_prompt": True, "tools": tools}
    if family == "ministral":
        encoded = processor.apply_chat_template(
            messages, tokenize=True, tools=tools
        )
        return list(encoded["input_ids"])
    try:
        return processor.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)
