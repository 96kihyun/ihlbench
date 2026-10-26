#!/usr/bin/env python3
"""OpenAI-compatible chat server for Ministral-3-14B-Instruct-2512, with tool calls.

    python serving/serve_ministral.py --port 8005 --device cuda:2 [--lora DIR]

Same shape as serve_gemma.py, but three things differ and each one breaks the
benchmark silently if handled the way Gemma's server handles it:

  * Prompts are built by mistral_common, not by a jinja template. `AutoTokenizer`
    resolves to `MistralCommonBackend` for this checkpoint, so the reference
    formatter is used as-is. It validates messages, which means every tool call
    id must be exactly 9 alphanumeric characters — the runner echoes our ids
    back in the next turn, so ids are rewritten on the way in as well as out.
  * The prompt is tokenized by that backend (`tokenize=True`) rather than
    rendered to text and re-encoded. Re-encoding a rendered prompt would add a
    second `<s>` and lose the control tokens, whose literal text does not
    encode back to them.
  * `tool_choice=required` is enforced by appending the `[TOOL_CALLS]` *control
    token id* to the input ids. Appending its text would append six ordinary
    tokens instead, which the model has never seen in that position.

The FP8 checkpoint cannot run natively on Ampere (needs compute capability
>= 8.9); transformers detects this and dequantizes the weights to bf16 at load,
which is why this needs ~30 GB rather than ~16 GB.
"""

# FastAPI resolves these annotations at runtime, so this module deliberately
# does not use `from __future__ import annotations`: postponed evaluation makes
# pydantic fail to resolve the request model declared inside build_app.

import argparse
import json
import os
import re
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForImageTextToText, AutoTokenizer

from mistral_tools import TOOL_CALL_OPEN, parse_tool_calls

DEFAULT_MODEL = os.environ.get("IHL_MINISTRAL_MODEL", "/path/to/Ministral-3-14B-Instruct-2512")

_VALID_CALL_ID = re.compile(r"^[A-Za-z0-9]{9}$")
# mistral_common.protocol.instruct.validator: names are [a-zA-Z0-9_-]{1,64}.
# The model sometimes emits a dotted near-miss of a real tool (N-15:
# fire_on.descending_aircrew). Scoring keeps the raw name; only the replayed
# history has to pass the validator, or turn 2 of that session 400s.
_VALID_FN_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str | list[Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    max_tokens: int | None = 256
    max_completion_tokens: int | None = None
    n: int = 1
    stop: list[str] | str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    seed: int | None = None


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def _coerce_call_id(raw_id: Any, seen: dict[str, str]) -> str:
    """Map any incoming id onto the 9-alphanumeric form mistral_common demands.

    The mapping is remembered per request so the id on an assistant tool call
    and the id on the tool result that answers it stay equal; the validator
    rejects the conversation otherwise.
    """
    key = str(raw_id) if raw_id is not None else ""
    if key in seen:
        return seen[key]
    if _VALID_CALL_ID.match(key):
        seen[key] = key
        return key
    new_id = uuid.uuid4().hex[:9]
    seen[key] = new_id
    return new_id


def _coerce_fn_name(raw: Any) -> str:
    name = str(raw) if raw else "unknown"
    if _VALID_FN_NAME.match(name):
        return name
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64].strip("_")
    if not cleaned:
        cleaned = "unknown"
    if cleaned[0].isdigit():
        cleaned = ("f_" + cleaned)[:64]
    print(f"coerced function name {name!r} -> {cleaned!r}", flush=True)
    return cleaned


def _normalize_tool_calls(
    tool_calls: list[dict[str, Any]] | None, seen: dict[str, str]
) -> list[dict[str, Any]] | None:
    if not tool_calls:
        return None
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        tc = dict(tc)
        fn_src = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        fn = dict(fn_src)
        args = fn.get("arguments", {})
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        elif not isinstance(args, str):
            args = json.dumps({"value": args}, ensure_ascii=False)
        fn["arguments"] = args
        fn.setdefault("name", tc.get("name", "unknown"))
        out.append(
            {
                "id": _coerce_call_id(tc.get("id"), seen),
                "type": "function",
                "function": {
                    "name": _coerce_fn_name(fn["name"]),
                    "arguments": fn["arguments"],
                },
            }
        )
    return out


def _messages_for_template(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    role_map = {"developer": "system", "function": "tool", "model": "assistant"}
    seen: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for m in messages:
        role = role_map.get(m.role, m.role)
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        row: dict[str, Any] = {"role": role, "content": _content_to_text(m.content)}
        if role == "assistant":
            tool_calls = _normalize_tool_calls(m.tool_calls, seen)
            if tool_calls:
                row["tool_calls"] = tool_calls
                # Validator: content xor tool_calls. Keep the call; drop prose.
                row["content"] = ""
            elif not row["content"]:
                # An assistant turn that produced neither text nor a call still
                # has to be replayable, or a refusal in turn 1 would abort every
                # later turn of the session with a validation error.
                row["content"] = "(no response)"
        if role == "tool":
            row["tool_call_id"] = _coerce_call_id(m.tool_call_id, seen)
            if m.name:
                row["name"] = _coerce_fn_name(m.name)
        out.append(row)
    return out


def _tool_call_token_id(tokenizer) -> int:
    inner = tokenizer.tokenizer.instruct_tokenizer.tokenizer
    for attr in ("get_special_token", "get_control_token"):
        getter = getattr(inner, attr, None)
        if getter is None:
            continue
        try:
            return int(getter(TOOL_CALL_OPEN))
        except Exception:
            continue
    raise RuntimeError(f"cannot resolve the {TOOL_CALL_OPEN} control token id")


def _load(
    model_path: str,
    dtype: str,
    device: str,
    load_device: str | None = None,
    lora_path: str | None = None,
):
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    materialize = load_device or device
    print(
        f"loading {model_path} lora={lora_path} dtype={dtype} device={device} load={materialize}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map=materialize,
    )
    if str(materialize) != str(device):
        print(f"moving weights {materialize} -> {device}", flush=True)
        model.to(device)
    if lora_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    print("model ready", flush=True)
    return tokenizer, model


def build_app(
    model_path: str,
    served_name: str,
    dtype: str = "bfloat16",
    device: str = "cuda:0",
    max_new_tokens_cap: int = 2048,
    load_device: str | None = None,
    lora_path: str | None = None,
) -> FastAPI:
    app = FastAPI()
    tokenizer, model = _load(
        model_path, dtype, device, load_device=load_device, lora_path=lora_path
    )
    tool_call_id_token = _tool_call_token_id(tokenizer)
    dev = next(model.parameters()).device
    eos_ids = [i for i in {tokenizer.eos_token_id, 2} if i is not None]

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": served_name,
                    "object": "model",
                    "owned_by": "local",
                    "model_path": model_path,
                    "lora": lora_path,
                    "server": "serve_ministral",
                }
            ],
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    @torch.inference_mode()
    def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        requested = req.max_completion_tokens or req.max_tokens or 256
        max_new = max(1, min(int(requested), max_new_tokens_cap))
        messages = _messages_for_template(req.messages)
        print(
            f"roles={[m['role'] for m in messages]} n_tools={len(req.tools or [])} "
            f"tool_choice={req.tool_choice}",
            flush=True,
        )
        kwargs: dict[str, Any] = {"tokenize": True}
        if req.tools:
            kwargs["tools"] = req.tools
        try:
            encoded = tokenizer.apply_chat_template(messages, **kwargs)
        except Exception as e:
            print(f"chat_template failed: {e!r}", flush=True)
            raise HTTPException(status_code=400, detail=f"chat_template: {e}")

        input_ids = list(encoded["input_ids"])
        forced = bool(req.tools) and req.tool_choice not in (None, "none", "auto")
        if forced:
            input_ids.append(tool_call_id_token)
        input_len = len(input_ids)
        ids = torch.tensor([input_ids], dtype=torch.long, device=dev)
        attn = torch.ones_like(ids)

        gen: dict[str, Any] = {"max_new_tokens": max_new, "do_sample": False}
        if req.temperature is not None and req.temperature > 0:
            gen.update(do_sample=True, temperature=req.temperature)
            if req.top_p is not None:
                gen["top_p"] = req.top_p
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_ids[0] if eos_ids else None
        if pad_id is not None:
            gen["pad_token_id"] = pad_id
        if eos_ids:
            gen["eos_token_id"] = eos_ids

        if req.seed is not None:
            torch.manual_seed(req.seed)
        out = model.generate(input_ids=ids, attention_mask=attn, **gen)
        new_tokens = out[0, input_len:]
        # Must keep special tokens: the tool markers are control tokens, and
        # dropping them would erase the call structure before it is parsed.
        text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        content, tool_calls = parse_tool_calls(
            (TOOL_CALL_OPEN if forced else "") + text
        )
        finish = "tool_calls" if tool_calls else "stop"
        print(f"parsed_tool_calls={len(tool_calls)} finish={finish}", flush=True)
        if forced and not tool_calls:
            # A forced turn that yielded nothing is unscorable downstream, so
            # keep the raw completion to tell refusal from a parse failure.
            print(f"  FORCED-BUT-NO-CALL raw={text[:400]!r}", flush=True)

        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": served_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": int(new_tokens.shape[-1]),
                "total_tokens": input_len + int(new_tokens.shape[-1]),
            },
        }

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--lora", default=None)
    p.add_argument("--served-model-name", default="mistral/ministral-3-14b-it")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8005)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--load-device",
        default=None,
        help="materialize weights here first (use cpu when the target GPU is tight)",
    )
    p.add_argument("--max-new-tokens-cap", type=int, default=2048)
    args = p.parse_args()
    app = build_app(
        args.model,
        args.served_model_name,
        args.dtype,
        args.device,
        max_new_tokens_cap=args.max_new_tokens_cap,
        load_device=args.load_device,
        lora_path=args.lora,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
