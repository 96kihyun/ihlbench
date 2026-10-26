#!/usr/bin/env python3
"""OpenAI-compatible chat server for Qwen3.5, with tool calls and an optional LoRA adapter.

    CUDA_VISIBLE_DEVICES=0 python serving/serve_qwen.py --port 8003 --served-model-name qwen-base
    CUDA_VISIBLE_DEVICES=1 python serving/serve_qwen.py --port 8010 --served-model-name qwen-dpo \\
        --lora /path/to/adapter/final

Base and adapter models go through this one server so that a base-vs-DPO
contrast cannot come from a difference in templating or parsing.

  * thinking is off (`enable_thinking=False`)
  * `tool_choice=required` is enforced by seeding the reply with the
    `<tool_call>` marker, so the model cannot answer a forced-tool turn in prose
  * completions are decoded with special tokens kept, then chat markers are
    stripped, and parsed by qwen_tools.py
  * `seed` in the request seeds the sampler, so a trial is reproducible
"""

# FastAPI resolves these annotations at runtime, so this module deliberately
# does not use `from __future__ import annotations`: postponed evaluation makes
# pydantic fail to resolve the request model declared inside build_app.

import argparse
import json
import os
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForImageTextToText, AutoProcessor

from qwen_tools import TOOL_CALL_OPEN, parse_tool_calls

DEFAULT_MODEL = os.environ.get("IHL_QWEN_MODEL", "/path/to/Qwen3.5-9B")


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
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def _normalize_tool_calls(tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """The template expects arguments as a mapping, not a JSON string."""
    if not tool_calls:
        return None
    out = []
    for tc in tool_calls:
        tc = dict(tc)
        fn = dict(tc.get("function") if isinstance(tc.get("function"), dict) else tc)
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"raw": args}
        if not isinstance(args, dict):
            args = {"value": args}
        fn["arguments"] = args
        fn.setdefault("name", tc.get("name", "unknown"))
        tc["function"] = fn
        out.append(tc)
    return out


def _messages_for_template(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    role_map = {"developer": "system", "function": "tool", "model": "assistant"}
    out = []
    for m in messages:
        role = role_map.get(m.role, m.role)
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        row: dict[str, Any] = {"role": role, "content": _content_to_text(m.content)}
        tool_calls = _normalize_tool_calls(m.tool_calls)
        if tool_calls:
            row["tool_calls"] = tool_calls
        if m.tool_call_id:
            row["tool_call_id"] = m.tool_call_id
        if m.name:
            row["name"] = m.name
        out.append(row)
    return out


def _load(model_path: str, lora_path: str | None, dtype: str, device: str):
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    print(f"loading {model_path} lora={lora_path} dtype={dtype} device={device}", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch_dtype, trust_remote_code=True)
    if lora_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_path)
    model.to(device)
    model.eval()
    print("model ready", flush=True)
    return processor, model


def build_app(
    model_path: str,
    served_name: str,
    dtype: str = "bfloat16",
    device: str = "cuda:0",
    max_new_tokens_cap: int = 2048,
    lora_path: str | None = None,
) -> FastAPI:
    app = FastAPI()
    processor, model = _load(model_path, lora_path, dtype, device)
    dev = next(model.parameters()).device

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
                    "server": "serve_qwen",
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
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        if req.tools:
            kwargs["tools"] = req.tools
        try:
            prompt = processor.apply_chat_template(messages, **kwargs)
        except Exception as e:
            print(f"chat_template failed: {e!r}", flush=True)
            raise HTTPException(status_code=400, detail=f"chat_template: {e}")

        forced = bool(req.tools) and req.tool_choice not in (None, "none", "auto")
        if forced:
            prompt = prompt + TOOL_CALL_OPEN

        inputs = processor(text=prompt, return_tensors="pt")
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        input_len = int(inputs["input_ids"].shape[-1])
        gen: dict[str, Any] = {"max_new_tokens": max_new, "do_sample": False}
        if req.temperature is not None and req.temperature > 0:
            gen.update(do_sample=True, temperature=req.temperature)
            if req.top_p is not None:
                gen["top_p"] = req.top_p
        pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        if pad_id is not None:
            gen["pad_token_id"] = pad_id
        if req.seed is not None:
            torch.manual_seed(req.seed)
        out = model.generate(**inputs, **gen)
        new_tokens = out[0, input_len:]
        text = processor.tokenizer.decode(new_tokens, skip_special_tokens=False)
        content, tool_calls = parse_tool_calls((TOOL_CALL_OPEN if forced else "") + text)
        finish = "tool_calls" if tool_calls else "stop"
        if forced and not tool_calls:
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
    p.add_argument("--served-model-name", default="qwen-base")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens-cap", type=int, default=2048)
    args = p.parse_args()
    app = build_app(
        args.model,
        args.served_model_name,
        args.dtype,
        args.device,
        max_new_tokens_cap=args.max_new_tokens_cap,
        lora_path=args.lora,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
