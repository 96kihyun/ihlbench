"""Stand-in OpenAI-compatible server, for exercising the scripts without a model.

    python tests/mock_server.py --port 8099 --mode random &
    python scripts/run_decision.py --base-url http://127.0.0.1:8099/v1 --model mock --run _mock/baseline

Turns without tools get a text reply (one of the two words when the question
asks for exactly one, chosen like a tool). Turns with tools get one call whose
arguments are filled from the schema: an enum takes the first or second value
(or a seeded random one), booleans are false, strings are a stub. The choice
of tool follows the same rule, so `random` with the request seed is
deterministic per trial.

It rejects a message list that is not a valid tool-calling conversation, the
failure that would otherwise only show up mid-run against a real server.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

MODES = ("first", "second", "random", "text_only")


def validate(messages: list[dict]) -> None:
    """A tool call must be followed by a tool result before the next user turn."""
    if not messages or messages[0].get("role") != "system":
        raise ValueError("first message must be the system prompt")
    pending: set[str] = set()
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                pending.add(call["id"])
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ValueError(f"tool result for unknown call id {call_id!r}")
            pending.discard(call_id)
        elif role == "user" and pending:
            raise ValueError(f"user turn while tool calls are unanswered: {sorted(pending)}")


def pick(values: list[Any], mode: str, rng: random.Random) -> Any:
    if mode == "first":
        return values[0]
    if mode == "second":
        return values[-1]
    return rng.choice(values)


def fill(schema: dict, mode: str, rng: random.Random) -> dict:
    out = {}
    for key, prop in (schema.get("properties") or {}).items():
        if "enum" in prop:
            out[key] = pick(prop["enum"], mode, rng)
        elif prop.get("type") == "boolean":
            out[key] = False
        else:
            out[key] = f"mock {key}"
    return out


class Handler(BaseHTTPRequestHandler):
    mode = "random"
    counter = 0

    def log_message(self, *args) -> None:
        pass

    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._send(200, {"object": "list", "data": [{"id": "mock", "object": "model", "mode": Handler.mode}]})

    def do_POST(self) -> None:
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            validate(payload.get("messages") or [])
        except (ValueError, KeyError) as e:
            print(f"REJECTED: {e}", flush=True)
            self._send(400, {"error": {"message": str(e)}})
            return
        rng = random.Random(payload.get("seed"))
        tools = payload.get("tools") or []
        Handler.counter += 1
        last_user = next((m.get("content") or "" for m in reversed(payload.get("messages") or []) if m.get("role") == "user"), "")
        one_word = re.search(r"Answer with exactly one word: (\w+) or (\w+)\.\s*$", last_user)
        if not tools and one_word:
            message = {"role": "assistant", "content": pick(list(one_word.groups()), Handler.mode, rng)}
            finish = "stop"
        elif not tools or Handler.mode == "text_only":
            message = {"role": "assistant", "content": "Mock prose answer."}
            finish = "stop"
        else:
            fn = pick([t["function"] for t in tools], Handler.mode, rng)
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{Handler.counter}",
                        "type": "function",
                        "function": {"name": fn["name"], "arguments": json.dumps(fill(fn.get("parameters") or {}, Handler.mode, rng))},
                    }
                ],
            }
            finish = "tool_calls"
        self._send(
            200,
            {
                "id": f"mock-{Handler.counter}",
                "object": "chat.completion",
                "model": payload.get("model", "mock"),
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--mode", default="random", choices=MODES)
    args = p.parse_args()
    Handler.mode = args.mode
    print(f"mock server on http://127.0.0.1:{args.port}/v1 mode={args.mode}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
