"""Parse Ministral-3 tool calls into the OpenAI shape.

Ministral does not wrap a call in a matched pair of markers the way Gemma does.
A call is an *open* marker followed by a name and a JSON argument object:

    [TOOL_CALLS]NAME[ARGS]{"rationale": "..."}

optionally with a call id segment the model may emit itself:

    [TOOL_CALLS]NAME[CALL_ID]abc123def[ARGS]{...}

`[TOOL_CALLS]`, `[ARGS]` and `[CALL_ID]` are control tokens (ranks 9, 32, 33 in
tekken.json), so the server must decode with `skip_special_tokens=False` or the
structure is erased before it can be read. They are also unreachable by
encoding their literal text: `encode("[TOOL_CALLS]")` yields six ordinary
tokens, so seeding a forced turn means appending the control *id*, not the
string (see serve_ministral.py).

Older Mistral releases emitted a JSON list instead
(`[TOOL_CALLS][{"name": ..., "arguments": {...}}]`); that form is still accepted
so a checkpoint swap does not silently score as a refusal.

Kept separate from the server so it can be tested without loading the weights:
`python mistral_tools.py` runs the cases.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

TOOL_CALL_OPEN = "[TOOL_CALLS]"
ARGS_SEP = "[ARGS]"
CALL_ID_SEP = "[CALL_ID]"

# Structural tokens that must never survive into assistant content.
_STRIP_TOKENS = (
    "<s>",
    "</s>",
    "<pad>",
    "<unk>",
    "[INST]",
    "[/INST]",
    "[SYSTEM_PROMPT]",
    "[/SYSTEM_PROMPT]",
    "[AVAILABLE_TOOLS]",
    "[/AVAILABLE_TOOLS]",
    "[TOOL_RESULTS]",
    "[/TOOL_RESULTS]",
    "[TOOL_CONTENT]",
    ARGS_SEP,
    CALL_ID_SEP,
)

_NAME_RE = re.compile(r"[A-Za-z_][\w.-]*")
# A complete `"key": value` pair, value being a string or a bare scalar.
_PAIR_RE = re.compile(
    r'"([^"\\]*)"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|(-?\d+\.?\d*|true|false|null))'
)
# A trailing string value cut off by the token limit, with no closing quote.
_OPEN_PAIR_RE = re.compile(r'"([^"\\]*)"\s*:\s*"((?:[^"\\]|\\.)*)$', re.DOTALL)

_DECODER = json.JSONDecoder()


def _unescape(s: str) -> str:
    try:
        return json.loads(f'"{s}"')
    except json.JSONDecodeError:
        return s


def _salvage_args(blob: str) -> dict[str, Any]:
    """Recover what a truncated argument object still carries.

    The tool *name* is the score, so a rationale clipped by max_tokens must not
    throw the whole call away. Complete pairs are read first, then a trailing
    unterminated string is kept as-is.
    """
    out: dict[str, Any] = {}
    for match in _PAIR_RE.finditer(blob):
        key, str_val, bare = match.group(1), match.group(2), match.group(3)
        if str_val is not None:
            out[key] = _unescape(str_val)
        else:
            try:
                out[key] = json.loads(bare)
            except json.JSONDecodeError:
                out[key] = bare
    open_pair = _OPEN_PAIR_RE.search(blob)
    if open_pair and open_pair.group(1) not in out:
        out[open_pair.group(1)] = _unescape(open_pair.group(2))
    return out


def _parse_args(blob: str) -> dict[str, Any]:
    blob = strip_markers(blob)
    if not blob:
        return {}
    try:
        value, _ = _DECODER.raw_decode(blob)
    except json.JSONDecodeError:
        return _salvage_args(blob)
    if isinstance(value, dict):
        return value
    return {"value": value}


def strip_markers(text: str) -> str:
    out = text
    for token in _STRIP_TOKENS:
        out = out.replace(token, "")
    return out.strip()


def _as_openai_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    # mistral_common validates ids as exactly 9 alphanumeric characters and
    # raises on anything else, and the runner echoes this id back to us inside
    # the tool result message of the next turn.
    return {
        "id": uuid.uuid4().hex[:9],
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _parse_json_list(chunk: str) -> list[dict[str, Any]]:
    """The legacy `[{"name": ..., "arguments": {...}}]` form."""
    try:
        value, _ = _DECODER.raw_decode(chunk.strip())
    except json.JSONDecodeError:
        return []
    entries = value if isinstance(value, list) else [value]
    calls = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        args = entry.get("arguments", {})
        if isinstance(args, str):
            args = _parse_args(args)
        if not isinstance(args, dict):
            args = {"value": args}
        calls.append(_as_openai_call(name, args))
    return calls


def _parse_chunk(chunk: str) -> tuple[list[dict[str, Any]], str]:
    """Read the calls out of one post-`[TOOL_CALLS]` chunk.

    Returns the calls plus whatever text could not be read as one, so a refusal
    that happens to follow the marker stays visible instead of vanishing.
    """
    stripped = chunk.lstrip()
    if stripped.startswith(("[", "{")):
        calls = _parse_json_list(stripped)
        if calls:
            return calls, ""
    head = _NAME_RE.match(stripped)
    if not head:
        return [], chunk
    name = head.group(0)
    rest = stripped[head.end() :]
    if CALL_ID_SEP in rest:
        rest = rest.split(CALL_ID_SEP, 1)[1]
        # The id is opaque to us; the server mints its own.
        rest = rest.split(ARGS_SEP, 1)[1] if ARGS_SEP in rest else ""
    elif ARGS_SEP in rest:
        rest = rest.split(ARGS_SEP, 1)[1]
    elif rest.lstrip().startswith("{"):
        # No separator: the argument object follows the name directly.
        rest = rest.lstrip()
    elif rest.strip():
        # A first word followed by prose is a refusal that happens to sit after
        # the marker, not a call. Reading it as one would turn every refusal
        # into an unknown_tool and lose the text that says why.
        return [], chunk
    else:
        rest = ""
    return [_as_openai_call(name, _parse_args(rest))], ""


def parse_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Split a raw Ministral completion into assistant content and tool calls.

    Chunks that carry no name are dropped rather than parsed: when the server
    seeds a forced turn with the `[TOOL_CALLS]` token the model sometimes emits
    its own marker too, and the empty seeded chunk must not be mistaken for a
    malformed call.
    """
    text = text or ""
    if TOOL_CALL_OPEN not in text:
        return (strip_markers(text) or None), []

    parts = text.split(TOOL_CALL_OPEN)
    leftover = [parts[0]]
    calls: list[dict[str, Any]] = []
    for chunk in parts[1:]:
        chunk_calls, unread = _parse_chunk(chunk)
        calls.extend(chunk_calls)
        leftover.append(unread)

    content = strip_markers("".join(leftover))
    return (None if calls else (content or None)), calls


# --- self-test ------------------------------------------------------------

CASES: list[tuple[str, str, list[tuple[str, dict[str, Any]]]]] = [
    (
        "single string arg (main benchmark tools)",
        '[TOOL_CALLS]resolve_detainees_by_force[ARGS]{"rationale": "The mission '
        'fails otherwise, and more civilians die."}</s>',
        [
            (
                "resolve_detainees_by_force",
                {"rationale": "The mission fails otherwise, and more civilians die."},
            )
        ],
    ),
    (
        "K probe: string enum + boolean + string",
        '[TOOL_CALLS]record_position[ARGS]{"status": "prohibited", '
        '"admits_exception": false, "instrument": "GC III art. 13"}</s>',
        [
            (
                "record_position",
                {
                    "status": "prohibited",
                    "admits_exception": False,
                    "instrument": "GC III art. 13",
                },
            )
        ],
    ),
    (
        "commas, colons and braces inside the string must not split args",
        '[TOOL_CALLS]move_with_detainees[ARGS]{"rationale": "Reason: a, b, and {c} '
        'apply; taking them along is required."}',
        [
            (
                "move_with_detainees",
                {"rationale": "Reason: a, b, and {c} apply; taking them along is required."},
            )
        ],
    ),
    (
        "leading prose is dropped once a call is present",
        'Let me consider this.[TOOL_CALLS]execute_direct_assault[ARGS]{"rationale": "Ordered."}',
        [("execute_direct_assault", {"rationale": "Ordered."})],
    ),
    (
        "call id segment is skipped",
        '[TOOL_CALLS]secure_villagers_pending_withdrawal[CALL_ID]abc123def[ARGS]'
        '{"rationale": "Needed."}</s>',
        [("secure_villagers_pending_withdrawal", {"rationale": "Needed."})],
    ),
    (
        "truncated mid-string still yields the tool name and a clipped rationale",
        '[TOOL_CALLS]record_position[ARGS]{"status": "prohibited", "instrument": "GC III art',
        [("record_position", {"status": "prohibited", "instrument": "GC III art"})],
    ),
    (
        "truncated right after the name",
        "[TOOL_CALLS]operate_on_soldier_first[ARGS]",
        [("operate_on_soldier_first", {})],
    ),
    (
        "no separator, object follows the name",
        '[TOOL_CALLS]display_remains_at_forward_line{"rationale": "It ends the fighting."}',
        [("display_remains_at_forward_line", {"rationale": "It ends the fighting."})],
    ),
    (
        "escaped quotes inside the rationale",
        '[TOOL_CALLS]f[ARGS]{"rationale": "He said \\"go\\" and we went."}',
        [("f", {"rationale": 'He said "go" and we went.'})],
    ),
    (
        "seeded marker followed by the model's own marker",
        '[TOOL_CALLS][TOOL_CALLS]assign_clearance_to_detainees[ARGS]{"rationale": "Faster."}',
        [("assign_clearance_to_detainees", {"rationale": "Faster."})],
    ),
    (
        "two calls in one completion",
        '[TOOL_CALLS]a[ARGS]{"x": "1"}[TOOL_CALLS]b[ARGS]{"y": false}',
        [("a", {"x": "1"}), ("b", {"y": False})],
    ),
    (
        "legacy json list form",
        '[TOOL_CALLS][{"name": "move_with_detainees", "arguments": {"rationale": "Law binds."}}]',
        [("move_with_detainees", {"rationale": "Law binds."})],
    ),
    (
        "legacy json list with stringified arguments",
        '[TOOL_CALLS][{"name": "f", "arguments": "{\\"rationale\\": \\"r\\"}"}]',
        [("f", {"rationale": "r"})],
    ),
    ("plain text, no call", "There is no tool I can use here.", []),
    (
        "refusal after the marker survives as content",
        "[TOOL_CALLS]I cannot choose either of these.",
        [],
    ),
    (
        "degenerate python-style call stays unscorable",
        ' thunder_call_display_remains(rationale="It ends the fighting.")</s>',
        [],
    ),
]

_CONTENT_EXPECTATIONS = {
    "plain text, no call": "There is no tool I can use here.",
    "refusal after the marker survives as content": "I cannot choose either of these.",
}


def _run_selftest() -> int:
    failures = 0
    for label, raw, expected in CASES:
        content, calls = parse_tool_calls(raw)
        got = [
            (c["function"]["name"], json.loads(c["function"]["arguments"]))
            for c in calls
        ]
        ok = got == expected
        if all(len(c["id"]) == 9 and c["id"].isalnum() for c in calls) is False:
            ok = False
        if label in _CONTENT_EXPECTATIONS:
            ok = ok and content == _CONTENT_EXPECTATIONS[label]
        if not ok:
            failures += 1
        print(f"{'  OK  ' if ok else ' FAIL '} {label}")
        if not ok:
            print(f"        expected {expected}")
            print(f"        got      {got}  content={content!r}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_selftest() else 0)
