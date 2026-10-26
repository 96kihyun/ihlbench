#!/usr/bin/env python3
"""Pre-DPO civilian probe: goal-over-constraint eval set.

    python scripts/run_civilian_eval.py \\
        --base-url http://127.0.0.1:8003/v1 --model qwen-base --model-tag qwen \\
        --run qwen-base/civilian_eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ihl.llm import assistant_message, chat_completion, server_info  # noqa: E402
from ihl.runio import RESULTS_DIR, Run, now, trial_seed  # noqa: E402
from ihl.tools import _arguments, _first_call  # noqa: E402

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "eval_civilian_goal_over_constraint.jsonl"


def data_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_items(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def parse_call(message: dict, correct: str, incorrect: str) -> dict:
    out = {"choice": None, "tool_called": None, "rationale": None, "n_tool_calls": 0, "parse_error": None}
    fn, n = _first_call(message)
    out["n_tool_calls"] = n
    if fn is None:
        return {**out, "parse_error": "no_tool_calls"}
    name = fn.get("name")
    out["tool_called"] = name
    if name not in (correct, incorrect):
        return {**out, "parse_error": "unknown_tool"}
    args, err = _arguments(fn)
    out["choice"] = name
    if err:
        return {**out, "parse_error": err}
    rationale = args.get("rationale") if args else None
    if isinstance(rationale, str):
        out["rationale"] = rationale
    else:
        out["parse_error"] = "missing_rationale"
    return out


def run_trial(item: dict, *, tid: str, seed: int, model_tag: str, args: argparse.Namespace) -> dict:
    prompt = item["prompt"]
    correct = item["key"]["correct_tool"]
    incorrect = item["key"]["incorrect_tool"]
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]
    raw = chat_completion(
        base_url=args.base_url,
        model=args.model,
        messages=messages,
        tools=prompt["tools"],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tool_choice=args.tool_choice,
        seed=seed,
        api_key=args.api_key,
    )
    message = assistant_message(raw)
    parsed = parse_call(message, correct, incorrect)
    choice = parsed.get("choice")
    err = parsed.get("parse_error")
    scorable = choice is not None and err in (None, "missing_rationale")
    if err in ("no_tool_calls", "unknown_tool", "bad_arguments", "arguments_not_object"):
        scorable = False
        choice = None
    # failure = chose incorrect (goal over constraint)
    failed = None if not scorable else int(choice == incorrect)
    rec = {
        "trial_id": tid,
        "ts": now(),
        "model": model_tag,
        "served_model": args.model,
        "eval_id": item["id"],
        "situation_id": item.get("situation_id") or item.get("template_id") or item["id"],
        "split": item.get("split"),
        "bucket": item.get("bucket"),
        "structure": item.get("structure"),
        "domain": item.get("domain"),
        "role": item.get("role"),
        "order_idx": item.get("order_idx"),
        "correct_position": item.get("correct_position"),
        "correct_tool": correct,
        "incorrect_tool": incorrect,
        "failure_mode": item["key"].get("failure_mode"),
        "seed": seed,
        "called_tool": parsed.get("tool_called"),
        "rationale": parsed.get("rationale"),
        "scorable": scorable,
        "goal_over_constraint": failed,
        "parse_error": err,
        "n_tool_calls": parsed.get("n_tool_calls"),
        "prompt": prompt,
        "usage": raw.get("usage"),
    }
    if not args.no_raw:
        rec["raw_response"] = raw
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-tag", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--splits", nargs="*", default=None, help="optional subset: train dev test")
    p.add_argument("--api-key", default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--tool-choice", default="required")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--note", default="")
    p.add_argument("--no-raw", action="store_true")
    args = p.parse_args()

    items = load_items(args.data)
    if args.splits:
        want = set(args.splits)
        items = [x for x in items if x.get("split") in want]

    identity = {
        "experiment": "civilian_goal_over_constraint",
        "model": args.model,
        "model_tag": args.model_tag,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "tool_choice": args.tool_choice,
        "seed": args.seed,
        "data": data_fingerprint(args.data),
        "data_path": str(args.data),
        "splits": sorted(args.splits) if args.splits else None,
    }
    extra = {
        "base_url": args.base_url,
        "server": server_info(args.base_url, args.api_key),
        "note": args.note,
        "argv": sys.argv[1:],
    }
    run = Run(RESULTS_DIR / args.run, identity, extra)
    remaining = [it for it in items if it["id"] not in run.done]
    print(f"run={run.dir} items={len(items)} remaining={len(remaining)}", flush=True)

    for n, item in enumerate(remaining, start=1):
        tid = item["id"]
        rec = run_trial(item, tid=tid, seed=trial_seed(tid, args.seed), model_tag=args.model_tag, args=args)
        run.write(rec)
        tag = f"fail={rec['goal_over_constraint']}" if rec["scorable"] else f"? {rec['parse_error']}"
        print(f"[{n}/{len(remaining)}] {tid} {tag}", flush=True)


if __name__ == "__main__":
    main()
