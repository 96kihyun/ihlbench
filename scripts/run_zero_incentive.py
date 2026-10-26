#!/usr/bin/env python3
"""Zero-incentive floor: baseline decision prompt on zero_incentive scenarios.

    python scripts/run_zero_incentive.py --base-url http://127.0.0.1:8003/v1 \\
        --model qwen-base --model-tag qwen --run qwen-base/zero_incentive

180 × 2 orders × 5 reps = 1,800 trials per model. Aggregated dump:
results/zero_incentive.jsonl (all models) via scripts/aggregate_zero_incentive.py
or this script's --also-aggregate flag after each write.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ihl.catalog import Scenario, data_fingerprint, select  # noqa: E402
from ihl.llm import assistant_message, chat_completion, server_info, tool_result_messages  # noqa: E402
from ihl.probe_kj import tool_order_label  # noqa: E402
from ihl.prompt import TOOL_ACK, build_decision_prompt  # noqa: E402
from ihl.runio import RESULTS_DIR, Run, now, trial_seed  # noqa: E402
from ihl.tools import ORDERS, parse_option_call  # noqa: E402

_ORDER_SHORT = {0: "vfirst", 1: "cfirst"}


def trial_id(scenario: Scenario, order_idx: int, model_tag: str, rep: int) -> str:
    return f"Z_{scenario.id}_{model_tag}_{_ORDER_SHORT[order_idx]}_rep{rep}"


def run_trial(
    scenario: Scenario,
    order_idx: int,
    *,
    tid: str,
    seed: int,
    rep: int,
    model_tag: str,
    args: argparse.Namespace,
) -> dict:
    prompt = build_decision_prompt(scenario, "baseline", order_idx)
    messages: list[dict] = [{"role": "system", "content": prompt.system}]
    turns: list[dict] = []
    parsed: dict = {}
    for step in prompt.steps:
        messages.append({"role": "user", "content": step.user})
        raw = chat_completion(
            base_url=args.base_url,
            model=args.model,
            messages=messages,
            tools=step.tools,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            tool_choice=args.tool_choice,
            seed=seed,
            api_key=args.api_key,
        )
        message = assistant_message(raw)
        turn = {
            "user": step.user,
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
            "finish_reason": raw["choices"][0].get("finish_reason"),
            "usage": raw.get("usage"),
        }
        if not args.no_raw:
            turn["raw_response"] = raw
        turns.append(turn)
        if step.parse == "option":
            parsed = parse_option_call(scenario, message)
        messages.append(message)
        messages.extend(tool_result_messages(message, TOOL_ACK))

    choice = parsed.get("choice")
    err = parsed.get("parse_error")
    scorable = choice is not None and err in (None, "missing_rationale")
    if err in ("no_tool_calls", "unknown_tool", "bad_arguments", "arguments_not_object"):
        scorable = False
        choice = None
    violation = None if not scorable else int(choice == scenario.measured_tool)
    return {
        "trial_id": tid,
        "ts": now(),
        "model": model_tag,
        "served_model": args.model,
        "scenario_id": scenario.id,
        "rule_id": scenario.rule_id,
        "group": scenario.group,
        "type": scenario.zero_type,
        "kind": "zero_incentive",
        "tool_order": tool_order_label(order_idx),
        "order_idx": order_idx,
        "rep": rep,
        "seed": seed,
        "called_tool": parsed.get("tool_called"),
        "rationale": parsed.get("rationale"),
        "scorable": scorable,
        "violation": violation,
        "parse_error": err,
        "n_tool_calls": parsed.get("n_tool_calls"),
        "prompt": {
            "system": prompt.system,
            "steps": [{"user": s.user, "tools": s.tools} for s in prompt.steps],
        },
        "turns": turns,
    }


def append_aggregate(rec: dict) -> None:
    """Append slim handoff row to results/zero_incentive.jsonl (dedupe by trial_id)."""
    slim = {
        "trial_id": rec["trial_id"],
        "model": rec["model"],
        "scenario_id": rec["scenario_id"],
        "rule_id": rec["rule_id"],
        "group": rec["group"],
        "type": rec["type"],
        "tool_order": rec["tool_order"],
        "rep": rec["rep"],
        "called_tool": rec["called_tool"],
        "rationale": rec["rationale"],
        "scorable": rec["scorable"],
        "violation": rec["violation"],
    }
    path = RESULTS_DIR / "zero_incentive.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                seen.add(json.loads(line)["trial_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    if slim["trial_id"] in seen:
        return
    with path.open("a") as f:
        f.write(json.dumps(slim, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-tag", required=True, help="short tag in trial_id: qwen|gemma|ministral")
    p.add_argument("--api-key", default=None)
    p.add_argument("--run", required=True, help="results/<run>, e.g. qwen-base/zero_incentive")
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--rules", type=int, nargs="*", default=None)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--orders", type=int, nargs="+", choices=ORDERS, default=list(ORDERS))
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--tool-choice", default="required")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--note", default="")
    p.add_argument("--no-raw", action="store_true")
    p.add_argument("--also-aggregate", action="store_true", help="append slim rows to results/zero_incentive.jsonl")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    pool = select(["zero_incentive"], ids=args.ids, rules=args.rules)
    jobs = [
        (s, o, r)
        for r in range(1, args.runs + 1)
        for s in pool
        for o in args.orders
    ]

    if args.dry_run:
        from ihl.prompt import render_transcript

        print(render_transcript(build_decision_prompt(pool[0], "baseline", args.orders[0])))
        print(f"scenarios={len(pool)} trials={len(jobs)}")
        return

    identity = {
        "experiment": "zero_incentive",
        "condition": "baseline",
        "kinds": ["zero_incentive"],
        "model": args.model,
        "model_tag": args.model_tag,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "tool_choice": args.tool_choice,
        "seed": args.seed,
        "data": data_fingerprint(),
    }
    extra = {
        "base_url": args.base_url,
        "server": server_info(args.base_url, args.api_key),
        "note": args.note,
        "argv": sys.argv[1:],
    }
    run = Run(RESULTS_DIR / args.run, identity, extra)
    remaining = [j for j in jobs if trial_id(*j[:2], args.model_tag, j[2]) not in run.done]
    print(
        f"run={run.dir} scenarios={len(pool)} trials={len(jobs)} remaining={len(remaining)}",
        flush=True,
    )
    for n, (scenario, order_idx, rep) in enumerate(remaining, start=1):
        tid = trial_id(scenario, order_idx, args.model_tag, rep)
        rec = run_trial(
            scenario,
            order_idx,
            tid=tid,
            seed=trial_seed(tid, args.seed),
            rep=rep,
            model_tag=args.model_tag,
            args=args,
        )
        run.write(rec)
        if args.also_aggregate:
            append_aggregate(rec)
        tag = f"viol={rec['violation']}" if rec["scorable"] else f"? {rec['parse_error']}"
        print(f"[{n}/{len(remaining)}] {tid} {tag}", flush=True)


if __name__ == "__main__":
    main()
