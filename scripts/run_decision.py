#!/usr/bin/env python3
"""Decision experiments: one condition x a set of scenarios x orders x repeats.

    python scripts/run_decision.py --base-url http://127.0.0.1:8003/v1 --model qwen-base \\
        --condition baseline --kinds dilemma --run qwen-base/baseline

    --kinds nocost               no-cost control (4.1)
    --kinds lawful               over-refusal control (4.5); --methods status_change|means_change
    --condition ihl_cue|advisor|prose   alternative explanations (4.4)

Records go to results/<run>/records.jsonl, one per trial. Re-running the same
command resumes; changing anything that defines the run refuses to.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ihl.catalog import KINDS, METHODS, Scenario, data_fingerprint, select  # noqa: E402
from ihl.llm import assistant_message, chat_completion, server_info, tool_result_messages  # noqa: E402
from ihl.prompt import CONDITIONS, TOOL_ACK, DecisionPrompt, build_decision_prompt, render_transcript  # noqa: E402
from ihl.runio import RESULTS_DIR, Run, now, trial_seed  # noqa: E402
from ihl.tools import ORDERS, is_measured, parse_option_call, parse_record_call  # noqa: E402


def trial_id(scenario: Scenario, order_idx: int, run_idx: int) -> str:
    return f"{scenario.kind}|{scenario.id}|o{order_idx}|r{run_idx:02d}"


def run_trial(
    scenario: Scenario,
    prompt: DecisionPrompt,
    *,
    tid: str,
    seed: int,
    run_idx: int,
    args: argparse.Namespace,
) -> dict:
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
        elif step.parse == "record":
            parsed = parse_record_call(scenario, message)
        messages.append(message)
        messages.extend(tool_result_messages(message, TOOL_ACK))

    return {
        "trial_id": tid,
        "ts": now(),
        "condition": prompt.condition,
        "scenario_id": scenario.id,
        "kind": scenario.kind,
        "method": scenario.method,
        "nocost_rewritten": scenario.nocost_rewritten,
        "rule_id": scenario.rule_id,
        "group": scenario.group,
        "victim_count": scenario.victim_count,
        "friendly_loss": scenario.friendly_loss,
        "order_idx": prompt.order_idx,
        "run_idx": run_idx,
        "seed": seed,
        "tool_order": prompt.tool_order,
        "measured_tool": scenario.measured_tool,
        "measured_position": prompt.measured_position,
        "choice": parsed.get("choice"),
        "measured": is_measured(scenario, parsed.get("choice")),
        "rationale": parsed.get("rationale"),
        "parse_error": parsed.get("parse_error"),
        "n_tool_calls": parsed.get("n_tool_calls"),
        "tool_called": parsed.get("tool_called"),
        "prompt": {
            "system": prompt.system,
            "steps": [{"user": s.user, "tools": s.tools} for s in prompt.steps],
        },
        "turns": turns,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True, help="served model name")
    p.add_argument("--api-key", default=None)
    p.add_argument("--run", required=True, help="results/<run>, e.g. qwen-base/baseline")
    p.add_argument("--condition", choices=CONDITIONS, default="baseline")
    p.add_argument("--kinds", nargs="+", choices=KINDS, default=["dilemma"])
    p.add_argument("--ids", nargs="*", default=None, help="subset of scenario ids")
    p.add_argument("--rules", type=int, nargs="*", default=None, help="subset of CIHL rule numbers")
    p.add_argument("--groups", nargs="*", default=None, help="subset of groups, e.g. G1 G10")
    p.add_argument("--methods", nargs="*", choices=METHODS, default=None, help="lawful variants only")
    p.add_argument("--runs", type=int, default=5, help="repeats per scenario x order")
    p.add_argument("--orders", type=int, nargs="+", choices=ORDERS, default=list(ORDERS))
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--tool-choice", default="required")
    p.add_argument("--seed", type=int, default=0, help="base seed; per-trial seeds derive from it")
    p.add_argument("--note", default="", help="free text stored in the manifest (adapter, training seed, ...)")
    p.add_argument("--no-raw", action="store_true", help="omit raw server responses")
    p.add_argument("--dry-run", action="store_true", help="print one transcript per scenario and exit")
    args = p.parse_args()

    if args.condition == "prose" and "lawful" in args.kinds:
        p.error("prose is an IHL format control; run lawful scenarios under baseline")
    pool = select(args.kinds, ids=args.ids, rules=args.rules, groups=args.groups, methods=args.methods)
    jobs = [(s, o, r) for r in range(1, args.runs + 1) for s in pool for o in args.orders]

    if args.dry_run:
        for s in pool:
            print(render_transcript(build_decision_prompt(s, args.condition, args.orders[0])))
            print()
        print(f"scenarios={len(pool)} trials={len(jobs)}")
        return

    identity = {
        "experiment": "decision",
        "condition": args.condition,
        "kinds": sorted(args.kinds),
        "methods": sorted(args.methods) if args.methods else None,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "tool_choice": args.tool_choice,
        "seed": args.seed,
        "data": data_fingerprint(),
    }
    extra = {"base_url": args.base_url, "server": server_info(args.base_url, args.api_key), "note": args.note, "argv": sys.argv[1:]}
    run = Run(RESULTS_DIR / args.run, identity, extra)
    remaining = [j for j in jobs if trial_id(*j) not in run.done]
    print(f"run={run.dir} condition={args.condition} scenarios={len(pool)} trials={len(jobs)} remaining={len(remaining)}", flush=True)

    for n, (scenario, order_idx, run_idx) in enumerate(remaining, start=1):
        tid = trial_id(scenario, order_idx, run_idx)
        prompt = build_decision_prompt(scenario, args.condition, order_idx)
        rec = run_trial(scenario, prompt, tid=tid, seed=trial_seed(tid, args.seed), run_idx=run_idx, args=args)
        run.write(rec)
        tag = {True: "MEASURED", False: "other", None: f"? {rec['parse_error']}"}[rec["measured"]]
        print(f"[{n}/{len(remaining)}] {tid} {tag}", flush=True)


if __name__ == "__main__":
    main()
