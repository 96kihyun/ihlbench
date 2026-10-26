#!/usr/bin/env python3
"""J probe: same dilemma user+tools as baseline D; system asks which course is right.

    python scripts/run_probe_j.py --base-url http://127.0.0.1:8003/v1 --model qwen-base \\
        --run qwen-base/probe_j --model-tag qwen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ihl.catalog import Scenario, data_fingerprint, select  # noqa: E402
from ihl.llm import assistant_message, chat_completion, server_info  # noqa: E402
from ihl.probe_kj import (  # noqa: E402
    assert_j_matches_d_user_and_tools,
    build_judgement_prompt,
    tool_order_label,
)
from ihl.runio import RESULTS_DIR, Run, now, trial_seed  # noqa: E402
from ihl.tools import ORDERS, parse_option_call  # noqa: E402


def trial_id(scenario: Scenario, order_idx: int, model_tag: str, rep: int) -> str:
    return f"J_{scenario.id}_{model_tag}_{tool_order_label(order_idx)}_rep{rep}"


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
    prompt = build_judgement_prompt(scenario, order_idx)
    messages: list[dict] = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.steps[0].user},
    ]
    raw = chat_completion(
        base_url=args.base_url,
        model=args.model,
        messages=messages,
        tools=prompt.steps[0].tools,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tool_choice="required",
        seed=seed,
        api_key=args.api_key,
    )
    message = assistant_message(raw)
    parsed = parse_option_call(scenario, message)
    choice = parsed.get("choice")
    scorable = choice is not None and parsed.get("parse_error") in (None, "missing_rationale")
    # missing_rationale still has a tool name; score the tool, keep the error note
    if parsed.get("parse_error") == "missing_rationale" and choice is not None:
        scorable = True
    if parsed.get("parse_error") in ("no_tool_calls", "unknown_tool", "bad_arguments", "arguments_not_object"):
        scorable = False
        choice = None
    judged = None if not scorable else int(choice == scenario.measured_tool)
    rec = {
        "trial_id": tid,
        "ts": now(),
        "model": model_tag,
        "served_model": args.model,
        "scenario_id": scenario.id,
        "rule_id": scenario.rule_id,
        "group": scenario.group,
        "tool_order": tool_order_label(order_idx),
        "order_idx": order_idx,
        "rep": rep,
        "seed": seed,
        "called_tool": parsed.get("tool_called"),
        "rationale": parsed.get("rationale"),
        "scorable": scorable,
        "judged_violation_right": judged,
        "parse_error": parsed.get("parse_error"),
        "n_tool_calls": parsed.get("n_tool_calls"),
        "prompt": {
            "system": prompt.system,
            "user": prompt.steps[0].user,
            "tools": prompt.steps[0].tools,
        },
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
    p.add_argument("--api-key", default=None)
    p.add_argument("--run", required=True)
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--rules", type=int, nargs="*", default=None)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--orders", type=int, nargs="+", choices=ORDERS, default=list(ORDERS))
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--note", default="")
    p.add_argument("--no-raw", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-assert", action="store_true", help="skip J≡D user/tools check (debug only)")
    args = p.parse_args()

    pool = select(["dilemma"], ids=args.ids, rules=args.rules)
    if not args.skip_assert:
        for s in pool:
            for o in args.orders:
                assert_j_matches_d_user_and_tools(s, o)

    if args.dry_run:
        from ihl.prompt import render_transcript

        print(render_transcript(build_judgement_prompt(pool[0], args.orders[0])))
        print(f"scenarios={len(pool)} trials={len(pool) * len(args.orders) * args.runs}")
        return

    identity = {
        "experiment": "probe_j",
        "format": "judgement",
        "kinds": ["dilemma"],
        "model": args.model,
        "model_tag": args.model_tag,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
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
    jobs = [
        (s, o, r)
        for r in range(1, args.runs + 1)
        for s in pool
        for o in args.orders
        if trial_id(s, o, args.model_tag, r) not in run.done
    ]
    print(f"run={run.dir} scenarios={len(pool)} remaining={len(jobs)}", flush=True)
    for i, (scenario, order_idx, rep) in enumerate(jobs, start=1):
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
        tag = (
            f"viol_right={rec['judged_violation_right']}"
            if rec["scorable"]
            else f"? {rec['parse_error']}"
        )
        print(f"[{i}/{len(jobs)}] {tid} {tag}", flush=True)


if __name__ == "__main__":
    main()
