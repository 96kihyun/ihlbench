#!/usr/bin/env python3
"""K probe: scenario-free knowledge, free text then record_position.

    python scripts/run_probe_k.py --base-url http://127.0.0.1:8003/v1 --model qwen-base \\
        --run qwen-base/probe_k --model-tag qwen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ihl.catalog import data_fingerprint  # noqa: E402
from ihl.llm import assistant_message, chat_completion, server_info  # noqa: E402
from ihl.probe_kj import (  # noqa: E402
    RECORD_SCHEMA,
    SYSTEM_KNOWLEDGE,
    USER_RECORD,
    Topic,
    knowledge_user,
    load_topics,
    parse_record_position,
    score_knowledge,
)
from ihl.runio import RESULTS_DIR, Run, now, trial_seed  # noqa: E402


def trial_id(topic: Topic, model_tag: str, rep: int) -> str:
    return f"K_r{topic.rule_id:03d}_{model_tag}_rep{rep}"


def run_trial(topic: Topic, *, tid: str, seed: int, rep: int, model_tag: str, args: argparse.Namespace) -> dict:
    user1 = knowledge_user(topic.topic)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_KNOWLEDGE},
        {"role": "user", "content": user1},
    ]
    raw1 = chat_completion(
        base_url=args.base_url,
        model=args.model,
        messages=messages,
        tools=None,
        temperature=args.temperature,
        max_tokens=args.max_tokens_free,
        seed=seed,
        api_key=args.api_key,
    )
    msg1 = assistant_message(raw1)
    turn1_text = msg1.get("content") or ""
    messages.append(msg1)
    messages.append({"role": "user", "content": USER_RECORD})
    raw2 = chat_completion(
        base_url=args.base_url,
        model=args.model,
        messages=messages,
        tools=[RECORD_SCHEMA],
        temperature=args.temperature,
        max_tokens=args.max_tokens_record,
        tool_choice="required",
        seed=seed,
        api_key=args.api_key,
    )
    msg2 = assistant_message(raw2)
    parsed = parse_record_position(msg2)
    scored = score_knowledge(parsed)
    rec = {
        "trial_id": tid,
        "ts": now(),
        "model": model_tag,
        "served_model": args.model,
        "rule_id": topic.rule_id,
        "group": topic.group,
        "scenario_ids": list(topic.scenario_ids),
        "rep": rep,
        "seed": seed,
        "topic": topic.topic,
        "turn1_text": turn1_text,
        "record": {
            "status": parsed.get("status"),
            "admits_exception": parsed.get("admits_exception"),
            "instrument": parsed.get("instrument"),
        },
        "scorable": scored["scorable"],
        "knows": scored["knows"],
        "absolute": scored["absolute"],
        "parse_error": parsed.get("parse_error"),
        "n_tool_calls": parsed.get("n_tool_calls"),
        "tool_called": parsed.get("tool_called"),
        "usage": {"free": raw1.get("usage"), "record": raw2.get("usage")},
    }
    if not args.no_raw:
        rec["raw_response"] = {"free": raw1, "record": raw2}
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True, help="served model name")
    p.add_argument("--model-tag", required=True, help="short tag in trial_id / records, e.g. qwen")
    p.add_argument("--api-key", default=None)
    p.add_argument("--run", required=True, help="results/<run>, e.g. qwen-base/probe_k")
    p.add_argument("--rules", type=int, nargs="*", default=None)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens-free", type=int, default=900)
    p.add_argument("--max-tokens-record", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--note", default="")
    p.add_argument("--no-raw", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    topics = [t for t in load_topics() if args.rules is None or t.rule_id in args.rules]
    if args.dry_run:
        t = topics[0]
        print("SYSTEM:", SYSTEM_KNOWLEDGE)
        print("USER1:", knowledge_user(t.topic))
        print("USER2:", USER_RECORD)
        print("tool:", RECORD_SCHEMA["function"]["name"])
        print(f"topics={len(topics)} trials={len(topics) * args.runs}")
        return

    identity = {
        "experiment": "probe_k",
        "format": "explain_then_record",
        "model": args.model,
        "model_tag": args.model_tag,
        "temperature": args.temperature,
        "max_tokens_free": args.max_tokens_free,
        "max_tokens_record": args.max_tokens_record,
        "seed": args.seed,
        "topics": data_fingerprint(),  # scenario data; topics file checked below
        "n_topics": len(topics),
    }
    # fingerprint topics file into identity so a topic edit refuses resume
    topics_path = Path(__file__).resolve().parents[1] / "data" / "k_topics.json"
    identity["topics_sha"] = __import__("hashlib").sha256(topics_path.read_bytes()).hexdigest()[:16]

    extra = {
        "base_url": args.base_url,
        "server": server_info(args.base_url, args.api_key),
        "note": args.note,
        "argv": sys.argv[1:],
    }
    run = Run(RESULTS_DIR / args.run, identity, extra)
    jobs = [(t, r) for r in range(1, args.runs + 1) for t in topics if trial_id(t, args.model_tag, r) not in run.done]
    print(f"run={run.dir} topics={len(topics)} remaining={len(jobs)}", flush=True)
    for i, (topic, rep) in enumerate(jobs, start=1):
        tid = trial_id(topic, args.model_tag, rep)
        rec = run_trial(topic, tid=tid, seed=trial_seed(tid, args.seed), rep=rep, model_tag=args.model_tag, args=args)
        run.write(rec)
        tag = (
            f"knows={rec['knows']} abs={rec['absolute']}"
            if rec["scorable"]
            else f"? {rec['parse_error']}"
        )
        print(f"[{i}/{len(jobs)}] {tid} {tag}", flush=True)


if __name__ == "__main__":
    main()
