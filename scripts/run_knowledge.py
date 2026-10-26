#!/usr/bin/env python3
"""Knowledge probe (4.2): one prohibited/permitted question per item x label order x repeat.

    python scripts/run_knowledge.py --base-url http://127.0.0.1:8003/v1 --model qwen-base \\
        --run qwen-base/knowledge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ihl.catalog import KNOWLEDGE_KINDS, KnowledgeItem, data_fingerprint, knowledge_items  # noqa: E402
from ihl.llm import assistant_message, chat_completion, server_info  # noqa: E402
from ihl.probes import KNOWLEDGE_SYSTEM, LABEL_ORDERS, gold_label, knowledge_prompt, label_order, parse_label  # noqa: E402
from ihl.runio import RESULTS_DIR, Run, now, trial_seed  # noqa: E402


def trial_id(item: KnowledgeItem, order_idx: int, run_idx: int) -> str:
    return f"{item.item_id}|o{order_idx}|r{run_idx:02d}"


def run_trial(item: KnowledgeItem, order_idx: int, *, tid: str, seed: int, run_idx: int, args: argparse.Namespace) -> dict:
    prompt = knowledge_prompt(item, order_idx)
    raw = chat_completion(
        base_url=args.base_url,
        model=args.model,
        messages=[{"role": "system", "content": KNOWLEDGE_SYSTEM}, {"role": "user", "content": prompt}],
        tools=None,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=seed,
        api_key=args.api_key,
    )
    response = assistant_message(raw).get("content") or ""
    said, err = parse_label(response)
    gold = gold_label(item)
    rec = {
        "trial_id": tid,
        "ts": now(),
        "item_id": item.item_id,
        "scenario_id": item.scenario_id,
        "rule_id": item.rule_id,
        "group": item.group,
        "item_kind": item.kind,
        "gold": gold,
        "label_order": order_idx,
        "labels": list(label_order(order_idx)),
        "run_idx": run_idx,
        "seed": seed,
        "prompt": prompt,
        "response": response,
        "said": said,
        "said_prohibited": None if said is None else said == "prohibited",
        "correct": None if said is None else said == gold,
        "parse_error": err,
        "finish_reason": raw["choices"][0].get("finish_reason"),
        "usage": raw.get("usage"),
    }
    if not args.no_raw:
        rec["raw_response"] = raw
    return rec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default=None)
    p.add_argument("--run", required=True)
    p.add_argument("--items", nargs="+", choices=KNOWLEDGE_KINDS, default=list(KNOWLEDGE_KINDS))
    p.add_argument("--ids", nargs="*", default=None, help="subset of scenario ids")
    p.add_argument("--rules", type=int, nargs="*", default=None)
    p.add_argument("--orders", type=int, nargs="+", choices=LABEL_ORDERS, default=list(LABEL_ORDERS))
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--note", default="")
    p.add_argument("--no-raw", action="store_true")
    args = p.parse_args()

    items = knowledge_items(args.items, ids=args.ids, rules=args.rules)
    identity = {
        "experiment": "knowledge",
        "format": "label",
        "items": sorted(args.items),
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "data": data_fingerprint(),
    }
    extra = {"base_url": args.base_url, "server": server_info(args.base_url, args.api_key), "note": args.note, "argv": sys.argv[1:]}
    run = Run(RESULTS_DIR / args.run, identity, extra)
    jobs = [
        (k, o, r)
        for r in range(1, args.runs + 1)
        for k in items
        for o in args.orders
        if trial_id(k, o, r) not in run.done
    ]
    print(f"run={run.dir} items={len(items)} remaining={len(jobs)}", flush=True)
    for i, (item, o, r) in enumerate(jobs, start=1):
        tid = trial_id(item, o, r)
        rec = run_trial(item, o, tid=tid, seed=trial_seed(tid, args.seed), run_idx=r, args=args)
        run.write(rec)
        print(f"[{i}/{len(jobs)}] {tid} gold={rec['gold']} said={rec['said'] or '? ' + str(rec['parse_error'])}", flush=True)


if __name__ == "__main__":
    main()
