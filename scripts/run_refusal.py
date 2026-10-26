#!/usr/bin/env python3
"""Refusal probe (4.2): the scenario's direct order to carry out the prohibited act, no role, no tools.

    python scripts/run_refusal.py --base-url http://127.0.0.1:8003/v1 --model qwen-base \\
        --run qwen-base/refusal

Generation only. Binary keyword grade (refuse / not_refuse):

    python scripts/score_refusal_kw.py qwen-base/refusal

Optional LLM judge (4-way rubric) is a separate step:

    python scripts/judge.py --task refusal --run qwen-base/refusal --judge-base-url ... --judge-model ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ihl.catalog import filter_entries, data_fingerprint  # noqa: E402
from ihl.llm import assistant_message, chat_completion, server_info  # noqa: E402
from ihl.probes import REFUSAL_SYSTEM  # noqa: E402
from ihl.runio import RESULTS_DIR, Run, now, trial_seed  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default=None)
    p.add_argument("--run", required=True)
    p.add_argument("--ids", nargs="*", default=None, help="subset of scenario ids")
    p.add_argument("--rules", type=int, nargs="*", default=None)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--note", default="")
    p.add_argument("--no-raw", action="store_true")
    args = p.parse_args()

    identity = {
        "experiment": "refusal",
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "data": data_fingerprint(),
    }
    extra = {"base_url": args.base_url, "server": server_info(args.base_url, args.api_key), "note": args.note, "argv": sys.argv[1:]}
    run = Run(RESULTS_DIR / args.run, identity, extra)
    entries = filter_entries(args.ids, args.rules, None)
    tid = lambda e, r: f"{e.id}|r{r:02d}"  # noqa: E731
    jobs = [(e, r) for r in range(1, args.runs + 1) for e in entries if tid(e, r) not in run.done]
    print(f"run={run.dir} requests={len(entries)} remaining={len(jobs)}", flush=True)
    for i, (entry, r) in enumerate(jobs, start=1):
        t = tid(entry, r)
        seed = trial_seed(t, args.seed)
        raw = chat_completion(
            base_url=args.base_url,
            model=args.model,
            messages=[{"role": "system", "content": REFUSAL_SYSTEM}, {"role": "user", "content": entry.refusal_request}],
            tools=None,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=seed,
            api_key=args.api_key,
        )
        rec = {
            "trial_id": t,
            "ts": now(),
            "scenario_id": entry.id,
            "rule_id": entry.rule_id,
            "group": entry.group,
            "request": entry.refusal_request,
            "run_idx": r,
            "seed": seed,
            "system": REFUSAL_SYSTEM,
            "response": assistant_message(raw).get("content") or "",
            "finish_reason": raw["choices"][0].get("finish_reason"),
            "usage": raw.get("usage"),
        }
        if not args.no_raw:
            rec["raw_response"] = raw
        run.write(rec)
        print(f"[{i}/{len(jobs)}] {t} len={len(rec['response'])}", flush=True)


if __name__ == "__main__":
    main()
