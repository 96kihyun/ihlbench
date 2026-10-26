#!/usr/bin/env python3
"""Grade a run with an LLM judge. The subject model must not judge itself.

    python scripts/judge.py --task refusal --run qwen-base/refusal \\
        --judge-base-url https://... --judge-model ... --api-key $KEY
    python scripts/judge.py --task rationale --run qwen-base/baseline ...

Writes results/<run>/judgments/<task>__<judge-model>.jsonl, append-only and
resumable on trial id.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ihl.judges import judgments_path, load_task  # noqa: E402
from ihl.llm import assistant_message, chat_completion  # noqa: E402
from ihl.runio import RESULTS_DIR, load_run, now, read_jsonl  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="data/judges/<task>.yaml")
    p.add_argument("--run", required=True)
    p.add_argument("--judge-base-url", required=True)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY"))
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--allow-self-judge", action="store_true")
    args = p.parse_args()

    task = load_task(args.task)
    run_dir = RESULTS_DIR / args.run
    manifest, records = load_run(run_dir)
    if manifest["identity"]["experiment"] != task.applies_to:
        raise SystemExit(f"task {task.name} applies to {task.applies_to} runs, not {manifest['identity']['experiment']}")
    if manifest["identity"]["model"] == args.judge_model and not args.allow_self_judge:
        raise SystemExit("judge model is the subject model; pass --allow-self-judge to do it anyway")

    out = judgments_path(run_dir, task.name, args.judge_model)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["trial_id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in records if r["trial_id"] not in done and all(r.get(f) is not None for f in task.fields)]
    print(f"task={task.name} run={run_dir} records={len(records)} remaining={len(todo)} out={out}", flush=True)
    with out.open("a", encoding="utf-8") as f:
        for i, rec in enumerate(todo, start=1):
            raw = chat_completion(
                base_url=args.judge_base_url,
                model=args.judge_model,
                messages=[{"role": "system", "content": task.system}, {"role": "user", "content": task.render(rec)}],
                tools=None,
                temperature=0.0,
                max_tokens=args.max_tokens,
                api_key=args.api_key,
            )
            text = assistant_message(raw).get("content")
            row = {
                "trial_id": rec["trial_id"],
                "label": task.parse(text),
                "raw": text,
                "judge_model": args.judge_model,
                "task": task.name,
                "ts": now(),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {rec['trial_id']} -> {row['label']}", flush=True)


if __name__ == "__main__":
    main()
