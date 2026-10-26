#!/usr/bin/env python3
"""Keyword-grade a refusal run. Binary: refuse / not_refuse.

    python scripts/score_refusal_kw.py qwen-base/refusal gemma/refusal ministral/refusal

Writes results/<run>/judgments/refusal__keyword.jsonl (overwrite).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ihl.judges import judgments_path  # noqa: E402
from ihl.refusal_kw import label, opening  # noqa: E402
from ihl.runio import RESULTS_DIR, load_run, now  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    args = p.parse_args()
    for name in args.runs:
        run_dir = RESULTS_DIR / name
        manifest, records = load_run(run_dir)
        if manifest["identity"]["experiment"] != "refusal":
            raise SystemExit(f"{name} is {manifest['identity']['experiment']}, not refusal")
        out = judgments_path(run_dir, "refusal", "keyword")
        out.parent.mkdir(parents=True, exist_ok=True)
        n_refuse = 0
        with out.open("w", encoding="utf-8") as f:
            for rec in records:
                lab = label(rec.get("response") or "")
                if lab == "refuse":
                    n_refuse += 1
                row = {
                    "trial_id": rec["trial_id"],
                    "label": lab,
                    "raw": opening(rec.get("response") or ""),
                    "judge_model": "keyword",
                    "task": "refusal",
                    "ts": now(),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}  refuse {n_refuse}/{len(records)}  -> {out}", flush=True)


if __name__ == "__main__":
    main()
