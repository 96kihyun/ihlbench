#!/usr/bin/env python3
"""Print exactly what the model sees, without calling a model.

    python scripts/dump_prompts.py --ids 47-1
    python scripts/dump_prompts.py --ids 1-1 --kinds lawful --condition advisor
    python scripts/dump_prompts.py --knowledge --ids 1-1
    python scripts/dump_prompts.py --all > prompts.txt

Read this before running anything: a fact list that leaks a third option makes
the scenario measure nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ihl.catalog import KINDS, filter_entries, knowledge_items, select  # noqa: E402
from ihl.probes import KNOWLEDGE_SYSTEM, LABEL_ORDERS, REFUSAL_SYSTEM, gold_label, knowledge_prompt  # noqa: E402
from ihl.prompt import CONDITIONS, build_decision_prompt, render_transcript  # noqa: E402
from ihl.tools import ORDERS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=CONDITIONS, default="baseline")
    p.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--rules", type=int, nargs="*", default=None)
    p.add_argument("--order", type=int, choices=ORDERS, default=0)
    p.add_argument("--knowledge", action="store_true", help="print the knowledge and refusal probes instead")
    p.add_argument("--all", action="store_true", help="every condition and both orders")
    args = p.parse_args()

    if args.knowledge:
        print(f"--- SYSTEM (knowledge) --- {KNOWLEDGE_SYSTEM}\n")
        orders = LABEL_ORDERS if args.all else (args.order,)
        for k in knowledge_items(ids=args.ids, rules=args.rules):
            for o in orders:
                print(f"[{k.item_id} | {k.kind} | gold {gold_label(k)} | o{o}] {knowledge_prompt(k, o)}")
        print(f"\n--- SYSTEM (refusal) --- {REFUSAL_SYSTEM}\n")
        for e in filter_entries(args.ids, args.rules, None):
            print(f"[{e.id}] {e.refusal_request}")
        return

    conditions = CONDITIONS if args.all else (args.condition,)
    orders = ORDERS if args.all else (args.order,)
    for scenario in select(args.kinds, ids=args.ids, rules=args.rules):
        for condition in conditions:
            if condition == "prose" and scenario.kind == "lawful":
                continue
            for order in orders:
                print(render_transcript(build_decision_prompt(scenario, condition, order)))
                print()


if __name__ == "__main__":
    main()
