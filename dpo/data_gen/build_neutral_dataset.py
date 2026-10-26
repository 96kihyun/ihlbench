#!/usr/bin/env python3
"""Build value-unrelated convention preference pairs.

Same count and format as the main DPO set (288 train / 72 holdout), but
chosen/rejected is house style only. No ethics, no goal ranking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_preference_dataset import expand, summarize, write_jsonl
from neutral_templates import NEUTRAL, NEUTRAL_HOLDOUT_IDS, NEUTRAL_SLOT_VARIANTS

CHOSEN_R = [
    "House convention names this option. I follow it.",
    "The style guide already picked this one. I stay with it.",
    "This is the house default for this class of work. I use it.",
]
REJECTED_R = [
    "I ignore the house convention and take the other equally valid option.",
    "I set the style guide aside and pick the other format.",
    "I skip the house default and use the other allowed option.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "controls"))
    args = ap.parse_args()
    out = Path(args.out)

    # Monkey-patch rationale pools via a thin local expand that sets them.
    import build_preference_dataset as bpd

    orig = bpd.build_item

    def build_item(seed, slots, variant_idx, order_idx, bucket, subtype, split):
        row = orig(seed, slots, variant_idx, order_idx, bucket, subtype, split)
        i = variant_idx
        chosen_r = CHOSEN_R[i % len(CHOSEN_R)]
        rejected_r = REJECTED_R[i % len(REJECTED_R)]
        row["chosen_rationale"] = chosen_r
        row["rejected_rationale"] = rejected_r
        row["chosen"] = bpd.completion(row["chosen_tool"], chosen_r)
        row["rejected"] = bpd.completion(row["rejected_tool"], rejected_r)
        row["control"] = "unrelated"
        return row

    bpd.build_item = build_item
    recs = expand(
        NEUTRAL, NEUTRAL_HOLDOUT_IDS, "unrelated", "convention", NEUTRAL_SLOT_VARIANTS
    )
    bpd.build_item = orig

    train = [r for r in recs if r["split"] == "train"]
    hold = [r for r in recs if r["split"] == "holdout"]
    write_jsonl(out / "unrelated_train.jsonl", train)
    write_jsonl(out / "unrelated_holdout.jsonl", hold)

    text = summarize(recs) + f"\ncontrol=unrelated train={len(train)} holdout={len(hold)}\n"
    (out / "unrelated_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
