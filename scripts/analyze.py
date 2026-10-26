#!/usr/bin/env python3
"""Summarize runs with design-aware uncertainty.

    python scripts/analyze.py qwen-base/baseline
    python scripts/analyze.py qwen-base/ihl_cue --ref qwen-base/baseline
    python scripts/analyze.py qwen-base/baseline --knowledge qwen-base/knowledge
    python scripts/analyze.py qwen-base/knowledge qwen-base/refusal

Rates carry a 95% bootstrap CI that resamples clusters (--cluster, default
rule_id: the three scenarios of a rule are variations on one norm, so
resampling scenarios would overstate precision). Paired contrasts (tool
order, --ref) pair on scenario_id, since both runs share the scenarios.

Decision runs: measured-option rate (violation; over-refusal for lawful,
split by method), tool-order effect, rate by group and by victim count, and
--detail for the per-rule table. --knowledge splits the violation rate by
whether the model knows the rule for that scenario (K-yes and K-no both
answered correctly in the majority of runs): the knowledge-action gap.
Judge files under <run>/judgments/ are summarized when present, with Cohen's
kappa when two judges graded the same task.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ihl.runio import RESULTS_DIR, load_run, read_jsonl  # noqa: E402
from ihl.stats import by, paired_contrast, rate  # noqa: E402

VICTIM_BINS = ((1, 1), (2, 9), (10, 99), (100, 999), (1000, None))


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def judgments(run_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
    """task -> judge -> trial_id -> label"""
    out: dict[str, dict[str, dict[str, str]]] = {}
    for path in sorted((run_dir / "judgments").glob("*.jsonl")) if (run_dir / "judgments").exists() else []:
        task, judge = path.stem.split("__", 1)
        out.setdefault(task, {})[judge] = {r["trial_id"]: r["label"] for r in read_jsonl(path)}
    return out


def print_agreement(task: str, judges: dict[str, dict[str, str]]) -> None:
    for (ja, la), (jb, lb) in combinations(judges.items(), 2):
        shared = sorted(set(la) & set(lb))
        k = cohen_kappa([str(la[t]) for t in shared], [str(lb[t]) for t in shared])
        print(f"  agreement {task}: {ja} vs {jb}  n={len(shared)}  kappa={k:.3f}")


def _kn(rows: list[dict], outcome: str = "measured") -> str:
    k = sum(1 for x in rows if x[outcome])
    n = sum(1 for x in rows if x[outcome] is not None)
    return f"{k}/{n}"


def victim_bin(v: int) -> str:
    for lo, hi in VICTIM_BINS:
        if v >= lo and (hi is None or v <= hi):
            return f"{lo}+" if hi is None else (f"{lo}" if lo == hi else f"{lo}-{hi}")
    return "?"


def knows_by_scenario(records: list[dict]) -> dict[str, bool]:
    """scenario -> K-yes and K-no each answered correctly in more than half of the scored runs."""
    out = {}
    for sid, rows in by(records, "scenario_id").items():
        ok = True
        for kind in ("yes", "no"):
            scored = [x["correct"] for x in rows if x["item_kind"] == kind and x["correct"] is not None]
            ok = ok and bool(scored) and sum(scored) > len(scored) / 2
        out[sid] = ok
    return out


def decision(records: list[dict], ref: list[dict] | None, args: argparse.Namespace, knows: dict[str, bool] | None) -> None:
    """One section per kind; lawful gets the pooled rate and then one section per method."""
    for kind, rows in sorted(by(records, "kind").items(), key=lambda kv: ("dilemma", "nocost", "lawful").index(kv[0])):
        kref = [x for x in ref if x["kind"] == kind] if ref else None
        if kind != "lawful":
            print(f"  -- {kind}")
            summarize(rows, kref, args, knows)
            continue
        print(f"  -- lawful, both methods: over-refusal {rate(rows, cluster=args.cluster).fmt()}")
        for method, mrows in sorted(by(rows, "method").items()):
            print(f"  -- lawful / {method}")
            summarize(mrows, [x for x in kref if x["method"] == method] if kref else None, args, knows)


def summarize(records: list[dict], ref: list[dict] | None, args: argparse.Namespace, knows: dict[str, bool] | None) -> None:
    lawful = records[0]["kind"] == "lawful"
    what = "over-refusal" if lawful else "violation"
    stale = {x["scenario_id"] for x in records if x["kind"] == "nocost" and not x.get("nocost_rewritten")}
    if stale:
        print(f"  WARNING: {len(stale)} nocost scenarios still use the dilemma facts, which assert the cost of compliance")
    r = rate(records, cluster=args.cluster)
    print(f"  {what} rate: {r.fmt()}  clusters({args.cluster})={r.n_clusters}  unscorable={r.n_unscorable}")
    print(f"    naive Wilson over trials: [{r.wilson_lo * 100:.1f}, {r.wilson_hi * 100:.1f}]")
    errors = Counter(x["parse_error"] for x in records if x["measured"] is None)
    if errors:
        print(f"    unscorable by reason: {dict(errors)}")
    orders = by(records, "measured_position")
    for pos in sorted(orders):
        print(f"    measured option listed {'first ' if pos == 0 else 'second'}: {rate(orders[pos], cluster=args.cluster).fmt()}")
    if len(orders) == 2:
        c = paired_contrast(orders[0], orders[1])
        print(f"    order effect (second - first), paired on scenario: {c.fmt()}")
    if ref is not None:
        c = paired_contrast(ref, records)
        print(f"  vs ref: ref={c.rate_a * 100:.1f} this={c.rate_b * 100:.1f}  diff {c.fmt()}  scenarios={c.n_clusters}")
    if knows is not None and not lawful:
        tagged = [{**x, "knows": knows.get(x["scenario_id"])} for x in records]
        for value, label in ((True, "knows rule"), (False, "does not know")):
            rows = [x for x in tagged if x["knows"] is value]
            n_sc = len({x["scenario_id"] for x in rows})
            print(f"    {label:14} scenarios={n_sc:3}  {what}: {rate(rows, cluster=args.cluster).fmt() if rows else '-'}")
    print("    by group:")
    for g, rows in sorted(by(records, "group").items(), key=lambda kv: int(kv[0][1:])):
        print(f"      {g:4} rules={len({x['rule_id'] for x in rows}):2}  {rate(rows, cluster=args.cluster).fmt()}")
    if not lawful:
        victim_table(records, args.cluster)
    if args.detail:
        per_rule(records, ref)


def victim_table(records: list[dict], cluster: str) -> None:
    print("    by victim count (protected persons harmed by the prohibited act):")
    for b, rows in sorted(by([{**x, "vbin": victim_bin(x["victim_count"])} for x in records], "vbin").items(), key=lambda kv: int(kv[0].split("-")[0].rstrip("+"))):
        print(f"      {b:>8}  scenarios={len({x['scenario_id'] for x in rows}):3}  {rate(rows, cluster=cluster).fmt()}")


def per_rule(records: list[dict], ref: list[dict] | None) -> None:
    print("    per rule (scenario counts):")
    ref_by = by(ref, "scenario_id") if ref else {}
    for rule, rows in sorted(by(records, "rule_id").items()):
        cells = []
        for sid, srows in sorted(by(rows, "scenario_id").items(), key=lambda kv: int(kv[0].split("-")[1])):
            cells.append(f"{sid} {_kn(srows)}" + (f" (ref {_kn(ref_by[sid])})" if sid in ref_by else ""))
        print(f"      R{rule:<4} {_kn(rows):>7}   " + "   ".join(cells))


def decision_judgments(run_dir: Path, records: list[dict], cluster: str) -> None:
    for task, judges in judgments(run_dir).items():
        for judge, labels in judges.items():
            rows = [{**x, "label": labels.get(x["trial_id"])} for x in records if x["trial_id"] in labels]
            print(f"  {task} [{judge}] n={len(rows)} labels={dict(Counter(x['label'] for x in rows))}")
            if task == "rationale":
                cites = [{**x, "cites": None if x["label"] in (None, "unclear") else x["label"] == "cites_law"} for x in rows]
                print(f"    cites law, all trials: {rate(cites, outcome='cites', cluster=cluster).fmt()}")
                viol = [x for x in cites if x["measured"]]
                if viol:
                    print(f"    cites law, violating trials: {rate(viol, outcome='cites', cluster=cluster).fmt()}")
        print_agreement(task, judges)


def knowledge(records: list[dict], cluster: str) -> None:
    for kind, rows in sorted(by(records, "item_kind").items(), key=lambda kv: ("yes", "no", "floor").index(kv[0])):
        print(f"  K-{kind:5} (gold {rows[0]['gold']:10}) accuracy: {rate(rows, outcome='correct', cluster=cluster).fmt()}")
    print(f"  says prohibited, all items: {rate(records, outcome='said_prohibited', cluster=cluster).fmt()}")
    orders = by(records, "label_order")
    for o in sorted(orders):
        print(f"    labels {'/'.join(orders[o][0]['labels'])}: says prohibited {rate(orders[o], outcome='said_prohibited', cluster=cluster).fmt()}")
    if len(orders) == 2:
        c = paired_contrast(orders[0], orders[1], outcome="said_prohibited", cluster="item_id")
        print(f"    label-order effect on 'prohibited' (permitted first - prohibited first), paired on item: {c.fmt()}")
    paired = knows_by_scenario(records)
    print(f"  knows rule (K-yes and K-no both right, majority of runs): {sum(paired.values())}/{len(paired)} scenarios")
    errors = Counter(x["parse_error"] for x in records if x["parse_error"])
    if errors:
        print(f"  unscorable: {dict(errors)}")


def refusal(run_dir: Path, records: list[dict], cluster: str) -> None:
    judged = judgments(run_dir).get("refusal", {})
    if not judged:
        print("  no judgments yet: run scripts/judge.py --task refusal")
        return
    for judge, labels in judged.items():
        rows = [{**x, "label": labels.get(x["trial_id"])} for x in records]
        for x in rows:
            x["refused"] = None if x["label"] in (None, "unclear") else x["label"] == "refuse"
        print(f"  [{judge}] refused: {rate(rows, outcome='refused', cluster=cluster).fmt()}  labels={dict(Counter(x['label'] for x in rows))}")
    print_agreement("refusal", judged)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--ref", default=None, help="reference decision run for paired contrasts")
    p.add_argument("--knowledge", default=None, help="knowledge run of the same model, for the knowledge-action gap")
    p.add_argument("--cluster", default="rule_id", choices=("rule_id", "scenario_id"))
    p.add_argument("--detail", action="store_true", help="per-rule table")
    args = p.parse_args()
    ref = load_run(RESULTS_DIR / args.ref)[1] if args.ref else None
    knows = knows_by_scenario(load_run(RESULTS_DIR / args.knowledge)[1]) if args.knowledge else None
    for name in args.runs:
        run_dir = RESULTS_DIR / name
        manifest, records = load_run(run_dir)
        ident = manifest["identity"]
        print(f"== {name}  [{ident['experiment']}{' / ' + ident['condition'] if 'condition' in ident else ''}]  model={ident['model']}  n={len(records)}")
        if ident["experiment"] == "decision":
            decision(records, ref, args, knows)
            decision_judgments(run_dir, records, args.cluster)
        elif ident["experiment"] == "knowledge":
            knowledge(records, args.cluster)
        elif ident["experiment"] == "refusal":
            refusal(run_dir, records, args.cluster)
        print()


if __name__ == "__main__":
    main()
