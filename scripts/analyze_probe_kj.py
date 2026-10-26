#!/usr/bin/env python3
"""Aggregate K/J runs and print the knowledge ladder + rule-level table.

    python scripts/analyze_probe_kj.py
    python scripts/analyze_probe_kj.py --write-audit-sample
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ihl.runio import RESULTS_DIR, read_jsonl  # noqa: E402
from ihl.stats import rate  # noqa: E402

MODELS = (
    ("qwen", "qwen-base"),
    ("gemma", "gemma"),
    ("ministral", "ministral"),
)
AMBIGUOUS = {25, 27, 28, 29, 31, 34, 47, 60, 75}


def load_records(run: str) -> list[dict]:
    path = RESULTS_DIR / run / "records.jsonl"
    return list(read_jsonl(path)) if path.exists() else []


def mean(xs: list[float | int | bool | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def k_summary(rows: list[dict]) -> dict:
    scored = [r for r in rows if r.get("scorable")]
    depends = sum(1 for r in scored if (r.get("record") or {}).get("status") == "depends_on_circumstances")
    return {
        "n": len(rows),
        "scored": len(scored),
        "unscorable": len(rows) - len(scored),
        "knows": mean([r.get("knows") for r in scored]),
        "absolute": mean([r.get("absolute") for r in scored]),
        "depends_share": depends / len(scored) if scored else None,
    }


def j_summary(rows: list[dict]) -> dict:
    scored = [r for r in rows if r.get("scorable")]
    return {
        "n": len(rows),
        "scored": len(scored),
        "unscorable": len(rows) - len(scored),
        "judged_violation_right": mean([r.get("judged_violation_right") for r in scored]),
    }


def d_violation(run: str) -> float | None:
    rows = load_records(run)
    if not rows:
        return None
    # dilemma measured rate
    r = rate([x for x in rows if x.get("kind") == "dilemma" or "kind" not in x], cluster="rule_id")
    return r.rate


def rule_means(k_rows: list[dict], j_rows: list[dict], d_rows: list[dict], nocost_rows: list[dict]) -> list[dict]:
    by_k: dict[int, list] = defaultdict(list)
    by_j: dict[int, list] = defaultdict(list)
    by_d: dict[int, list] = defaultdict(list)
    by_n: dict[int, list] = defaultdict(list)
    groups = {}
    for r in k_rows:
        if r.get("scorable"):
            by_k[r["rule_id"]].append(r)
            groups[r["rule_id"]] = r.get("group")
    for r in j_rows:
        if r.get("scorable"):
            by_j[r["rule_id"]].append(r)
            groups[r["rule_id"]] = r.get("group")
    for r in d_rows:
        if r.get("measured") is not None:
            by_d[r["rule_id"]].append(r)
            groups[r["rule_id"]] = r.get("group")
    for r in nocost_rows:
        if r.get("measured") is not None:
            by_n[r["rule_id"]].append(r)
            groups[r["rule_id"]] = r.get("group")
    out = []
    for rid in sorted(set(by_k) | set(by_j) | set(by_d) | set(by_n)):
        out.append(
            {
                "rule_id": rid,
                "group": groups.get(rid),
                "K_knows": mean([r["knows"] for r in by_k.get(rid, [])]),
                "K_absolute": mean([r["absolute"] for r in by_k.get(rid, [])]),
                "J_viol": mean([r["judged_violation_right"] for r in by_j.get(rid, [])]),
                "D_viol": mean([r["measured"] for r in by_d.get(rid, [])]),
                "nocost_viol": mean([r["measured"] for r in by_n.get(rid, [])]),
            }
        )
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    # average ranks for ties
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = sum((a - mx) ** 2 for a in rx) ** 0.5
    deny = sum((b - my) ** 2 for b in ry) ** 0.5
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def write_audit_sample(path: Path, seed: int = 0) -> None:
    """§5.2 stratified sample of ~100 K trials for human coding."""
    all_rows = []
    for tag, run_prefix in MODELS:
        for r in load_records(f"{run_prefix}/probe_k"):
            if r.get("turn1_text"):
                all_rows.append(r)
    if not all_rows:
        print("no K records yet")
        return
    rng = random.Random(seed)
    by_model: dict[str, list] = defaultdict(list)
    for r in all_rows:
        by_model[r["model"]].append(r)
    sample = []
    per = max(1, 100 // max(1, len(by_model)))
    for model, rows in by_model.items():
        by_status: dict[str, list] = defaultdict(list)
        for r in rows:
            st = (r.get("record") or {}).get("status") or "missing"
            by_status[st].append(r)
        picked = []
        for st in ("prohibited", "permitted", "depends_on_circumstances"):
            pool = by_status.get(st, [])
            rng.shuffle(pool)
            picked.extend(pool[: min(5, len(pool))])
        # prefer ambiguous rules
        amb = [r for r in rows if r["rule_id"] in AMBIGUOUS and r not in picked]
        rng.shuffle(amb)
        for r in amb:
            if len(picked) >= per:
                break
            picked.append(r)
        rest = [r for r in rows if r not in picked]
        rng.shuffle(rest)
        while len(picked) < per and rest:
            picked.append(rest.pop())
        sample.extend(picked[:per])
    # trim/pad to ~100
    rng.shuffle(sample)
    sample = sample[:100]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "trial_id",
                "model",
                "rule_id",
                "group",
                "topic",
                "turn1_text",
                "record_status",
                "record_admits_exception",
                "record_instrument",
                "c_status",
                "c_exception",
                "c_exception_type",
            ],
        )
        w.writeheader()
        for r in sample:
            rec = r.get("record") or {}
            w.writerow(
                {
                    "trial_id": r["trial_id"],
                    "model": r["model"],
                    "rule_id": r["rule_id"],
                    "group": r.get("group"),
                    "topic": r.get("topic"),
                    "turn1_text": r.get("turn1_text"),
                    "record_status": rec.get("status"),
                    "record_admits_exception": rec.get("admits_exception"),
                    "record_instrument": rec.get("instrument"),
                    "c_status": "",
                    "c_exception": "",
                    "c_exception_type": "",
                }
            )
    print(f"wrote {path} n={len(sample)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--write-audit-sample", action="store_true")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args()

    if args.write_audit_sample:
        write_audit_sample(args.out_dir / "k_audit.csv")
        return

    # aggregate jsonl
    k_all, j_all = [], []
    print("== knowledge ladder ==")
    print(f"{'model':10} {'K_knows':>8} {'K_abs':>8} {'J_viol':>8} {'D_viol':>8} {'nocost':>8}  unscorable_K/J")
    rule_rows_all = []
    for tag, prefix in MODELS:
        k = load_records(f"{prefix}/probe_k")
        j = load_records(f"{prefix}/probe_j")
        d = load_records(f"{prefix}/baseline")
        n = load_records(f"{prefix}/nocost")
        for r in k:
            k_all.append(r)
        for r in j:
            j_all.append(r)
        ks, js = k_summary(k), j_summary(j)
        d_rate = mean([r["measured"] for r in d if r.get("measured") is not None])
        n_rate = mean([r["measured"] for r in n if r.get("measured") is not None])
        def fmt(x):
            return f"{100*x:6.1f}%" if x is not None else "     —"
        print(
            f"{tag:10} {fmt(ks['knows'])} {fmt(ks['absolute'])} {fmt(js['judged_violation_right'])} "
            f"{fmt(d_rate)} {fmt(n_rate)}  K {ks['unscorable']}/{ks['n']}  J {js['unscorable']}/{js['n']}"
        )
        if ks["depends_share"] is not None:
            print(f"           depends_on_circumstances share among scored K: {100*ks['depends_share']:.1f}%")
        rules = rule_means(k, j, d, n)
        for row in rules:
            row["model"] = tag
            rule_rows_all.append(row)
        # spearman
        pairs = [
            (row["K_absolute"], row["nocost_viol"])
            for row in rules
            if row["K_absolute"] is not None and row["nocost_viol"] is not None
        ]
        if pairs:
            rho = spearman([a for a, _ in pairs], [b for _, b in pairs])
            print(f"           Spearman K_absolute vs nocost_viol (n={len(pairs)}): {rho}")

    # write aggregates
    k_path = args.out_dir / "probe_k.jsonl"
    j_path = args.out_dir / "probe_j.jsonl"
    with k_path.open("w", encoding="utf-8") as f:
        for r in k_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with j_path.open("w", encoding="utf-8") as f:
        for r in j_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rule_path = args.out_dir / "rule_level.csv"
    if rule_rows_all:
        with rule_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rule_rows_all[0].keys()))
            w.writeheader()
            w.writerows(rule_rows_all)
    print(f"wrote {k_path} ({len(k_all)}) {j_path} ({len(j_all)}) {rule_path} ({len(rule_rows_all)})")
    print("NOTE: nocost figures are from current results/; re-run nocost on rewritten data before trusting K×nocost.")


if __name__ == "__main__":
    main()
