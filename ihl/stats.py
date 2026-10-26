"""Rates with uncertainty that respects the design.

Trials of one scenario are repeated samples of the same prompt, not
independent observations. With 18 scenarios x 10 samples the effective sample
size is close to the number of scenarios, so every interval here resamples
scenarios (cluster bootstrap), and every condition contrast is paired on
scenario.

The Wilson interval over trials is reported next to it only as the naive
reference; it is too narrow whenever scenarios differ.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

N_BOOT = 10_000


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _clusters(rows: Iterable[dict[str, Any]], outcome: str, cluster: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        v = r.get(outcome)
        if v is None:
            continue
        out[r[cluster]].append(int(bool(v)))
    return dict(out)


@dataclass
class Rate:
    rate: float
    k: int
    n: int
    n_clusters: int
    n_unscorable: int
    ci_lo: float
    ci_hi: float
    wilson_lo: float
    wilson_hi: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fmt(self, pct: bool = True) -> str:
        s = 100 if pct else 1
        return f"{self.rate * s:.1f} [{self.ci_lo * s:.1f}, {self.ci_hi * s:.1f}] ({self.k}/{self.n})"


def rate(
    rows: list[dict[str, Any]],
    outcome: str = "measured",
    cluster: str = "scenario_id",
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> Rate:
    """Pooled rate over trials; 95% CI from resampling clusters with replacement."""
    groups = _clusters(rows, outcome, cluster)
    n_unscorable = sum(1 for r in rows if r.get(outcome) is None)
    ks = np.array([sum(v) for v in groups.values()], dtype=float)
    ns = np.array([len(v) for v in groups.values()], dtype=float)
    k, n = int(ks.sum()), int(ns.sum())
    if n == 0:
        return Rate(math.nan, 0, 0, 0, n_unscorable, math.nan, math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ks), size=(n_boot, len(ks)))
    boot = ks[idx].sum(axis=1) / ns[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    wl, wh = wilson(k, n)
    return Rate(k / n, k, n, len(ks), n_unscorable, float(lo), float(hi), wl, wh)


@dataclass
class Contrast:
    diff: float  # rate(b) - rate(a), over shared clusters
    ci_lo: float
    ci_hi: float
    p_perm: float  # two-sided, sign-flip of per-cluster differences
    n_clusters: int
    rate_a: float
    rate_b: float

    def fmt(self, pct: bool = True) -> str:
        s = 100 if pct else 1
        return f"{self.diff * s:+.1f}%p [{self.ci_lo * s:+.1f}, {self.ci_hi * s:+.1f}] p={self.p_perm:.3g}"


def paired_contrast(
    a: list[dict[str, Any]],
    b: list[dict[str, Any]],
    outcome: str = "measured",
    cluster: str = "scenario_id",
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> Contrast:
    """Condition b minus condition a, paired on cluster (scenario).

    Each cluster contributes its own rate difference; clusters are weighted
    equally. CI: bootstrap over clusters. p: exact or Monte Carlo sign-flip
    permutation of the per-cluster differences (H0: no condition effect).
    """
    ga, gb = _clusters(a, outcome, cluster), _clusters(b, outcome, cluster)
    shared = sorted(set(ga) & set(gb))
    if not shared:
        return Contrast(math.nan, math.nan, math.nan, math.nan, 0, math.nan, math.nan)
    ra = np.array([np.mean(ga[c]) for c in shared])
    rb = np.array([np.mean(gb[c]) for c in shared])
    d = rb - ra
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    observed = abs(d.mean())
    if len(d) <= 16:
        signs = np.array(np.meshgrid(*[[-1, 1]] * len(d))).reshape(len(d), -1).T
    else:
        signs = rng.choice([-1, 1], size=(n_boot, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    p = float((np.sum(null >= observed - 1e-12) + (0 if len(d) <= 16 else 1)) / (len(null) + (0 if len(d) <= 16 else 1)))
    return Contrast(float(d.mean()), float(lo), float(hi), p, len(shared), float(ra.mean()), float(rb.mean()))


def by(rows: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    out: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[r.get(key)].append(r)
    return dict(out)
