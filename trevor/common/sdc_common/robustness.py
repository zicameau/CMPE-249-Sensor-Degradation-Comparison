"""Robustness aggregation on bootstrap draws: RR, mRR, failure threshold (all with scene-bootstrap CIs).

Every number here is computed *per resample* and then summarised, so intervals respect the pairing between
the clean run and each corrupted run (all runs must be evaluated with the same bootstrap weights).

Aggregation order (state it in the report):
  RR(c, s, seed)  = corrupted / clean
  RR(c, s)        = mean over seeds            (1 seed for deterministic corruptions)
  mRR             = mean over corruptions of (mean over severities of RR(c, s))     [equal weight per corruption]
  failure(c)      = lowest severity whose RR(c, s) < 0.8; ``NEVER`` (= 6) if none does
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np

from .nusc_eval import Draws
from .stats import Estimate, RR_FAIL_THRESHOLD

NEVER = 6   # "does not fail within severities 1..5"

RunKey = Tuple[str, int, int]   # (corruption, severity, seed)


def _est(point: float, boot: np.ndarray) -> Estimate:
    b = boot[np.isfinite(boot)]
    lo, hi = (np.percentile(b, 2.5), np.percentile(b, 97.5)) if b.size else (float("nan"),) * 2
    return Estimate(float(point), float(lo), float(hi))


def rr_arrays(clean: Draws, corr: Draws, key: str) -> Tuple[float, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = corr.boot[key] / clean.boot[key]
    c = clean.point[key]
    return (corr.point[key] / c if c else float("nan")), boot


@dataclass
class RobustnessSummary:
    rr: Dict[RunKey, Estimate]                   # per run (incl. each seed)
    rr_cs: Dict[Tuple[str, int], Estimate]       # seed-mean RR per (corruption, severity)
    mrr: Estimate
    failure: Dict[str, Estimate]                 # per corruption; value/CI in severity units (NEVER = 6)


def summarize(clean: Draws, runs: Mapping[RunKey, Draws], key: str,
              threshold: float = RR_FAIL_THRESHOLD) -> RobustnessSummary:
    per_run = {k: rr_arrays(clean, d, key) for k, d in runs.items() if k[1] > 0}
    rr = {k: _est(p, b) for k, (p, b) in per_run.items()}

    # seed-mean per (corruption, severity)
    groups: Dict[Tuple[str, int], list] = {}
    for (c, s, _seed), pb in per_run.items():
        groups.setdefault((c, s), []).append(pb)
    cs = {k: (float(np.nanmean([p for p, _ in v])), np.mean([b for _, b in v], axis=0)) for k, v in groups.items()}
    rr_cs = {k: _est(p, b) for k, (p, b) in cs.items()}

    corruptions = sorted({c for c, _ in cs})
    # mRR: mean over corruptions of mean over severities
    pt_c, bt_c = [], []
    for c in corruptions:
        sevs = sorted(s for cc, s in cs if cc == c)
        pt_c.append(np.nanmean([cs[(c, s)][0] for s in sevs]))
        bt_c.append(np.mean([cs[(c, s)][1] for s in sevs], axis=0))
    mrr = _est(float(np.nanmean(pt_c)), np.mean(bt_c, axis=0)) if corruptions else _est(float("nan"), np.array([]))

    failure = {}
    for c in corruptions:
        sevs = sorted(s for cc, s in cs if cc == c)
        pt = next((s for s in sevs if cs[(c, s)][0] < threshold), NEVER)
        stack = np.stack([cs[(c, s)][1] for s in sevs])                 # (n_sev, n_boot)
        below = stack < threshold                                       # NaN compares False
        first = np.where(below.any(axis=0), below.argmax(axis=0), -1)
        boot = np.array([sevs[i] if i >= 0 else NEVER for i in first], dtype=float)
        failure[c] = _est(float(pt), boot)
    return RobustnessSummary(rr, rr_cs, mrr, failure)
