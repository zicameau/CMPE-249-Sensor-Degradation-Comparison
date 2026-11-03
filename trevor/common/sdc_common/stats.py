"""Statistics from the proposal: scene bootstrap, paired bootstrap, RR/mRR, failure threshold, latency.

Design note - weights instead of re-indexing
--------------------------------------------
A bootstrap resample of N scenes is represented as an integer weight vector ``w`` of
length N (``w[i]`` = number of times scene i was drawn, ``w.sum() == N``). A metric is any
callable ``metric(w) -> float`` that treats ``w`` as per-scene multiplicities. This is
exactly equivalent to resampling scenes with replacement, but lets a metric precompute
per-scene sufficient statistics once and evaluate 1,000 resamples cheaply. We resample
*scenes*, never frames, because frames inside a scene are correlated.

``metric(np.ones(N))`` is the point estimate on the full fixed frame set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

N_BOOT = 1000
RR_FAIL_THRESHOLD = 0.8
LATENCY_WARMUP_FRAMES = 50

Metric = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class Estimate:
    value: float
    ci_low: float
    ci_high: float

    def excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def bootstrap_weights(n_scenes: int, n_boot: int = N_BOOT, seed: int = 12345) -> np.ndarray:
    """(n_boot, n_scenes) multiplicity matrix. Use one matrix for everything you want paired."""
    if n_scenes < 2:
        raise ValueError("need at least 2 scenes to bootstrap")
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_scenes, np.full(n_scenes, 1.0 / n_scenes), size=n_boot).astype(np.float64)


def _ci(samples: np.ndarray) -> tuple[float, float]:
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def bootstrap_ci(metric: Metric, n_scenes: int, n_boot: int = N_BOOT, seed: int = 12345,
                 weights: np.ndarray | None = None) -> Estimate:
    """Point estimate on all scenes + 95% percentile interval over scene resamples."""
    W = bootstrap_weights(n_scenes, n_boot, seed) if weights is None else weights
    point = float(metric(np.ones(n_scenes)))
    lo, hi = _ci(np.array([metric(w) for w in W]))
    return Estimate(point, lo, hi)


def paired_bootstrap_diff(metric_a: Metric, metric_b: Metric, n_scenes: int, n_boot: int = N_BOOT,
                          seed: int = 12345, weights: np.ndarray | None = None) -> Estimate:
    """CI on ``metric_a - metric_b`` using the *same* resampled scenes for both models.

    A difference counts only if ``Estimate.excludes_zero()``. Report intervals, not p-values.
    """
    W = bootstrap_weights(n_scenes, n_boot, seed) if weights is None else weights
    ones = np.ones(n_scenes)
    point = float(metric_a(ones) - metric_b(ones))
    lo, hi = _ci(np.array([metric_a(w) - metric_b(w) for w in W]))
    return Estimate(point, lo, hi)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
def resilience_rate(corrupted: float, clean: float) -> float:
    """RR = corrupted score / clean score."""
    return float("nan") if clean == 0 else corrupted / clean


def rr_bootstrap(clean_metric: Metric, corrupted_metric: Metric, n_scenes: int, n_boot: int = N_BOOT,
                 seed: int = 12345, weights: np.ndarray | None = None) -> Estimate:
    """RR with a CI where clean and corrupted are recomputed on the same resampled scenes."""
    W = bootstrap_weights(n_scenes, n_boot, seed) if weights is None else weights
    ones = np.ones(n_scenes)
    point = resilience_rate(corrupted_metric(ones), clean_metric(ones))
    draws = np.array([resilience_rate(corrupted_metric(w), clean_metric(w)) for w in W])
    lo, hi = _ci(draws)
    return Estimate(point, lo, hi)


def mrr(rr_values: Sequence[float]) -> float:
    """Mean RR across severities and corruptions in a sensor group (NaNs ignored)."""
    a = np.asarray(rr_values, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def failure_threshold(rr_by_severity: Mapping[int, float], threshold: float = RR_FAIL_THRESHOLD) -> int | None:
    """Lowest severity whose RR falls below ``threshold``; None if it never does."""
    for sev in sorted(rr_by_severity):
        if sev > 0 and rr_by_severity[sev] < threshold:
            return sev
    return None


def seed_summary(values_by_seed: Mapping[int, float]) -> tuple[float, float, float]:
    """(mean, min, max) across seeds for stochastic corruptions."""
    v = np.array(list(values_by_seed.values()), dtype=float)
    return float(v.mean()), float(v.min()), float(v.max())


# ---------------------------------------------------------------------------
# System / latency
# ---------------------------------------------------------------------------
def latency_summary(runs: Sequence[Sequence[float]], warmup: int = LATENCY_WARMUP_FRAMES) -> dict:
    """Latency percentiles from pooled post-warm-up frames across repeated runs.

    Returns p50/p95/p99 from the pooled frames, plus for each metric (ci_low, ci_high) = the
    min/max of the per-run value across the repeated runs (the proposal's non-bootstrap rule).
    """
    trimmed = [np.asarray(r, dtype=float)[warmup:] for r in runs]
    if any(t.size == 0 for t in trimmed):
        raise ValueError(f"a run has <= {warmup} frames; nothing left after warm-up")
    pooled = np.concatenate(trimmed)
    out = {}
    for name, q in (("p50", 50), ("p95", 95), ("p99", 99)):
        per_run = [float(np.percentile(t, q)) for t in trimmed]
        out[name] = Estimate(float(np.percentile(pooled, q)), min(per_run), max(per_run))
    return out
