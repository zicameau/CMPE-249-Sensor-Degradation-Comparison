"""LiDAR corruptions: point drop, beam reduction, range limit, missing sweeps.

Pure numpy, no ROS / torch dependency, so the exact same code is used by
  * the offline sweep driver (BEVFusion evaluation under corruption),
  * the ROS 2 corruption node (live / replay),
  * the unit tests.

Conventions
-----------
* A *sweep* is an ``(N, C)`` float array whose first columns are
  ``x, y, z, intensity, ring`` (nuScenes LIDAR_TOP layout: 32 beams, ``ring`` in 0..31).
  Any extra columns (e.g. a time-lag column added after multi-sweep
  accumulation) are carried through untouched.
* Severity is an int: 0 = clean (identity), 1..5 = increasingly harsh.
* Stochastic corruptions (point drop, missing sweeps) are seeded. The RNG is
  derived from ``(seed, frame_key, sweep_idx)`` so the result for a given frame
  does not depend on processing order, and re-running with the same seed is
  bit-identical. The proposal uses seeds 0, 1, 2.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

X, Y, Z, INTENSITY, RING = 0, 1, 2, 3, 4
SEVERITIES = (1, 2, 3, 4, 5)
SEEDS = (0, 1, 2)

# nuScenes LIDAR_TOP is a Velodyne HDL-32E.
NUSCENES_NUM_BEAMS = 32

# ---------------------------------------------------------------------------
# Severity tables. These are *design choices* - edit here (and only here) and
# record the change in docs/lidar_corruptions.md so the report matches.
# ---------------------------------------------------------------------------
POINT_DROP_PROB = (0.10, 0.25, 0.40, 0.60, 0.80)           # fraction of points removed
BEAMS_KEPT = (24, 16, 12, 8, 4)                              # of 32 beams
RANGE_LIMIT_M = (50.0, 40.0, 30.0, 20.0, 10.0)              # max horizontal range (m)
SWEEPS_DROPPED_FRAC = (0.2, 0.4, 0.6, 0.8, 1.0)              # of the *past* sweeps


def frame_rng(seed: int, frame_key: str | int = "", sweep_idx: int = 0) -> np.random.Generator:
    """Deterministic RNG for one (seed, frame, sweep). Stable across processes/platforms."""
    if isinstance(frame_key, str):
        frame_key = zlib.crc32(frame_key.encode("utf-8"))
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(frame_key), int(sweep_idx)]))


def _check_severity(severity: int) -> int:
    if severity not in (0, *SEVERITIES):
        raise ValueError(f"severity must be 0..5, got {severity!r}")
    return int(severity)


def _check_points(points: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"expected (N, >=4) point array, got shape {points.shape}")


# ---------------------------------------------------------------------------
# Single-sweep corruptions
# ---------------------------------------------------------------------------
def point_drop(points: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Remove each point independently with probability ``POINT_DROP_PROB[severity-1]``."""
    _check_points(points)
    severity = _check_severity(severity)
    if severity == 0:
        return points
    keep = rng.random(len(points)) >= POINT_DROP_PROB[severity - 1]
    return points[keep]


def estimate_rings(points: np.ndarray, num_beams: int = NUSCENES_NUM_BEAMS,
                   fov_deg: tuple[float, float] = (-30.67, 10.67)) -> np.ndarray:
    """Fallback ring index from elevation angle, for clouds without a ring column.

    Uniform binning over the HDL-32E vertical FOV. Only an approximation (the real
    HDL-32E spacing is non-uniform); prefer the true ring column whenever it exists.
    """
    rng_xy = np.hypot(points[:, X], points[:, Y])
    elev = np.degrees(np.arctan2(points[:, Z], np.maximum(rng_xy, 1e-6)))
    lo, hi = fov_deg
    frac = np.clip((elev - lo) / (hi - lo), 0.0, 1.0 - 1e-9)
    return (frac * num_beams).astype(np.int64)


def beam_reduction(points: np.ndarray, severity: int, rng: np.random.Generator | None = None,
                   num_beams: int = NUSCENES_NUM_BEAMS, has_ring: bool = True) -> np.ndarray:
    """Emulate a lower-channel sensor by keeping ``BEAMS_KEPT[severity-1]`` evenly spaced rings.

    Deterministic (``rng`` accepted only for a uniform signature).
    """
    _check_points(points)
    severity = _check_severity(severity)
    if severity == 0:
        return points
    n_keep = BEAMS_KEPT[severity - 1]
    rings = points[:, RING].astype(np.int64) if has_ring and points.shape[1] > RING \
        else estimate_rings(points, num_beams)
    kept_rings = np.unique(np.round(np.linspace(0, num_beams - 1, n_keep)).astype(np.int64))
    return points[np.isin(rings, kept_rings)]


def range_limit(points: np.ndarray, severity: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Drop everything beyond ``RANGE_LIMIT_M[severity-1]`` metres (horizontal distance)."""
    _check_points(points)
    severity = _check_severity(severity)
    if severity == 0:
        return points
    r = np.hypot(points[:, X], points[:, Y])
    return points[r <= RANGE_LIMIT_M[severity - 1]]


# ---------------------------------------------------------------------------
# Multi-sweep corruption
# ---------------------------------------------------------------------------
def missing_sweep_indices(n_sweeps: int, severity: int, rng: np.random.Generator) -> list[int]:
    """Indices (into a current-frame-first list) that survive. Index 0 is always kept."""
    severity = _check_severity(severity)
    if severity == 0 or n_sweeps <= 1:
        return list(range(n_sweeps))
    n_past = n_sweeps - 1
    n_drop = int(round(SWEEPS_DROPPED_FRAC[severity - 1] * n_past))
    drop = set((rng.choice(n_past, size=n_drop, replace=False) + 1).tolist()) if n_drop else set()
    return [i for i in range(n_sweeps) if i not in drop]


def missing_sweeps(sweeps: Sequence[np.ndarray], severity: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Drop a random subset of the *past* sweeps. ``sweeps[0]`` (current frame) is always kept.

    BEVFusion/nuScenes accumulates 10 sweeps (current + 9 past). Severity 5 removes every
    past sweep (single-sweep input); severity 1 removes 20% (2 of 9).
    """
    sweeps = list(sweeps)
    return [sweeps[i] for i in missing_sweep_indices(len(sweeps), severity, rng)]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Corruption:
    name: str
    fn: Callable
    stochastic: bool
    multi_sweep: bool
    param_name: str
    levels: tuple

    def describe(self, severity: int) -> str:
        return "clean" if severity == 0 else f"{self.param_name}={self.levels[severity - 1]}"


REGISTRY: dict[str, Corruption] = {
    "point_drop": Corruption("point_drop", point_drop, True, False, "drop_prob", POINT_DROP_PROB),
    "beam_reduction": Corruption("beam_reduction", beam_reduction, False, False, "beams_kept", BEAMS_KEPT),
    "range_limit": Corruption("range_limit", range_limit, False, False, "max_range_m", RANGE_LIMIT_M),
    "missing_sweeps": Corruption("missing_sweeps", missing_sweeps, True, True,
                                 "past_sweeps_dropped_frac", SWEEPS_DROPPED_FRAC),
}


def seeds_for(name: str) -> tuple[int, ...]:
    """3 seeds for stochastic corruptions, a single run for deterministic ones."""
    return SEEDS if REGISTRY[name].stochastic else (0,)


def apply_with_indices(name: str, sweeps: Sequence[np.ndarray], severity: int, seed: int = 0,
                       frame_key: str | int = "") -> tuple[list[np.ndarray], list[int]]:
    """Like :func:`apply_to_sweeps` but also returns which input sweeps survived (their indices).

    Needed when per-sweep metadata (pose, timestamp) must stay aligned with the points.
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown corruption {name!r}; choose from {sorted(REGISTRY)}")
    c = REGISTRY[name]
    severity = _check_severity(severity)
    if c.multi_sweep:
        idx = missing_sweep_indices(len(sweeps), severity, frame_rng(seed, frame_key, 0))
        return [sweeps[i] for i in idx], idx
    return ([c.fn(s, severity, frame_rng(seed, frame_key, i)) for i, s in enumerate(sweeps)],
            list(range(len(sweeps))))


def apply_to_sweeps(name: str, sweeps: Sequence[np.ndarray], severity: int, seed: int = 0,
                    frame_key: str | int = "") -> list[np.ndarray]:
    """Apply corruption ``name`` to a frame's sweeps (current frame first). Always returns a list."""
    return apply_with_indices(name, sweeps, severity, seed, frame_key)[0]


def apply(name: str, points: np.ndarray, severity: int, seed: int = 0, frame_key: str | int = "") -> np.ndarray:
    """Convenience wrapper for a single-sweep cloud (not valid for ``missing_sweeps``)."""
    if REGISTRY[name].multi_sweep:
        raise ValueError(f"{name} needs a list of sweeps; use apply_to_sweeps()")
    return apply_to_sweeps(name, [points], severity, seed, frame_key)[0]
