"""Multi-sweep accumulation with corruptions injected on the *raw* sweeps (numpy only).

Mirrors ``LoadPointsFromMultiSweeps`` from mit-han-lab/bevfusion (test mode) step for step,
with one difference: corruptions are applied to each raw sweep *before* ego-motion alignment,
because that is where the physical sensor degradation happens - and because the raw files still
carry the ``ring`` column that BEVFusion later overwrites with the time lag.

BEVFusion steps reproduced (see mmdet3d/datasets/pipelines/loading.py):
  1. keyframe points: use_dim columns, column 4 (ring) overwritten with time lag 0
  2. if the sample has no sweeps and ``pad_empty_sweeps``: repeat the keyframe ``sweeps_num`` times
  3. else take the first ``sweeps_num`` sweeps (test mode = nearest in time); for each:
     remove points within 1 m (|x|<1 and |y|<1), select use_dim, rotate+translate into the
     keyframe LiDAR frame, column 4 = keyframe_time - sweep_time
  4. concatenate
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import corruptions as C


@dataclass
class RawSweep:
    points: np.ndarray          # (N, 5) x, y, z, intensity, ring in the sweep's own sensor frame
    rotation: np.ndarray        # (3, 3) sensor2lidar_rotation
    translation: np.ndarray     # (3,)   sensor2lidar_translation
    timestamp_s: float          # seconds


def remove_close(points: np.ndarray, radius: float = 1.0) -> np.ndarray:
    close = (np.abs(points[:, 0]) < radius) & (np.abs(points[:, 1]) < radius)
    return points[~close]


def accumulate(key_points: np.ndarray, key_timestamp_s: float, sweeps: Sequence[RawSweep], *,
               sweeps_num: int = 9, use_dim: Sequence[int] = (0, 1, 2, 3, 4),
               remove_close_points: bool = True, pad_empty_sweeps: bool = True,
               corruption: str | None = None, severity: int = 0, seed: int = 0,
               frame_key: str | int = "") -> np.ndarray:
    """Build the (M, len(use_dim)) accumulated cloud, optionally corrupted.

    ``key_points`` must be the raw (N, 5) keyframe including ring (column 4).
    """
    use_dim = list(use_dim)
    if key_points.ndim != 2 or key_points.shape[1] < 5:
        raise ValueError("key_points must be raw (N, 5) with the ring column intact")
    corrupt = bool(corruption) and severity != 0

    chosen = list(sweeps[:sweeps_num])
    raw = [key_points] + [s.points for s in chosen]
    kept_idx = list(range(len(raw)))
    if corrupt:
        raw, kept_idx = C.apply_with_indices(corruption, raw, severity, seed, frame_key)

    key = np.array(raw[0], dtype=np.float32, copy=True)[:, use_dim]
    key[:, 4] = 0.0  # use_dim must keep column 4 (time-lag slot), as in BEVFusion configs
    out = [key]

    if pad_empty_sweeps and len(sweeps) == 0:
        # Keyframe repeated; a missing-sweeps corruption has nothing to remove here.
        padded = remove_close(key) if remove_close_points else key
        out.extend(padded for _ in range(sweeps_num))
    else:
        for pts, i in zip(raw[1:], kept_idx[1:]):
            sw = chosen[i - 1]
            p = np.array(pts, dtype=np.float32, copy=True)
            if remove_close_points:
                p = remove_close(p)
            p = p[:, use_dim]
            p[:, :3] = p[:, :3] @ np.asarray(sw.rotation, dtype=np.float32).T
            p[:, :3] += np.asarray(sw.translation, dtype=np.float32)
            p[:, 4] = key_timestamp_s - sw.timestamp_s
            out.append(p)
    return np.concatenate(out, axis=0)
