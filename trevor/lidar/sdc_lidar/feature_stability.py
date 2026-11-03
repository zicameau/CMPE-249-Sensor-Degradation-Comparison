"""Feature stability (Tier 3): mean cosine similarity of clean vs corrupted features inside GT boxes.

Model-agnostic: works on any per-point features (Sonata, or a point-wise readout of a backbone).

Definition (write this in the report): for every GT box that contains at least one clean point, take
each clean in-box point and compare its feature with the feature of the *nearest corrupted point that
is still inside the same box* (a surviving point is its own nearest neighbour, distance 0). The box
score is the mean cosine over its clean points. A box with no corrupted points left inside is
*uncovered*: it has no defined similarity and is reported through ``coverage`` instead of being
scored 0 or silently dropped, so range-limit / heavy point-drop cannot hide behind a high mean.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


def points_in_box(xyz: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Boolean mask. ``box`` = [cx, cy, cz, length, width, height, yaw]; length runs along heading."""
    cx, cy, cz, l, w, h, yaw = [float(v) for v in box]
    d = xyz[:, :3] - np.array([cx, cy, cz])
    c, s = np.cos(-yaw), np.sin(-yaw)
    x = d[:, 0] * c - d[:, 1] * s
    y = d[:, 0] * s + d[:, 1] * c
    return (np.abs(x) <= l / 2) & (np.abs(y) <= w / 2) & (np.abs(d[:, 2]) <= h / 2)


def _normalize(f: np.ndarray) -> np.ndarray:
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)


@dataclass
class FeatureStability:
    per_box: np.ndarray        # (n_boxes,) mean cosine per box; NaN = no clean points or uncovered
    n_clean_points: np.ndarray  # (n_boxes,)
    covered: np.ndarray        # (n_boxes,) bool, True if >=1 corrupted point inside (and >=1 clean)

    @property
    def n_boxes(self) -> int:
        """Boxes that contain at least one clean point (the denominator for coverage)."""
        return int((self.n_clean_points > 0).sum())

    @property
    def coverage(self) -> float:
        return float(self.covered.sum() / self.n_boxes) if self.n_boxes else float("nan")

    @property
    def mean_cosine(self) -> float:
        v = self.per_box[self.covered]
        return float(v.mean()) if v.size else float("nan")


def feature_stability(clean_xyz, clean_feat, corr_xyz, corr_feat, boxes) -> FeatureStability:
    clean_xyz, corr_xyz = np.asarray(clean_xyz), np.asarray(corr_xyz)
    clean_feat, corr_feat = _normalize(np.asarray(clean_feat, dtype=np.float64)), \
        _normalize(np.asarray(corr_feat, dtype=np.float64))
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    n = len(boxes)
    per_box = np.full(n, np.nan)
    n_clean = np.zeros(n, dtype=int)
    covered = np.zeros(n, dtype=bool)
    for b, box in enumerate(boxes):
        ci = np.flatnonzero(points_in_box(clean_xyz, box))
        n_clean[b] = len(ci)
        if len(ci) == 0:
            continue
        ki = np.flatnonzero(points_in_box(corr_xyz, box)) if len(corr_xyz) else np.array([], dtype=int)
        if len(ki) == 0:
            continue
        _, nn = cKDTree(corr_xyz[ki, :3]).query(clean_xyz[ci, :3], k=1)
        sims = np.einsum("ij,ij->i", clean_feat[ci], corr_feat[ki[nn]])
        per_box[b] = float(sims.mean())
        covered[b] = True
    return FeatureStability(per_box, n_clean, covered)
