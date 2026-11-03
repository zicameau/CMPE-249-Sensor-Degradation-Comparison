"""BEVFusion plugin: LiDAR corruptions inside the official test pipeline. Import it, nothing else.

    import sdc_lidar.bevfusion_plugin   # must run BEFORE the dataset is built

It re-registers ``LoadPointsFromMultiSweeps`` (``force=True``) with a subclass, so *no config file
changes are needed* - the stock YAML configs keep working. The corruption is chosen by environment
variables (or constructor kwargs):

    SDC_CORRUPTION = point_drop | beam_reduction | range_limit | missing_sweeps   (unset = clean)
    SDC_SEVERITY   = 0..5
    SDC_SEED       = int (default 0)

With no corruption set, ``__call__`` delegates to the original class, so the clean run is exactly
the published pipeline. This module needs the BEVFusion/mmdet3d environment (Docker image) and is
therefore NOT covered by the host unit tests; the numpy logic it calls is
(``sdc_lidar.multisweep`` / ``sdc_lidar.corruptions``).
"""
from __future__ import annotations

import os

import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines.loading import LoadPointsFromMultiSweeps as _Original

from sdc_lidar import corruptions as C
from sdc_lidar.multisweep import RawSweep, accumulate


@PIPELINES.register_module(name="LoadPointsFromMultiSweeps", force=True)
class CorruptedLoadPointsFromMultiSweeps(_Original):
    def __init__(self, *args, corruption=None, severity=None, seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.corruption = corruption or os.environ.get("SDC_CORRUPTION") or None
        self.severity = int(severity if severity is not None else os.environ.get("SDC_SEVERITY", 0))
        self.seed = int(seed if seed is not None else os.environ.get("SDC_SEED", 0))
        if self.corruption and self.corruption not in C.REGISTRY:
            raise KeyError(f"SDC_CORRUPTION={self.corruption!r}; choose from {sorted(C.REGISTRY)}")

    def __call__(self, results):
        if not self.corruption or self.severity == 0:
            return super().__call__(results)

        points = results["points"]                       # LiDARPoints from LoadPointsFromFile
        key_raw = points.tensor.numpy()                  # (N, 5): x y z intensity ring
        sweeps = []
        for sw in results["sweeps"][: self.sweeps_num]:
            pts = np.copy(self._load_points(sw["data_path"])).reshape(-1, self.load_dim)
            sweeps.append(RawSweep(pts, sw["sensor2lidar_rotation"], sw["sensor2lidar_translation"],
                                   sw["timestamp"] / 1e6))
        frame_key = results.get("sample_idx") or results.get("token") or ""
        merged = accumulate(
            key_raw, results["timestamp"] / 1e6, sweeps,
            sweeps_num=self.sweeps_num, use_dim=self.use_dim,
            remove_close_points=self.remove_close, pad_empty_sweeps=self.pad_empty_sweeps,
            corruption=self.corruption, severity=self.severity, seed=self.seed, frame_key=frame_key)
        results["points"] = points.new_point(merged)
        return results
