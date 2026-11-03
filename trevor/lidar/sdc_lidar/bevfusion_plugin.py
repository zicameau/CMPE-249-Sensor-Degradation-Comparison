"""BEVFusion plugin: LiDAR corruptions inside the official test pipeline. Import it, nothing else.

    import sdc_lidar.bevfusion_plugin   # must run BEFORE the dataset is built

It re-registers ``LoadPointsFromMultiSweeps`` (``force=True``) with a subclass, so *no config file
changes are needed* - the stock YAML configs keep working. The corruption is chosen by environment
variables (or constructor kwargs):

    SDC_CORRUPTION = point_drop | beam_reduction | range_limit | missing_sweeps   (unset = clean)
    SDC_SEVERITY   = 0..5
    SDC_SEED       = int (default 0)
    SDC_DETERMINISTIC_SWEEPS = 1   use the nearest-N sweeps for EVERY run, clean included (see below)

With no corruption set and SDC_DETERMINISTIC_SWEEPS unset, ``__call__`` delegates to the original class, so the clean
run is exactly the published pipeline.

Why SDC_DETERMINISTIC_SWEEPS exists: the stock test pipeline does not pass ``test_mode=True`` to
``LoadPointsFromMultiSweeps``. nuScenes infos hold 10 sweeps and the configs use ``sweeps_num: 9``, so the ORIGINAL
loader draws 9 of the 10 sweeps at random (unseeded) at test time. Our corrupted runs take the 9 nearest sweeps, so a
clean baseline from the original loader would differ from them by more than the corruption. The sweep therefore sets
this flag for all runs, making clean and corrupted runs paired and reproducible; the reproduction gate keeps the
original loader. This module needs the BEVFusion/mmdet3d environment (Docker image) and is
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


# --- optional stubs for the camera / radar loaders --------------------------------------------------
# The stock LiDAR-only config still lists LoadMultiViewImageFromFiles and LoadRadarPointsMultiSweeps in its test
# pipeline, although the LiDAR-only model never reads ``img`` or ``radar``.  With SDC_SKIP_CAMERA_RADAR=1 they are
# replaced by cheap stand-ins (black images / empty radar cloud) so the LiDAR-only evaluation needs only
# samples/LIDAR_TOP + sweeps/LIDAR_TOP on disk.  Detections are unaffected because nothing downstream consumes them.
if os.environ.get("SDC_SKIP_CAMERA_RADAR") == "1":
    from PIL import Image
    from mmdet3d.core.points import RadarPoints
    from mmdet3d.datasets.pipelines.loading import LoadMultiViewImageFromFiles as _Cam
    from mmdet3d.datasets.pipelines.loading import LoadRadarPointsMultiSweeps as _Radar

    @PIPELINES.register_module(name="LoadMultiViewImageFromFiles", force=True)
    class StubLoadMultiViewImageFromFiles(_Cam):
        def __call__(self, results):
            imgs = [Image.new("RGB", (1600, 900)) for _ in results["image_paths"]]
            results.update(filename=results["image_paths"], img=imgs, img_shape=imgs[0].size,
                           ori_shape=imgs[0].size, pad_shape=imgs[0].size, scale_factor=1.0)
            return results

    @PIPELINES.register_module(name="LoadRadarPointsMultiSweeps", force=True)
    class StubLoadRadarPointsMultiSweeps(_Radar):
        def __call__(self, results):
            n = len(self.use_dim)
            results["radar"] = RadarPoints(np.zeros((1, n), dtype=np.float32), points_dim=n, attribute_dims=None)
            return results


@PIPELINES.register_module(name="LoadPointsFromMultiSweeps", force=True)
class CorruptedLoadPointsFromMultiSweeps(_Original):
    def __init__(self, *args, corruption=None, severity=None, seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.corruption = corruption or os.environ.get("SDC_CORRUPTION") or None
        self.severity = int(severity if severity is not None else os.environ.get("SDC_SEVERITY", 0))
        self.seed = int(seed if seed is not None else os.environ.get("SDC_SEED", 0))
        self.deterministic_sweeps = os.environ.get("SDC_DETERMINISTIC_SWEEPS") == "1"
        if self.corruption and self.corruption not in C.REGISTRY:
            raise KeyError(f"SDC_CORRUPTION={self.corruption!r}; choose from {sorted(C.REGISTRY)}")

    def __call__(self, results):
        if (not self.corruption or self.severity == 0) and not self.deterministic_sweeps:
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
