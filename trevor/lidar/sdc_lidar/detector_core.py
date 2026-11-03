"""Core of the LiDAR-only ROS 2 detector node. No rclpy here, so it can be tested without ROS.

* ``SweepBuffer``        keeps recent raw clouds and builds the accumulated, ego-motion-aligned input with the SAME
                         ``multisweep.accumulate`` the offline sweep uses (so corruptions, incl. ``missing_sweeps``,
                         behave identically online and offline).
* ``LidarOnlyDetector``  BEVFusion LiDAR-only (TransFusion-L) built from the stock config + checkpoint (needs the
                         BEVFusion/mmdet3d image; torch is imported lazily so the buffer is unit-testable on a host).
* ``box_to_nuscenes``    the exact box-parameter conversion of BEVFusion's ``output_to_nusc_box``.

Input contract: raw nuScenes-style clouds, columns ``x y z intensity ring``; intensity on the nuScenes scale (0..255).
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .multisweep import RawSweep, accumulate

CLASS_NAMES = ("car", "truck", "construction_vehicle", "bus", "trailer",
               "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone")
POINT_CLOUD_RANGE = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)

# (key_frame, key_stamp_ns, sweep_frame, sweep_stamp_ns) -> (R (3,3), t (3,)) with  p_key = R @ p_sweep + t, or None.
# Stamps are integer nanoseconds: float seconds cannot represent them exactly and TF lookups at the cache edge care.
TransformLookup = Callable[[str, int, str, int], Optional[Tuple[np.ndarray, np.ndarray]]]


@dataclass
class _Cloud:
    stamp_ns: int
    frame_id: str
    points: np.ndarray          # (N, 5) raw


class SweepBuffer:
    def __init__(self, sweeps_num: int = 9, max_age_s: float = 1.0):
        self.sweeps_num = int(sweeps_num)
        self.max_age_ns = int(max_age_s * 1e9)
        # extra slots: clouds keep arriving while the model is busy with an earlier key cloud
        self._clouds: deque[_Cloud] = deque(maxlen=max(self.sweeps_num, 1) + 12)
        self._lock = threading.Lock()

    def push(self, stamp_ns: int, frame_id: str, points: np.ndarray) -> None:
        with self._lock:
            self._clouds.append(_Cloud(int(stamp_ns), frame_id, np.asarray(points, np.float32)))

    def build(self, lookup: Optional[TransformLookup], *, key_stamp_ns: Optional[int] = None,
              corruption: str | None = None, severity: int = 0, seed: int = 0, frame_key: int | str = "",
              use_dim: Sequence[int] = (0, 1, 2, 3, 4)) -> Tuple[np.ndarray, int]:
        """Accumulate a key cloud (default: the newest) with up to ``sweeps_num`` OLDER ones, nearest first.

        ``lookup=None`` assumes a static sensor (identity transforms). Returns (points (M, 5), sweeps_used)."""
        with self._lock:
            clouds = list(self._clouds)
        if key_stamp_ns is not None:
            clouds = [c for c in clouds if c.stamp_ns <= key_stamp_ns]
        if not clouds:
            raise RuntimeError("empty buffer")
        key = clouds[-1]
        sweeps = []
        for c in reversed(clouds[:-1]):                         # most recent previous sweep first, as in the info files
            if key.stamp_ns - c.stamp_ns > self.max_age_ns or len(sweeps) >= self.sweeps_num:
                break
            if lookup is None:
                R, t = np.eye(3), np.zeros(3)
            else:
                tr = lookup(key.frame_id, key.stamp_ns, c.frame_id, c.stamp_ns)
                if tr is None:
                    continue                                    # no pose for this sweep: skip it rather than mis-align
                R, t = tr
            sweeps.append(RawSweep(c.points, np.asarray(R, np.float32), np.asarray(t, np.float32), c.stamp_ns * 1e-9))
        pts = accumulate(key.points, key.stamp_ns * 1e-9, sweeps, sweeps_num=self.sweeps_num, use_dim=use_dim,
                         corruption=corruption, severity=severity, seed=seed, frame_key=frame_key)
        return pts, len(sweeps)

def range_filter(points: np.ndarray, pcr: Sequence[float] = POINT_CLOUD_RANGE) -> np.ndarray:
    m = ((points[:, 0] >= pcr[0]) & (points[:, 0] <= pcr[3]) & (points[:, 1] >= pcr[1]) & (points[:, 1] <= pcr[4])
         & (points[:, 2] >= pcr[2]) & (points[:, 2] <= pcr[5]))
    return points[m]


def box_to_nuscenes(boxes: np.ndarray):
    """BEVFusion LiDAR boxes (x y z dx dy dz yaw vx vy) -> centre, size (length, width, height), heading.

    Same as ``output_to_nusc_box`` (nuscenes_dataset.py): ``dims`` (columns 3:6) are handed to ``NuScenesBox`` as
    (w, l, h), and heading = -yaw - pi/2. So the box's *width* is column 3, its *length* is column 4, and the length
    axis points along ``heading`` in the LiDAR frame (ROS convention: box x = length).
    """
    boxes = np.asarray(boxes, np.float64).reshape(-1, boxes.shape[-1])
    centre = boxes[:, :3].copy()
    centre[:, 2] += boxes[:, 5] * 0.5        # BEVFusion boxes are bottom-centred -> gravity centre (as in the converter)
    size = boxes[:, [4, 3, 5]]              # length, width, height
    heading = -boxes[:, 6] - np.pi / 2
    return centre, size, heading


class LidarOnlyDetector:
    def __init__(self, config: str, checkpoint: str, device: str = "cuda", score_threshold: float = 0.3,
                 deterministic_voxelization: bool = True):
        import torch
        from mmcv import Config
        from mmcv.runner import load_checkpoint
        from mmdet3d.models import build_model
        from mmdet3d.utils import recursive_eval
        from torchpack.utils.config import configs

        configs.load(config, recursive=True)
        cfg = Config(recursive_eval(configs), filename=config)
        # Same preparation as tools/test.py: the stock config has an ``fp16`` section, so the published numbers (and
        # the offline sweep) come from fp16 inference. Not wrapping gives different scores AND a ~10x slower model.
        cfg.model.train_cfg = None
        self.model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        if cfg.get("fp16", None) is not None:
            from mmcv.runner import wrap_fp16_model
            wrap_fp16_model(self.model)
        load_checkpoint(self.model, checkpoint, map_location="cpu")
        # The stock config voxelizes deterministically (what the published numbers used). That kernel is ~200 ms of a
        # ~450 ms frame on a laptop GPU; False is ~2x faster overall (top-20 scores moved by <1e-3 in our test) but
        # picks an arbitrary 10 points in over-full voxels.
        self.model.encoders["lidar"]["voxelize"].deterministic = bool(deterministic_voxelization)
        self.model.to(device).eval()
        self.device = device
        self.score_threshold = float(score_threshold)
        self._torch = torch

    def __call__(self, points: np.ndarray):
        """points: (M, 5) accumulated cloud -> (boxes (K, 9), scores (K,), labels (K,)) above the score threshold."""
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        torch = self._torch
        pts = torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32)).to(self.device)
        metas = [dict(box_type_3d=LiDARInstance3DBoxes)]
        with torch.no_grad():
            # forward_single(img, points, camera2ego, lidar2ego, lidar2camera, lidar2image, camera_intrinsics,
            #                camera2lidar, img_aug_matrix, lidar_aug_matrix, metas): the LiDAR-only model reads
            # nothing but ``points`` and ``metas['box_type_3d']``.
            out = self.model.forward_single(None, [pts], None, None, None, None, None, None, None, None, metas)
        o = out[0]
        boxes, scores, labels = o["boxes_3d"].tensor.numpy(), o["scores_3d"].numpy(), o["labels_3d"].numpy()
        keep = scores >= self.score_threshold
        return boxes[keep], scores[keep], labels[keep]
