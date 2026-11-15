"""Project nuScenes-format global 3D detections into camera 2D boxes."""

import json
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points

from .projection import COMMON_CLASSES, common_class


def prediction_class(name):
    """Map nuScenes detection names to the shared six classes."""
    if name == "pedestrian":
        return "person"
    if name in COMMON_CLASSES:
        return name
    return common_class(name)


def project_predictions(nusc, results, camera="CAM_FRONT",
                        allowed_tokens=None):
    """
    Convert nuScenes detection results into Camera evaluator rows.

    results: dict mapping sample_token to lists of nuScenes detections.
    allowed_tokens: optional set of sample tokens to evaluate.

    Policy: skip boxes crossing the camera near plane (z <= 0.1 m),
    matching the existing ground-truth projection policy.
    """
    allowed = set(allowed_tokens) if allowed_tokens is not None else None
    rows = []

    for sample_token, detections in results.items():
        if allowed is not None and sample_token not in allowed:
            raise ValueError(
                f"Prediction sample not in manifest: {sample_token}"
            )

        sample = nusc.get("sample", sample_token)
        camera_data = nusc.get(
            "sample_data", sample["data"][camera]
        )
        ego_pose = nusc.get(
            "ego_pose", camera_data["ego_pose_token"]
        )
        calibration = nusc.get(
            "calibrated_sensor",
            camera_data["calibrated_sensor_token"]
        )

        K = np.asarray(calibration["camera_intrinsic"])
        width = camera_data["width"]
        height = camera_data["height"]

        ego_q = Quaternion(ego_pose["rotation"])
        sensor_q = Quaternion(calibration["rotation"])

        for det in detections:
            cls = prediction_class(det["detection_name"])
            if cls is None:
                continue

            box = Box(
                center=det["translation"],
                size=det["size"],
                orientation=Quaternion(det["rotation"]),
            )

            # Global -> ego vehicle -> camera coordinates.
            box.translate(-np.asarray(ego_pose["translation"]))
            box.rotate(ego_q.inverse)
            box.translate(-np.asarray(calibration["translation"]))
            box.rotate(sensor_q.inverse)

            corners = box.corners()

            if np.any(corners[2] <= 0.1):
                continue

            uv = view_points(corners, K, normalize=True)[:2]
            x1, y1 = uv.min(axis=1)
            x2, y2 = uv.max(axis=1)

            x1, x2 = np.clip([x1, x2], 0, width)
            y1, y2 = np.clip([y1, y2], 0, height)

            if x2 <= x1 or y2 <= y1:
                continue

            rows.append({
                "sample_token": sample_token,
                "camera": camera,
                "cls": cls,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "confidence": float(det["detection_score"]),
            })

    return rows


def load_nuscenes_predictions(path):
    """Read the official nuScenes detection submission structure."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not isinstance(
        data.get("results"), dict
    ):
        raise ValueError(
            "Expected nuScenes JSON with a 'results' dictionary"
        )

    return data["results"]
