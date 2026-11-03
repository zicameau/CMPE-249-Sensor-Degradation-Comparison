"""PointCloud2 <-> numpy helpers and the per-message corruption logic for the ROS 2 node.

No rclpy / sensor_msgs import here, so it is unit-testable on any machine. ``msg`` is anything with
the PointCloud2 attributes (fields, point_step, width, height, data).

Key idea: every per-sweep corruption only *removes points*, so the node corrupts a small
``[x, y, z, intensity, ring, row_index]`` array and then selects the surviving **rows of the original
buffer**. All original fields (timestamp, ring, ... whatever the driver publishes) and the original
field layout pass through untouched.
"""
from __future__ import annotations

import numpy as np

from . import corruptions as C

_DATATYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def structured_view(msg) -> np.ndarray:
    """Zero-copy structured array of the cloud's rows (little-endian clouds only)."""
    if getattr(msg, "is_bigendian", False):
        raise ValueError("big-endian PointCloud2 is not supported")
    names, formats, offsets = [], [], []
    for f in msg.fields:
        if f.count != 1:
            continue
        names.append(f.name)
        formats.append("<" + _DATATYPES[f.datatype])
        offsets.append(f.offset)
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step})
    n = msg.width * msg.height
    return np.frombuffer(bytes(msg.data) if not isinstance(msg.data, (bytes, bytearray)) else msg.data,
                         dtype=dtype, count=n)


def rows_to_xyzirr(rows: np.ndarray) -> np.ndarray:
    """(N, 6) float32: x y z intensity ring row_index. Missing intensity -> 0; missing ring -> estimated."""
    n = len(rows)
    names = rows.dtype.names
    out = np.zeros((n, 6), dtype=np.float32)
    for j, k in enumerate(("x", "y", "z")):
        out[:, j] = rows[k]
    if "intensity" in names:
        out[:, C.INTENSITY] = rows["intensity"]
    if "ring" in names:
        out[:, C.RING] = rows["ring"]
    else:
        out[:, C.RING] = C.estimate_rings(out)
    out[:, 5] = np.arange(n, dtype=np.float32)
    return out


class StreamCorruptor:
    """Stateless-per-message corruption with the same seeding rule as the offline sweep."""

    PASSTHROUGH = ("none", "clean", "", "missing_sweeps")

    def __init__(self, corruption: str = "none", severity: int = 0, seed: int = 0):
        self.configure(corruption, severity, seed)

    def configure(self, corruption: str, severity: int, seed: int) -> None:
        if corruption not in self.PASSTHROUGH and corruption not in C.REGISTRY:
            raise ValueError(f"unknown corruption {corruption!r}")
        if severity not in (0, *C.SEVERITIES):
            raise ValueError(f"severity must be 0..5, got {severity}")
        self.corruption, self.severity, self.seed = corruption, int(severity), int(seed)

    @property
    def is_passthrough(self) -> bool:
        return self.corruption in self.PASSTHROUGH or self.severity == 0

    def surviving_rows(self, rows: np.ndarray, frame_key) -> np.ndarray:
        """Indices (into ``rows``) of the points that survive. ``frame_key``: e.g. the header stamp in ns."""
        if self.is_passthrough:
            return np.arange(len(rows))
        arr = rows_to_xyzirr(rows)
        out = C.apply(self.corruption, arr, self.severity, seed=self.seed, frame_key=int(frame_key))
        return out[:, 5].astype(np.int64)
