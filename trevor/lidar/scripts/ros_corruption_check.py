#!/usr/bin/env python
"""End-to-end check of the rclpy corruption node (run inside the ``sdc-lidar-ros`` image).

    bash lidar/scripts/ros_corruption_check.sh          # builds the package with colcon, starts the node, runs this

Publishes one real nuScenes LIDAR_TOP sweep as PointCloud2 (x y z intensity ring, float32) to the node's input topic,
changes the node's parameters at runtime with ``ros2 param set``, and checks the node's output against the offline
corruption code (same seed + same header stamp => identical surviving points).
"""
import argparse
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointField, PointCloud2

from sdc_lidar import corruptions as C
from sdc_lidar.ros_cloud import rows_to_xyzirr, structured_view


def make_cloud(raw: np.ndarray, stamp_ns: int) -> PointCloud2:
    msg = PointCloud2()
    msg.header.frame_id = "LIDAR_TOP"
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp_ns, 10**9)
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(("x", "y", "z", "intensity", "ring"))]
    msg.point_step, msg.height, msg.width = 20, 1, len(raw)
    msg.row_step = 20 * len(raw)
    msg.is_dense, msg.is_bigendian = True, False
    msg.data = raw.astype("<f4").tobytes()
    return msg


def param_set(name, value):
    r = subprocess.run(["ros2", "param", "set", "/lidar_corruption", name, str(value)],
                       capture_output=True, text=True, timeout=30)
    return r.stdout.strip() + r.stderr.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd", required=True, help="a nuScenes samples/LIDAR_TOP/*.pcd.bin file")
    a = ap.parse_args()
    raw = np.fromfile(a.pcd, dtype=np.float32).reshape(-1, 5)
    REF_ROWS = raw.view([(k, "<f4") for k in ("x", "y", "z", "intensity", "ring")]).reshape(-1)
    print(f"input cloud: {len(raw)} points, max horizontal range {np.hypot(raw[:, 0], raw[:, 1]).max():.1f} m")

    rclpy.init()
    node = rclpy.create_node("corruption_check")
    pub = node.create_publisher(PointCloud2, "/lidar/points", qos_profile_sensor_data)
    got = []
    node.create_subscription(PointCloud2, "/lidar/points_corrupted", got.append, qos_profile_sensor_data)

    def roundtrip(stamp_ns, timeout=10.0):
        got.clear()
        t0 = time.time()
        while not got and time.time() - t0 < timeout:
            pub.publish(make_cloud(raw, stamp_ns))
            rclpy.spin_once(node, timeout_sec=0.2)
        assert got, "no message came back from the node"
        return structured_view(got[-1]).copy()

    # wait for DDS discovery both ways
    t0 = time.time()
    while (pub.get_subscription_count() == 0) and time.time() - t0 < 30:
        rclpy.spin_once(node, timeout_sec=0.2)
    assert pub.get_subscription_count() > 0, "the corruption node is not subscribed - is it running?"

    failures = []

    def check(label, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + label + (f"  [{detail}]" if detail else ""))
        if not ok:
            failures.append(label)

    stamp = 1_533_201_470_448_696_000

    out = roundtrip(stamp)
    check("pass-through (corruption=none) returns the cloud unchanged",
          len(out) == len(raw) and np.array_equal(np.stack([out[k] for k in out.dtype.names], 1), raw))

    print("ros2 param set corruption range_limit ->", param_set("corruption", "range_limit"))
    for sev in (1, 3, 5):
        print(f"ros2 param set severity {sev} ->", param_set("severity", sev))
        out = roundtrip(stamp)
        r = np.hypot(out["x"], out["y"])
        lim = C.RANGE_LIMIT_M[sev - 1]
        check(f"range_limit severity {sev}: all points within {lim:g} m", len(out) > 0 and r.max() <= lim + 1e-4,
              f"{len(out)}/{len(raw)} kept, max range {r.max():.1f} m")
        expect = C.apply("range_limit", rows_to_xyzirr(REF_ROWS),
                         sev, seed=0, frame_key=stamp)
        check(f"range_limit severity {sev}: identical to the offline implementation",
              len(out) == len(expect) and np.array_equal(out["x"], raw[expect[:, 5].astype(int), 0]))

    for name in ("point_drop", "beam_reduction"):
        param_set("corruption", name)
        param_set("severity", 3)
        param_set("seed", 7)
        o1, o2 = roundtrip(stamp), roundtrip(stamp)
        rows = REF_ROWS
        expect = C.apply(name, rows_to_xyzirr(rows), 3, seed=7, frame_key=stamp)
        check(f"{name} severity 3: node output == offline output (same seed+stamp)",
              len(o1) == len(expect) and np.array_equal(o1["x"], o2["x"])
              and np.array_equal(o1["x"], raw[expect[:, 5].astype(int), 0]),
              f"{len(o1)}/{len(raw)} kept, offline {len(expect)}")
        o3 = roundtrip(stamp + 100_000_000)
        if name == "point_drop":
            check("point_drop: a different stamp gives a different random subset", not np.array_equal(o1["x"][:50], o3["x"][:50]))

    bad = param_set("severity", 9)
    check("invalid severity is rejected by the parameter callback", "Setting parameter failed" in bad, bad)

    node.destroy_node()
    rclpy.shutdown()
    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
