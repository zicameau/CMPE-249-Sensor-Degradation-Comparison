#!/usr/bin/env python
"""Replay real nuScenes samples through the ROS 2 detector node and compare with the OFFLINE evaluation output.

For each chosen keyframe it publishes, oldest -> newest, the 9 previous sweeps and the keyframe as PointCloud2, with the
LiDAR's global pose of every cloud on /tf (``global`` -> ``LIDAR_TOP``). The node (fixed_frame=global) aligns sweeps by
TF; its detections are then converted to the global frame and matched against the boxes the official pipeline wrote to
``results_nusc.json`` for the same sample token. Run via ``ros_detector_check.sh``.
"""
import argparse
import json
import pickle
import subprocess
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import TransformBroadcaster
from vision_msgs.msg import Detection3DArray

sys.path.insert(0, "/workspace/sdc/lidar/ros2/lidar_detector")
from lidar_detector.detector_node import quat_to_matrix, yaw_to_quat  # noqa: E402


def quat_wxyz_to_matrix(q):
    w, x, y, z = q
    return quat_to_matrix(x, y, z, w)


def pose_matrix(t, q_wxyz):
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_matrix(q_wxyz)
    T[:3, 3] = t
    return T


def lidar_to_global(entry, key=False):
    if key:
        return pose_matrix(entry["ego2global_translation"], entry["ego2global_rotation"]) @ \
               pose_matrix(entry["lidar2ego_translation"], entry["lidar2ego_rotation"])
    return pose_matrix(entry["ego2global_translation"], entry["ego2global_rotation"]) @ \
           pose_matrix(entry["sensor2ego_translation"], entry["sensor2ego_rotation"])


CLASS_RANGE = {"car": 50, "truck": 50, "bus": 50, "trailer": 50, "construction_vehicle": 50, "pedestrian": 40,
               "motorcycle": 40, "bicycle": 40, "traffic_cone": 30, "barrier": 30}      # detection_cvpr_2019


def fix(path):
    return path.replace("./data/nuscenes/", "/dataset/nuscenes/")


def make_cloud(raw, stamp_ns):
    m = PointCloud2()
    m.header.frame_id = "LIDAR_TOP"
    m.header.stamp.sec, m.header.stamp.nanosec = divmod(int(stamp_ns), 10**9)
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                for i, n in enumerate(("x", "y", "z", "intensity", "ring"))]
    m.point_step, m.height, m.width = 20, 1, len(raw)
    m.row_step = 20 * len(raw)
    m.is_dense, m.is_bigendian = True, False
    m.data = raw.astype("<f4").tobytes()
    return m


def tf_msg(T, stamp_ns):
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id = "global", "LIDAR_TOP"
    t.header.stamp.sec, t.header.stamp.nanosec = divmod(int(stamp_ns), 10**9)
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, T[:3, 3])
    # rotation matrix -> quaternion
    R = T[:3, :3]
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x, y, z = (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)
    t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = x, y, z, w
    return t


def rot_angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.arccos(np.clip(c, -1, 1)))


def param_set(name, value):
    r = subprocess.run(["ros2", "param", "set", "/lidar_detector", name, str(value)], capture_output=True, text=True,
                       timeout=30)
    return (r.stdout + r.stderr).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", default="/dataset/nuscenes/mini_infos_val.pkl")
    ap.add_argument("--clean", default="/results/mini_sweep_det/clean/results_nusc.json")
    ap.add_argument("--corrupted", default="/results/mini_sweep_det/range_limit_s3_seed0/results_nusc.json")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--score", type=float, default=0.3)
    a = ap.parse_args()

    infos = pickle.load(open(a.infos, "rb"))["infos"]
    chosen = [i for i in infos if len(i["sweeps"]) >= 9][3:3 + a.samples]
    ref = {"clean": json.load(open(a.clean))["results"], "range_limit": json.load(open(a.corrupted))["results"]}

    rclpy.init()
    node = rclpy.create_node("detector_check")
    pub = node.create_publisher(PointCloud2, "/lidar/points",
                                QoSProfile(depth=30, reliability=QoSReliabilityPolicy.RELIABLE))
    br = TransformBroadcaster(node)
    got = []
    node.create_subscription(Detection3DArray, "/lidar/detections", got.append, 10)

    t0 = time.time()
    while (pub.get_subscription_count() == 0 or node.count_publishers("/lidar/detections") == 0) \
            and time.time() - t0 < 240:
        rclpy.spin_once(node, timeout_sec=0.5)
    assert pub.get_subscription_count() > 0, "detector node never subscribed (model failed to load?)"
    time.sleep(1.0)

    failures = []

    def check(label, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + label + (f"  [{detail}]" if detail else ""))
        if not ok:
            failures.append(label)

    def replay(info):
        seq = [(s["data_path"], s["timestamp"], lidar_to_global(s)) for s in reversed(info["sweeps"][:9])]
        seq.append((info["lidar_path"], info["timestamp"], lidar_to_global(info, key=True)))
        got.clear()
        for path, ts, T in seq:
            raw = np.fromfile(fix(path), dtype=np.float32).reshape(-1, 5)
            br.sendTransform(tf_msg(T, ts * 1000))
            rclpy.spin_once(node, timeout_sec=0.05)
            pub.publish(make_cloud(raw, ts * 1000))
            rclpy.spin_once(node, timeout_sec=0.05)
        key_ns = seq[-1][1] * 1000
        t1 = time.time()

        def key_msgs():
            return [m for m in got if m.header.stamp.sec * 10**9 + m.header.stamp.nanosec == key_ns]

        while not key_msgs() and time.time() - t1 < 60:
            rclpy.spin_once(node, timeout_sec=0.1)
        assert key_msgs(), "no detections for the key cloud came back"
        return key_msgs()[-1], seq[-1][2]

    def compare(mode, det_msg, T_gl, token, results):
        offline = [b for b in results[token] if b["detection_score"] >= a.score]
        online = []
        for d in det_msg.detections:
            c = d.bbox.center
            R = T_gl[:3, :3] @ quat_to_matrix(c.orientation.x, c.orientation.y, c.orientation.z, c.orientation.w)
            p = T_gl[:3, :3] @ np.array([c.position.x, c.position.y, c.position.z]) + T_gl[:3, 3]
            online.append((d.results[0].id, d.results[0].score, p, R,
                           np.array([d.bbox.size.x, d.bbox.size.y, d.bbox.size.z])))
        matched, worst = 0, dict(dist=0.0, size=0.0, rot=0.0, score=0.0)
        for b in offline:
            cand = [o for o in online if o[0] == b["detection_name"]]
            if not cand:
                continue
            o = min(cand, key=lambda o: np.linalg.norm(o[2] - np.array(b["translation"])))
            dist = np.linalg.norm(o[2] - np.array(b["translation"]))
            w, l, h = b["size"]
            if dist < 0.05:
                matched += 1
                worst["dist"] = max(worst["dist"], dist)
                worst["size"] = max(worst["size"], float(np.abs(o[4] - np.array([l, w, h])).max()))
                worst["rot"] = max(worst["rot"], rot_angle(o[3], quat_wxyz_to_matrix(b["rotation"])))
                worst["score"] = max(worst["score"], abs(o[1] - b["detection_score"]))
        n_off = len(offline)
        # BEVFusion's result writer drops boxes beyond the nuScenes per-class evaluation range; a live node keeps them.
        # So extra node boxes are fine if (and only if) they sit at/over that range (2 m slack: ego vs LiDAR origin).
        inv = np.linalg.inv(T_gl)
        used = set()
        for b in offline:
            cand = [(k, o) for k, o in enumerate(online) if o[0] == b["detection_name"] and k not in used]
            if cand:
                k, o = min(cand, key=lambda ko: np.linalg.norm(ko[1][2] - np.array(b["translation"])))
                if np.linalg.norm(o[2] - np.array(b["translation"])) < 0.05:
                    used.add(k)
        extras = [o for k, o in enumerate(online) if k not in used]
        unexplained = [o for o in extras
                       if np.hypot(*(inv[:3, :3] @ o[2] + inv[:3, 3])[:2]) < CLASS_RANGE[o[0]] - 2.0]
        check(f"[{mode}] {token[:8]}: node matches offline boxes",
              n_off > 0 and matched == n_off and not unexplained,
              f"matched {matched}/{n_off} offline, node emitted {len(online)} ({len(extras)} extra, beyond class range; "
              f"{len(unexplained)} unexplained); worst |d|={worst['dist']:.4f} m size={worst['size']:.4f} m "
              f"rot={worst['rot']:.4f} rad score={worst['score']:.4f}")
        return online

    print(f"replaying {len(chosen)} samples from {chosen[0]['token'][:8]}...")
    for info in chosen:
        msg, T = replay(info)
        compare("clean", msg, T, info["token"], ref["clean"])

    print("ros2 param set corruption range_limit ->", param_set("corruption", "range_limit"))
    print("ros2 param set severity 3 ->", param_set("severity", 3))
    for info in chosen:
        msg, T = replay(info)
        online = compare("range_limit s3", msg, T, info["token"], ref["range_limit"])
        key_T_inv = np.linalg.inv(T)
        r = [np.hypot(*(key_T_inv[:3, :3] @ o[2] + key_T_inv[:3, 3])[:2]) for o in online]
        check(f"[range_limit s3] {info['token'][:8]}: no detection centred beyond ~35 m",
              all(x < 35 for x in r), f"max centre range {max(r) if r else 0:.1f} m")

    print("ros2 param set severity 9 ->", param_set("severity", 9))
    check("invalid severity rejected", "failed" in param_set("severity", 9))

    node.destroy_node()
    rclpy.shutdown()
    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILED: ' + '; '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
