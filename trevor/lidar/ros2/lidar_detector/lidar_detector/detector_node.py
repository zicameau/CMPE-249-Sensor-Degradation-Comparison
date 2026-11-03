"""ROS 2 LiDAR-only 3D detector node (BEVFusion LiDAR-only / TransFusion-L).

PointCloud2 in  ->  vision_msgs/Detection3DArray (+ MarkerArray for RViz) out, in the input cloud's frame.

Input clouds need fields x y z intensity [ring] as float32, intensity on the nuScenes scale (0..255).

Parameters (corruption / severity / seed / score_threshold can be changed at runtime with ``ros2 param set``):
  input_topic       /lidar/points
  detections_topic  /lidar/detections           (vision_msgs/Detection3DArray; hypothesis id = class name)
  markers_topic     /lidar/detections_markers   (visualization_msgs/MarkerArray)
  config            configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml   (relative to the BEVFusion repo)
  checkpoint        pretrained/lidar-only-det.pth
  score_threshold   0.3
  sweeps_num        9      previous sweeps accumulated with the key cloud (published baseline: 9; 0 = key cloud only)
  fixed_frame       ""     TF frame in which ego motion is known (e.g. "odom"/"map"). Empty = static sensor: sweeps are
                           stacked with identity transforms (fine for a parked vehicle, wrong while moving).
  corruption        none | point_drop | beam_reduction | range_limit | missing_sweeps
  severity          0..5 (0 = off)
  seed              int
  reliable          false  use RELIABLE QoS (depth 30) instead of best-effort sensor QoS: needed when no cloud
                           may be lost (offline replay / rosbag); live drivers normally publish best-effort
  verbose           false  log every processed cloud
  deterministic_voxelization true  stock (published) setting; false is ~2x faster, see detector_core.py

The corruption is applied *inside* the sweep accumulator, exactly like the offline sweep (same code, same seeding by
header stamp), so ``missing_sweeps`` works too. Chaining the separate ``lidar_corruption`` node in front is possible
for the per-sweep corruptions but cannot express ``missing_sweeps``.

Run from the BEVFusion checkout (cwd) so the relative config / checkpoint paths resolve, with
``PYTHONPATH=<repo>/common:<repo>/lidar:$PYTHONPATH``.
"""
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rcl_interfaces.msg import SetParametersResult
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from sdc_lidar import corruptions as C
from sdc_lidar.detector_core import (CLASS_NAMES, LidarOnlyDetector, SweepBuffer, box_to_nuscenes, range_filter)
from sdc_lidar.ros_cloud import rows_to_xyzirr, structured_view

CONFIG_DEFAULT = "configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml"


def yaw_to_quat(yaw):
    return 0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))


def quat_to_matrix(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class LidarDetectorNode(Node):
    def __init__(self):
        super().__init__("lidar_detector")
        for name, default in (("input_topic", "/lidar/points"), ("detections_topic", "/lidar/detections"),
                              ("markers_topic", "/lidar/detections_markers"), ("config", CONFIG_DEFAULT),
                              ("checkpoint", "pretrained/lidar-only-det.pth"), ("fixed_frame", ""),
                              ("corruption", "none")):
            self.declare_parameter(name, default)
        self.declare_parameter("score_threshold", 0.3)
        self.declare_parameter("sweeps_num", 9)
        self.declare_parameter("severity", 0)
        self.declare_parameter("seed", 0)
        self.declare_parameter("reliable", False)
        self.declare_parameter("deterministic_voxelization", True)
        self.declare_parameter("verbose", False)
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.get_logger().info("loading model ...")
        t0 = time.time()
        self.detector = LidarOnlyDetector(g("config"), g("checkpoint"), score_threshold=g("score_threshold"),
                                      deterministic_voxelization=g("deterministic_voxelization"))
        self.get_logger().info(f"model ready in {time.time() - t0:.1f}s")

        self.buffer = SweepBuffer(sweeps_num=g("sweeps_num"))
        self.fixed_frame = g("fixed_frame")
        self.corruption, self.severity, self.seed = g("corruption"), g("severity"), g("seed")
        self._check_corruption(self.corruption, self.severity)
        if self.fixed_frame:
            self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
            # own spin thread: TF keeps filling while a cloud callback is blocked waiting for its transform
            self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        elif self.buffer.sweeps_num > 0:
            self.get_logger().warn("fixed_frame is empty: sweeps are stacked WITHOUT ego-motion compensation "
                                   "(set fixed_frame, or sweeps_num=0)")
        self.add_on_set_parameters_callback(self._on_params)

        self.pub_det = self.create_publisher(Detection3DArray, g("detections_topic"), 10)
        self.pub_mk = self.create_publisher(MarkerArray, g("markers_topic"), 10)
        qos = (QoSProfile(depth=30, reliability=QoSReliabilityPolicy.RELIABLE) if g("reliable")
               else qos_profile_sensor_data)
        self.verbose = bool(g("verbose"))
        self.create_subscription(PointCloud2, g("input_topic"), self._on_cloud, qos)
        self._n = 0
        self._cv = threading.Condition()
        self._pending = None
        threading.Thread(target=self._worker, daemon=True).start()

    @staticmethod
    def _check_corruption(name, severity):
        if name not in ("none", "") and name not in C.REGISTRY:
            raise ValueError(f"unknown corruption {name!r}")
        if severity not in (0, *C.SEVERITIES):
            raise ValueError(f"severity must be 0..5, got {severity}")

    def _on_params(self, params):
        new = {"corruption": self.corruption, "severity": self.severity, "seed": self.seed}
        for p in params:
            if p.name in new:
                new[p.name] = p.value
            elif p.name == "score_threshold":
                self.detector.score_threshold = float(p.value)
        try:
            self._check_corruption(new["corruption"], new["severity"])
        except ValueError as e:
            return SetParametersResult(successful=False, reason=str(e))
        self.corruption, self.severity, self.seed = new["corruption"], int(new["severity"]), int(new["seed"])
        return SetParametersResult(successful=True)

    def _lookup(self, key_frame, key_ns, sweep_frame, sweep_ns):
        """sweep -> key transform through the fixed frame (p_key = R p_sweep + t)."""
        try:
            tf = self.tf_buffer.lookup_transform_full(
                key_frame, Time(nanoseconds=int(key_ns)), sweep_frame, Time(nanoseconds=int(sweep_ns)),
                self.fixed_frame, timeout=Duration(seconds=0.5))   # TF and clouds travel on different topics
        except Exception as e:  # noqa: BLE001  (extrapolation / missing pose: skip that sweep)
            self.get_logger().warn(f"no TF for a sweep, skipping it: {e}", throttle_duration_sec=5.0)
            return None
        q, tr = tf.transform.rotation, tf.transform.translation
        return quat_to_matrix(q.x, q.y, q.z, q.w), np.array([tr.x, tr.y, tr.z])

    def _on_cloud(self, msg: PointCloud2):
        """Ingest every cloud (cheap); the worker thread runs the model on the newest one only."""
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        raw = rows_to_xyzirr(structured_view(msg))[:, :5]
        self.buffer.push(stamp_ns, msg.header.frame_id, raw)
        with self._cv:
            self._pending = (msg.header, stamp_ns)       # overwrites an unprocessed older key: latest wins
            self._cv.notify()

    def _worker(self):
        while rclpy.ok():
            with self._cv:
                while self._pending is None:
                    if not self._cv.wait(timeout=0.5) and not rclpy.ok():
                        return
                header, stamp_ns = self._pending
                self._pending = None
            try:
                self._process(header, stamp_ns)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"detection failed: {e!r}")

    def _process(self, header, stamp_ns):
        t0 = time.time()
        points, used = self.buffer.build(
            self._lookup if self.fixed_frame else None, key_stamp_ns=stamp_ns,
            corruption=None if self.corruption in ("none", "") else self.corruption,
            severity=self.severity, seed=self.seed, frame_key=stamp_ns)
        points = range_filter(points)
        t1 = time.time()
        boxes, scores, labels = self.detector(points)
        t2 = time.time()
        self._publish(header, boxes, scores, labels)
        self._n += 1
        if self.verbose or self._n % 20 == 1:
            self.get_logger().info(f"{len(boxes)} detections | {used} sweeps | {len(points)} pts | "
                                   f"prep {1e3 * (t1 - t0):.0f} ms, model {1e3 * (t2 - t1):.0f} ms")
    def _publish(self, header, boxes, scores, labels):
        centre, size, heading = box_to_nuscenes(boxes) if len(boxes) else (np.zeros((0, 3)),) * 3
        det_arr = Detection3DArray()
        det_arr.header = header
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        for i in range(len(boxes)):
            d = Detection3D()
            d.header = header
            h = ObjectHypothesisWithPose()
            h.id = CLASS_NAMES[int(labels[i])]
            h.score = float(scores[i])
            d.results.append(h)
            d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z = map(float, centre[i])
            (d.bbox.center.orientation.x, d.bbox.center.orientation.y,
             d.bbox.center.orientation.z, d.bbox.center.orientation.w) = yaw_to_quat(heading[i])
            d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = map(float, size[i])
            det_arr.detections.append(d)

            m = Marker()
            m.header, m.ns, m.id, m.type, m.action = header, "det", i, Marker.CUBE, Marker.ADD
            m.pose = d.bbox.center
            m.scale = d.bbox.size
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.2, 0.5
            m.lifetime = Duration(seconds=0.3).to_msg()
            markers.markers.append(m)
        self.pub_det.publish(det_arr)
        self.pub_mk.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = LidarDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
