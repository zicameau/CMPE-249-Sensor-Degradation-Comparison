"""ROS 2 LiDAR corruption node: PointCloud2 in -> corrupted PointCloud2 out.

Parameters (all can be changed at runtime with ``ros2 param set``):
  input_topic   (str, default /lidar/points)
  output_topic  (str, default /lidar/points_corrupted)
  corruption    (str: none | point_drop | beam_reduction | range_limit | missing_sweeps)
  severity      (int 0..5; 0 = pass-through)
  seed          (int; stochastic corruptions are reproducible per (seed, header stamp))

``missing_sweeps`` acts on the multi-sweep *accumulator* (it removes past sweeps), so this node
passes clouds through unchanged for it; the detector node applies it with
``sdc_lidar.corruptions.missing_sweep_indices`` on its sweep buffer (same table, same seeding).

All original fields and the field layout are preserved - only whole points are removed.
Requires ``sdc_lidar`` on PYTHONPATH (``export PYTHONPATH=<repo>/common:<repo>/lidar:$PYTHONPATH``).
"""
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2

from sdc_lidar.ros_cloud import StreamCorruptor, structured_view


class LidarCorruptionNode(Node):
    def __init__(self):
        super().__init__("lidar_corruption")
        self.declare_parameter("input_topic", "/lidar/points")
        self.declare_parameter("output_topic", "/lidar/points_corrupted")
        self.declare_parameter("corruption", "none")
        self.declare_parameter("severity", 0)
        self.declare_parameter("seed", 0)

        self._core = StreamCorruptor()
        self._apply_params({p.name: p.value for p in self.get_parameters(["corruption", "severity", "seed"])})
        self.add_on_set_parameters_callback(self._on_params)

        self._pub = self.create_publisher(PointCloud2, self.get_parameter("output_topic").value,
                                          qos_profile_sensor_data)
        self._sub = self.create_subscription(PointCloud2, self.get_parameter("input_topic").value,
                                             self._on_cloud, qos_profile_sensor_data)
        self._n = 0
        self.get_logger().info(
            f"corruption={self._core.corruption} severity={self._core.severity} seed={self._core.seed}")

    def _apply_params(self, values):
        cur = dict(corruption=self._core.corruption, severity=self._core.severity, seed=self._core.seed)
        cur.update(values)
        self._core.configure(str(cur["corruption"]), int(cur["severity"]), int(cur["seed"]))

    def _on_params(self, params):
        try:
            self._apply_params({p.name: p.value for p in params
                                if p.name in ("corruption", "severity", "seed")})
        except ValueError as e:
            return SetParametersResult(successful=False, reason=str(e))
        return SetParametersResult(successful=True)

    def _on_cloud(self, msg: PointCloud2):
        if self._core.is_passthrough:
            self._pub.publish(msg)
            return
        rows = structured_view(msg)
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        keep = self._core.surviving_rows(rows, stamp_ns)
        out = PointCloud2()
        out.header = msg.header
        out.fields = msg.fields
        out.point_step = msg.point_step
        out.is_bigendian = msg.is_bigendian
        out.height, out.width = 1, int(len(keep))
        out.row_step = msg.point_step * out.width
        out.is_dense = msg.is_dense
        out.data = rows[keep].tobytes()
        self._pub.publish(out)
        self._n += 1
        if self._n % 100 == 1:
            self.get_logger().info(f"kept {len(keep)}/{len(rows)} points ({100 * len(keep) / max(len(rows), 1):.0f}%)")


def main(args=None):
    rclpy.init(args=args)
    node = LidarCorruptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
