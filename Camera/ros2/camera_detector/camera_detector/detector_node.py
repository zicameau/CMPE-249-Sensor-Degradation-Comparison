
import time

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

from ultralytics import YOLO
from sdc_camera.latency import LatencyRecorder


class CameraDetectorNode(Node):
    def __init__(self):
        super().__init__("camera_detector_node")

        self.declare_parameter(
            "input_topic", "/camera/image_corrupted"
        )
        self.declare_parameter(
            "output_topic", "/camera/detections"
        )
        self.declare_parameter("model_path", "yolo11s.pt")
        self.declare_parameter("confidence", 0.25)
        self.declare_parameter(
            "latency_csv", "results/ros2_detector_latency.csv"
        )
        self.declare_parameter("warmup_frames", 5)

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        model_path = self.get_parameter("model_path").value

        self.confidence = float(
            self.get_parameter("confidence").value
        )

        latency_csv = self.get_parameter("latency_csv").value
        warmup_frames = int(
            self.get_parameter("warmup_frames").value
        )

        self.recorder = LatencyRecorder(
            latency_csv,
            warmup_frames=warmup_frames,
        )

        self.bridge = CvBridge()

        self.get_logger().info(
            f"Loading YOLO model: {model_path}"
        )
        self.model = YOLO(model_path)

        # Initialize YOLO before subscribing to real camera images.
        # These predictions are excluded from latency measurements.
        self.get_logger().info("Warming up YOLO11s on CPU...")

        warmup_image = np.zeros(
            (900, 1600, 3), dtype=np.uint8
        )

        warmup_start = time.perf_counter()

        for _ in range(3):
            self.model.predict(
                source=warmup_image,
                conf=self.confidence,
                verbose=False,
                device="cpu",
            )

        warmup_seconds = time.perf_counter() - warmup_start

        self.get_logger().info(
            f"Model warm-up complete: {warmup_seconds:.2f} seconds"
        )

        self.publisher = self.create_publisher(
            Detection2DArray, output_topic, 10
        )

        self.subscription = self.create_subscription(
            Image, input_topic, self.image_callback, 10
        )

        self.get_logger().info(
            f"YOLO11 detector ready: "
            f"{input_topic} -> {output_topic}"
        )
        self.get_logger().info(
            f"Latency CSV: {latency_csv}"
        )

    def image_callback(self, msg):
        callback_start = time.perf_counter()

        try:
            image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )

            inference_start = time.perf_counter()

            results = self.model.predict(
                source=image,
                conf=self.confidence,
                verbose=False,
                device="cpu",
            )

            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000.0

            output = Detection2DArray()
            output.header = msg.header

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = (
                        box.xyxy[0].cpu().numpy().tolist()
                    )

                    detection = Detection2D()
                    detection.header = msg.header

                    detection.bbox.center.position.x = (
                        x1 + x2
                    ) / 2.0
                    detection.bbox.center.position.y = (
                        y1 + y2
                    ) / 2.0

                    detection.bbox.size_x = x2 - x1
                    detection.bbox.size_y = y2 - y1

                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = str(
                        int(box.cls[0].item())
                    )
                    hypothesis.hypothesis.score = float(
                        box.conf[0].item()
                    )

                    detection.results.append(hypothesis)
                    output.detections.append(detection)

            self.publisher.publish(output)

            callback_ms = (
                time.perf_counter() - callback_start
            ) * 1000.0

            record = self.recorder.add(
                inference_ms=inference_ms,
                callback_ms=callback_ms,
                detection_count=len(output.detections),
            )

            self.recorder.save()

            self.get_logger().info(
                f"Frame {record['frame_index']} "
                f"| Detections: {len(output.detections)} "
                f"| Inference: {inference_ms:.1f} ms "
                f"| Callback: {callback_ms:.1f} ms"
            )

        except Exception as exc:
            self.get_logger().error(
                f"Detection failed: {exc}"
            )

    def print_latency_summary(self):
        summary = self.recorder.summary()

        if summary is None:
            self.get_logger().info(
                "Not enough frames for latency summary."
            )
            return

        self.get_logger().info(
            "Latency summary (excluding warmup): "
            f"{summary['frames']} frames"
        )
        self.get_logger().info(
            f"Inference mean: "
            f"{summary['inference_mean_ms']:.2f} ms"
        )
        self.get_logger().info(
            f"Inference median: "
            f"{summary['inference_median_ms']:.2f} ms"
        )
        self.get_logger().info(
            f"Inference P95: "
            f"{summary['inference_p95_ms']:.2f} ms"
        )
        self.get_logger().info(
            f"Callback mean: "
            f"{summary['callback_mean_ms']:.2f} ms"
        )
        self.get_logger().info(
            f"Callback P95: "
            f"{summary['callback_p95_ms']:.2f} ms"
        )
        self.get_logger().info(
            f"Estimated inference FPS: "
            f"{summary['inference_fps_estimate']:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CameraDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.print_latency_summary()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
