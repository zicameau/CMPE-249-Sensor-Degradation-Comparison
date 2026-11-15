
#!/usr/bin/env python3

from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


DATASET_DIR = Path(
    "/mnt/c/SJSU/datasets/nuScenes/samples/CAM_FRONT"
)


class NuScenesPublisher(Node):
    def __init__(self):
        super().__init__("nuscenes_image_publisher")

        self.publisher = self.create_publisher(
            Image, "/camera/image_raw", 10
        )

        self.bridge = CvBridge()

        image_files = sorted(DATASET_DIR.glob("*.jpg"))

        if not image_files:
            raise FileNotFoundError(
                f"No images found in {DATASET_DIR}"
            )

        self.image_path = image_files[0]
        self.image = cv2.imread(str(self.image_path))

        if self.image is None:
            raise RuntimeError(
                f"Could not read {self.image_path}"
            )

        self.timer = self.create_timer(1.0, self.publish_image)

        self.get_logger().info(
            f"Loaded: {self.image_path.name}"
        )
        self.get_logger().info(
            f"Image dimensions: {self.image.shape}"
        )

    def publish_image(self):
        msg = self.bridge.cv2_to_imgmsg(
            self.image, encoding="bgr8"
        )

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "CAM_FRONT"

        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = NuScenesPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
