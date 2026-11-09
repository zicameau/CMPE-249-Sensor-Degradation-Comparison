
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class TestCameraPublisher(Node):
    def __init__(self):
        super().__init__("test_camera_publisher")

        self.publisher = self.create_publisher(
            Image, "/camera/image_raw", 10
        )

        self.bridge = CvBridge()
        self.timer = self.create_timer(1.0, self.publish_image)

    def publish_image(self):
        # Generate a bright synthetic BGR image.
        image = np.full(
            (480, 640, 3), 200, dtype=np.uint8
        )

        msg = self.bridge.cv2_to_imgmsg(
            image, encoding="bgr8"
        )

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_front"

        self.publisher.publish(msg)
        self.get_logger().info("Published test image")


def main():
    rclpy.init()
    node = TestCameraPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
