
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class ImageChecker(Node):
    def __init__(self):
        super().__init__("image_checker")

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            "/camera/image_corrupted",
            self.callback,
            10,
        )

    def callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="bgr8"
        )

        print("Image shape:", image.shape)
        print("Mean pixel value:", np.mean(image))
        print("Frame ID:", msg.header.frame_id)

        raise SystemExit


def main():
    rclpy.init()
    node = ImageChecker()

    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
