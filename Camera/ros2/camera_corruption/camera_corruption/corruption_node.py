"""ROS 2 image corruption node for camera robustness experiments."""
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

BRIGHTNESS_FACTORS = {0: 1.0, 1: 0.5, 2: 0.25, 3: 0.1}
BLUR_SIGMAS = {0: 0.0, 1: 2.0, 2: 5.0, 3: 10.0}


def apply_corruption(image, corruption, severity):
    """Apply deterministic degradation to a BGR uint8 image."""
    if severity not in BRIGHTNESS_FACTORS:
        raise ValueError('severity must be 0, 1, 2, or 3')
    if corruption == 'none' or severity == 0:
        return image.copy()
    if corruption == 'brightness':
        factor = BRIGHTNESS_FACTORS[severity]
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if corruption == 'blur':
        return cv2.GaussianBlur(image, (0, 0), BLUR_SIGMAS[severity])
    raise ValueError("corruption must be 'none', 'brightness', or 'blur'")


class CameraCorruptionNode(Node):
    def __init__(self):
        super().__init__('camera_corruption_node')
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/camera/image_corrupted')
        self.declare_parameter('corruption', 'brightness')
        self.declare_parameter('severity', 1)
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.corruption = self.get_parameter('corruption').value
        self.severity = self.get_parameter('severity').value
        if self.corruption not in ('none', 'brightness', 'blur'):
            raise ValueError('Invalid corruption parameter')
        if self.severity not in BRIGHTNESS_FACTORS:
            raise ValueError('Invalid severity parameter')
        if input_topic == output_topic:
            raise ValueError('input_topic and output_topic must differ')
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, output_topic, 10)
        self.subscription = self.create_subscription(Image, input_topic, self.image_callback, 10)
        self.get_logger().info(
            f'{input_topic} -> {output_topic} | corruption={self.corruption}, severity={self.severity}'
        )

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            output = apply_corruption(image, self.corruption, self.severity)
            output_msg = self.bridge.cv2_to_imgmsg(output, encoding='bgr8')
            output_msg.header = msg.header
            self.publisher.publish(output_msg)
        except Exception as exc:
            self.get_logger().error(f'Failed to process image: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CameraCorruptionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
