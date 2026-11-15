"""Replay nuScenes CAM_FRONT images through ROS 2."""

import argparse
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from nuscenes.nuscenes import NuScenes
from rclpy.node import Node
from sensor_msgs.msg import Image


class NuScenesCameraPublisher(Node):
    def __init__(self, nusc, tokens, topic, interval):
        super().__init__("nuscenes_camera_publisher")

        self.nusc = nusc
        self.tokens = tokens
        self.index = 0
        self.bridge = CvBridge()

        self.publisher = self.create_publisher(
            Image, topic, 10
        )

        self.timer = self.create_timer(
            interval, self.publish_next
        )

        self.get_logger().info(
            f"Ready to replay {len(tokens)} images to {topic}"
        )

    def publish_next(self):
        if self.index >= len(self.tokens):
            self.timer.cancel()
            self.get_logger().info(
                "All images published. Waiting 5 seconds before shutdown."
            )
            self.shutdown_timer = self.create_timer(
                5.0, self.finish_replay
            )
            return

        token = self.tokens[self.index]
        sample = self.nusc.get("sample", token)

        camera = self.nusc.get(
            "sample_data",
            sample["data"]["CAM_FRONT"],
        )

        image_path = (
            Path(self.nusc.dataroot) / camera["filename"]
        )

        image = cv2.imread(str(image_path))

        if image is None:
            self.get_logger().error(
                f"Could not load: {image_path}"
            )
            self.index += 1
            return

        msg = self.bridge.cv2_to_imgmsg(
            image, encoding="bgr8"
        )

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "CAM_FRONT"

        self.publisher.publish(msg)

        self.index += 1

        self.get_logger().info(
            f"Published {self.index}/{len(self.tokens)} "
            f"| sample={token[:8]}"
        )


    def finish_replay(self):
        self.shutdown_timer.cancel()
        self.get_logger().info(
            "Replay complete. Shutting down."
        )
        rclpy.try_shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot",
        default="/mnt/c/SJSU/datasets/nuScenes",
    )
    parser.add_argument(
        "--manifest",
        default="../configs/mini_val_sample_tokens.txt",
    )
    parser.add_argument(
        "--topic",
        default="/camera/image_corrupted",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    tokens = [
        token.strip()
        for token in Path(args.manifest).read_text().splitlines()
        if token.strip()
    ]

    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=args.dataroot,
        verbose=False,
    )

    rclpy.init()

    node = NuScenesCameraPublisher(
        nusc,
        tokens,
        args.topic,
        args.interval,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
