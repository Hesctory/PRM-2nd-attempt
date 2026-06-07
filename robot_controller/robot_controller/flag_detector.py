#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge
import cv2
import numpy as np


class FlagDetector(Node):

    def __init__(self):
        super().__init__('flag_detector')

        self.bridge = CvBridge()

        self.create_subscription(
            Image,
            '/robot_cam/colored_map',
            self.camera_callback,
            10
        )

        self.pub_detected = self.create_publisher(String, '/flag_detected', 10)

        self.get_logger().info('Flag detector started.')

    def camera_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return

        h, w, _ = frame.shape

        # Flag color in BGR: (B=227, G=73, R=0) — adjust if arena uses a different color
        lower = np.array([217, 63, 0])
        upper = np.array([237, 83, 10])
        mask = cv2.inRange(frame, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 200:
                M = cv2.moments(c)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    pos_norm = cx / w
                    _, y_bb, _, h_bb = cv2.boundingRect(c)
                    # Full-mask height: span from topmost to bottommost flag pixel,
                    # unaffected by the gun barrel splitting the contour in two.
                    ys = np.where(mask > 0)[0]
                    h_mask = int(ys.max() - ys.min()) + 1 if len(ys) > 0 else h_bb
                    self.pub_detected.publish(
                        String(data=f'detected:{pos_norm:.2f}:{area:.0f}:{h_bb}:{h_mask}')
                    )


def main(args=None):
    rclpy.init(args=args)
    node = FlagDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
