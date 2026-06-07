#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster

CELL_FLAG     = 50   # grid value for the flag pole
CELL_OBSTACLE = 100
FLAG_RAY_HALF_WIDTH = 2   # indices on each side of the flag ray to mark as flag


class RobotMapper(Node):

    # Grid covers 20 x 20 m centred at world origin — fits the full arena
    GRID_SIZE  = 100    # cells
    RESOLUTION = 0.2    # metres per cell
    FLAG_DETECT_TIMEOUT = 1.0   # seconds — clear flag_lidar_idx if detection is stale

    def __init__(self):
        super().__init__('robot_mapper')

        self.create_subscription(LaserScan, '/scan',                 self.scan_callback,  10)
        self.create_subscription(Pose,      '/model/prm_robot/pose', self.pose_callback,  10)
        self.create_subscription(String,    '/flag_detected',        self._flag_callback, 10)

        self.map_pub = self.create_publisher(OccupancyGrid, '/rc_grid_map', 10)
        self.timer   = self.create_timer(0.5, self.publish_occupancy_grid)

        # Robot pose (world frame)
        self.x       = 0.0
        self.y       = 0.0
        self.heading = 0.0

        # -1 = unknown, 0 = free, 50 = flag, 100 = occupied
        self.grid_map = -np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # Flag detection state
        self.flag_lidar_idx      = None   # LiDAR index pointing at the flag
        self._last_flag_time     = None   # rclpy.Time of last /flag_detected message

        # Broadcast static map → odom_gt transform so RViz can display the grid
        self.tf_static = StaticTransformBroadcaster(self)
        tf             = TransformStamped()
        tf.header.stamp       = self.get_clock().now().to_msg()
        tf.header.frame_id    = 'map'
        tf.child_frame_id     = 'odom_gt'
        tf.transform.rotation.w = 1.0
        self.tf_static.sendTransform(tf)

    # ── Sensor callbacks ───────────────────────────────────────────────────────

    def _flag_callback(self, msg: String):
        if msg.data.startswith('detected:'):
            try:
                flag_pos = float(msg.data.split(':')[1])
                # Camera: 0=left edge, 0.5=centre, 1=right edge
                # LiDAR:  0=front, positive=CCW (left)
                self.flag_lidar_idx  = int((0.5 - flag_pos) * 90) % 360
                self._last_flag_time = self.get_clock().now()
            except Exception:
                pass

    def _flag_ray_active(self) -> bool:
        if self._last_flag_time is None or self.flag_lidar_idx is None:
            return False
        elapsed = (self.get_clock().now() - self._last_flag_time).nanoseconds / 1e9
        return elapsed < self.FLAG_DETECT_TIMEOUT

    def _is_flag_ray(self, idx: int, n: int) -> bool:
        if not self._flag_ray_active():
            return False
        d = abs(idx - self.flag_lidar_idx)
        return min(d, n - d) <= FLAG_RAY_HALF_WIDTH

    def pose_callback(self, msg: Pose):
        self.x = msg.position.x
        self.y = msg.position.y

        q          = msg.orientation
        siny_cosp  = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp  = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.heading = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg: LaserScan):
        robot_gx, robot_gy = self.world_to_grid(self.x, self.y)
        if not self._in_bounds(robot_gx, robot_gy):
            return

        n     = len(msg.ranges)
        angle = msg.angle_min
        for i, r in enumerate(msg.ranges):
            in_range = math.isfinite(r) and msg.range_min < r < msg.range_max
            if in_range:
                world_angle = self.heading + angle
                ex          = self.x + r * math.cos(world_angle)
                ey          = self.y + r * math.sin(world_angle)
                end_gx, end_gy = self.world_to_grid(ex, ey)
                is_flag = self._is_flag_ray(i, n)
                self._bresenham(robot_gx, robot_gy, end_gx, end_gy,
                                mark_end=True, is_flag=is_flag)
            angle += msg.angle_increment

    # ── Publisher ──────────────────────────────────────────────────────────────

    def publish_occupancy_grid(self):
        msg                    = OccupancyGrid()
        msg.header.stamp       = self.get_clock().now().to_msg()
        msg.header.frame_id    = 'map'
        msg.info.resolution    = self.RESOLUTION
        msg.info.width         = self.GRID_SIZE
        msg.info.height        = self.GRID_SIZE

        origin                   = Pose()
        half                     = (self.GRID_SIZE * self.RESOLUTION) / 2.0
        origin.position.x        = -half
        origin.position.y        = -half
        origin.orientation.w     = 1.0
        msg.info.origin          = origin

        msg.data = self.grid_map.flatten().tolist()
        self.map_pub.publish(msg)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def world_to_grid(self, x: float, y: float):
        half = self.GRID_SIZE * self.RESOLUTION / 2.0
        gx   = int((x + half) / self.RESOLUTION)
        gy   = int((y + half) / self.RESOLUTION)
        return gx, gy

    def _in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.GRID_SIZE and 0 <= gy < self.GRID_SIZE

    def _bresenham(self, x0: int, y0: int, x1: int, y1: int,
                   mark_end: bool, is_flag: bool = False):
        """Walk cells from (x0,y0) to (x1,y1), marking free; end marked as flag or obstacle."""
        dx  = abs(x1 - x0)
        dy  = abs(y1 - y0)
        sx  = 1 if x1 > x0 else -1
        sy  = 1 if y1 > y0 else -1
        err = dx - dy

        while True:
            at_end = (x0 == x1 and y0 == y1)

            if self._in_bounds(x0, y0):
                if at_end and mark_end:
                    # Flag cell: write 50 only if not already confirmed obstacle from other rays
                    if is_flag:
                        self.grid_map[y0, x0] = CELL_FLAG
                    else:
                        # Obstacle ray: overwrite free/unknown, but don't clobber a flag marker
                        if self.grid_map[y0, x0] != CELL_FLAG:
                            self.grid_map[y0, x0] = CELL_OBSTACLE
                elif not at_end and self.grid_map[y0, x0] not in (CELL_OBSTACLE, CELL_FLAG):
                    self.grid_map[y0, x0] = 0

            if at_end:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0  += sx
            if e2 < dx:
                err += dx
                y0  += sy


def main(args=None):
    rclpy.init(args=args)
    node = RobotMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
