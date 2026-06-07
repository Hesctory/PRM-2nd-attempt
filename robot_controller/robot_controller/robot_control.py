#!/usr/bin/env python3
import heapq
import math
from enum import Enum

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String
from tf_transformations import euler_from_quaternion

# ── Reactive navigation constants ──────────────────────────────────────────────
STOP_DISTANCE      = 0.35
AVOID_DISTANCE     = 0.70
WALL_SPAN          = 30
CENTER_POS         = 0.5
FLAG_STOP_DISTANCE = 0.4
MAX_ADJUST         = 0.8
K_AREA_DISTANCE    = 20000.0
# Pinhole constant for height-based distance: K_H = real_height_m × focal_px.
# Calibrate: stop the robot at a known distance D from the flag, read h_bb pixels,
# then set K_FLAG_HEIGHT = D × h_bb.  84.0 is a placeholder from controle_robo.py.
K_FLAG_HEIGHT      = 139.0
MIN_FLAG_HEIGHT    = 5     # px — below this the height reading is too noisy to trust

# ── A* / grid constants ────────────────────────────────────────────────────────
GRID_SIZE               = 100    # must match robot_mapper
GRID_RESOLUTION         = 0.2   # m/cell — must match robot_mapper
OBSTACLE_INFLATE_RADIUS = 1     # cells (= 0.4 m safety margin)
WAYPOINT_TOLERANCE      = 0.20  # m
WAYPOINT_STRIDE         = 2
REPLAN_INTERVAL         = 50    # ticks between periodic re-plans (~5 s)
CAMERA_FOV              = 1.57  # rad (90°) horizontal FOV
# A* goal placed this far from the flag pole — safely outside the inflation zone
# and matching the visual-servo handoff distance.
ASTAR_GOAL_OFFSET = FLAG_STOP_DISTANCE + 0.3   # 0.7 m


class State(str, Enum):
    EXPLORING           = 'EXPLORING'
    FLAG_DETECTED       = 'FLAG_DETECTED'
    NAVIGATING_TO_FLAG  = 'NAVIGATING_TO_FLAG'
    NEAR_FLAG           = 'NEAR_FLAG'


class RobotControl(Node):

    def __init__(self):
        super().__init__('robot_control')

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub    = self.create_publisher(Path,  '/astar_path', 10)

        self.create_subscription(LaserScan,     '/scan',                 self.scan_callback,  10)
        self.create_subscription(Imu,           '/imu',                  self.imu_callback,   10)
        self.create_subscription(Odometry,      '/odom',                 self.odom_callback,  10)
        self.create_subscription(Pose,          '/model/prm_robot/pose', self.pose_callback,  10)
        self.create_subscription(String,        '/flag_detected',        self.flag_callback,  10)
        self.create_subscription(OccupancyGrid, '/rc_grid_map',          self._map_callback,  10)

        self.timer = self.create_timer(0.1, self.move_robot)

        # ── Reactive state ──────────────────────────────────────────────────────
        self.state          = State.EXPLORING
        self.obstacle_ahead = False
        self.lidar_data     = []
        self.flag_pos       = 0.5
        self.flag_area      = 0.0
        self.flag_height    = 0     # bounding-box height in pixels

        # ── Relative heading (wall-follow only) ────────────────────────────────
        self.yaw         = 0.0
        self.initial_yaw = None

        # ── Ground-truth pose (A* navigation) ──────────────────────────────────
        self.robot_x       = 0.0
        self.robot_y       = 0.0
        self.robot_heading = 0.0

        # ── Occupancy grid ──────────────────────────────────────────────────────
        self.grid_map          = None
        self._replan_requested = False

        # ── A* navigation ───────────────────────────────────────────────────────
        self.flag_world_pos  = None   # (x, y) in world frame, continuously refined
        self.waypoints       = []
        self.waypoint_idx    = 0
        self._replan_ticks   = 0
        self._last_flag_time    = None   # rclpy.Time of last flag detection message
        self._last_turn_dir     = -0.5   # held turn direction for _open_side_turn hysteresis
        self._flag_detect_ticks = 0      # ticks spent stopped in FLAG_DETECTED waiting for map

    # ── Sensor callbacks ───────────────────────────────────────────────────────

    def scan_callback(self, msg):
        self.lidar_data = list(msg.ranges)
        front   = msg.ranges[0:30] + msg.ranges[330:360]
        valid   = [d for d in front if d != float('inf') and d > 0.0]
        self.obstacle_ahead = bool(valid) and min(valid) < AVOID_DISTANCE

    def imu_callback(self, msg):
        q = msg.orientation
        _, _, yaw_read = euler_from_quaternion([q.x, q.y, q.z, q.w])
        if self.initial_yaw is None:
            self.initial_yaw = yaw_read
        self.yaw = self._normalize_angle(yaw_read - self.initial_yaw)

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        _, _, yaw_read = euler_from_quaternion([q.x, q.y, q.z, q.w])
        if self.initial_yaw is None:
            self.initial_yaw = yaw_read
        self.yaw = self._normalize_angle(yaw_read - self.initial_yaw)

    def pose_callback(self, msg: Pose):
        self.robot_x = msg.position.x
        self.robot_y = msg.position.y
        q = msg.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_heading = math.atan2(siny, cosy)

    def flag_callback(self, msg):
        if msg.data.startswith('detected:'):
            try:
                parts            = msg.data.split(':')
                self.flag_pos    = float(parts[1])
                self.flag_area   = float(parts[2]) if len(parts) >= 3 else 0.0
                self.flag_height = int(float(parts[3])) if len(parts) >= 4 else 0
                self._last_flag_time = self.get_clock().now()
                if self.state == State.EXPLORING:
                    self.state = State.FLAG_DETECTED
            except Exception as e:
                self.get_logger().warn(f'Invalid format on /flag_detected: {e}')

    def _map_callback(self, msg: OccupancyGrid):
        raw = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)

        # Derive flag world position from cells marked 50 by the mapper
        flag_cells = np.argwhere(raw == 50)
        if len(flag_cells) > 0:
            gy_m = float(np.mean(flag_cells[:, 0]))
            gx_m = float(np.mean(flag_cells[:, 1]))
            fx, fy = self._grid_to_world(int(round(gx_m)), int(round(gy_m)))
            self.flag_world_pos = (fx, fy)

        # Inflate both obstacle (100) and flag (50) cells so A* treats them as blocked
        self.grid_map = self._inflate_obstacles(raw, OBSTACLE_INFLATE_RADIUS)

        if self._replan_requested and self.flag_world_pos is not None:
            self._replan_requested = False
            self._plan_path_to_flag()

    # ── State machine ──────────────────────────────────────────────────────────

    def move_robot(self):
        twist = Twist()

        if self.state == State.EXPLORING:
            self.get_logger().info('State: EXPLORING')
            twist = self._wall_follow()

        elif self.state == State.FLAG_DETECTED:
            self._flag_detect_ticks += 1
            if self.flag_world_pos is not None:
                # Map has a confirmed flag cell — position is reliable, start navigating
                self.get_logger().info(
                    f'FLAG_DETECTED: flag mapped after {self._flag_detect_ticks} ticks — starting navigation'
                )
                self._flag_detect_ticks = 0
                self.state = State.NAVIGATING_TO_FLAG
                self._start_flag_navigation()
            elif self._flag_detect_ticks > 30:
                # 3 s timeout — map never got a clean cell, fall back to EXPLORING
                self.get_logger().warn('FLAG_DETECTED: map timeout — returning to EXPLORING')
                self._flag_detect_ticks = 0
                self.state = State.EXPLORING
            # twist stays zero: robot is stopped while waiting

        elif self.state == State.NAVIGATING_TO_FLAG:
            twist = self._navigate_to_flag()

        elif self.state == State.NEAR_FLAG:
            self.get_logger().info('State: NEAR_FLAG — stopped.')

        self.cmd_vel_pub.publish(twist)

    # ── Flag navigation ────────────────────────────────────────────────────────

    def _start_flag_navigation(self):
        """Called once when entering NAVIGATING_TO_FLAG."""
        self.waypoints     = []
        self.waypoint_idx  = 0
        self._replan_ticks = 0
        if self.flag_world_pos is not None:
            # Map already has flag position from cell-50 detection — use it directly
            self._plan_path_to_flag()
            if not self.waypoints:
                self._replan_requested = True
        else:
            # Map hasn't seen the flag yet — wait for next map update to trigger replan
            self._replan_requested = True

    def _navigate_to_flag(self) -> Twist:
        self.get_logger().info('State: NAVIGATING_TO_FLAG')

        distance = self._flag_distance()
        if distance < FLAG_STOP_DISTANCE:
            self.state = State.NEAR_FLAG
            self.get_logger().info(
                f'Flag reached at {distance:.2f} m — mission complete.'
            )
            return Twist()

        # Refine flag world position from camera bounding-box height while flag is visible.
        # Pinhole model: dist = K / pixel_height — flag-specific, not confused by walls.
        if self._flag_visible() and self.flag_height >= MIN_FLAG_HEIGHT:
            angular_err = (self.flag_pos - CENTER_POS) * CAMERA_FOV
            bearing     = self.robot_heading - angular_err
            dist        = K_FLAG_HEIGHT / self.flag_height
            if math.isfinite(dist) and dist > 0.2:
                self.flag_world_pos = (
                    self.robot_x + dist * math.cos(bearing),
                    self.robot_y + dist * math.sin(bearing),
                )

        # Periodic re-plan while actively following waypoints
        if self.waypoints and self.flag_world_pos is not None:
            self._replan_ticks += 1
            if self._replan_ticks >= REPLAN_INTERVAL:
                self._replan_ticks = 0
                self.get_logger().info('Periodic A* re-plan')
                self._plan_path_to_flag()

        # Priority 1 — A* waypoints
        if self.waypoints and self.waypoint_idx < len(self.waypoints):
            return self._follow_waypoints_twist()

        # Priority 2 — flag visible: visual servo handles close-range approach
        if self._flag_visible():
            return self._visual_servo_twist()

        # Priority 3 — flag lost and no waypoints: replan if possible, else explore
        if self.flag_world_pos is not None:
            self.get_logger().warn('Flag not visible and no waypoints — replanning A*')
            self._plan_path_to_flag()
            if self.waypoints:
                return Twist()

        self.get_logger().warn('Flag lost, no path — returning to EXPLORING')
        self.state = State.EXPLORING
        self.flag_world_pos = None
        return Twist()

    def _follow_waypoints_twist(self) -> Twist:
        wx, wy     = self.waypoints[self.waypoint_idx]
        dx         = wx - self.robot_x
        dy         = wy - self.robot_y
        dist_to_wp = math.sqrt(dx * dx + dy * dy)

        if dist_to_wp < WAYPOINT_TOLERANCE:
            self.waypoint_idx += 1
            if self.waypoint_idx >= len(self.waypoints):
                self.get_logger().info('A* waypoints exhausted')
                # If flag is not visible from here, the A* goal was wrong — replan
                if not self._flag_visible() and self.flag_world_pos is not None:
                    self.get_logger().warn('Flag not visible at goal — replanning A*')
                    self._plan_path_to_flag()
                else:
                    self.waypoints = []
            return Twist()   # one-tick pause while advancing index

        # Interrupt A* when an obstacle is within AVOID_DISTANCE (0.7 m) and we are
        # heading toward it — proactive, matching controle_robo's OBSTACLE_FRONT threshold.
        front_vals = [self.lidar_data[i] for i in list(range(0, 30)) + list(range(330, 360))
                      if self.lidar_data[i] not in (float('inf'), float('-inf')) and self.lidar_data[i] > 0.0]
        if front_vals and min(front_vals) < AVOID_DISTANCE:
            target_bearing = math.atan2(dy, dx)
            heading_err    = abs(self._angle_diff(target_bearing, self.robot_heading))
            if heading_err < 0.5:
                self.get_logger().info('Emergency obstacle on waypoint path — requesting A* re-plan')
                self._replan_requested = True
                twist           = Twist()
                twist.angular.z = self._open_side_turn()
                return twist

        target_bearing  = math.atan2(dy, dx)
        heading_err     = self._angle_diff(target_bearing, self.robot_heading)
        twist           = Twist()
        twist.angular.z = max(-0.5, min(0.5, 2.0 * heading_err))
        twist.linear.x  = 0.3 if abs(heading_err) < 0.3 else 0.0
        return twist

    def _visual_servo_twist(self) -> Twist:
        """Proportional visual servoing — fallback when A* is unavailable or at close range."""
        twist = Twist()
        error = self.flag_pos - CENTER_POS
        if self.obstacle_ahead:
            twist.linear.x  = 0.0
            twist.angular.z = self._open_side_turn()
        else:
            twist.angular.z = -error * 1.5
            # Move forward proportionally: full speed when centred, slows as error grows,
            # zero only when flag is more than 60% off-centre (hard to correct while moving).
            twist.linear.x  = max(0.0, 0.3 * (1.0 - abs(error) / 0.3)) if abs(error) < 0.3 else 0.0
        return twist

    # ── Reactive helpers ───────────────────────────────────────────────────────

    def _normalize_angle(self, a):
        while a > math.pi:  a -= 2 * math.pi
        while a < -math.pi: a += 2 * math.pi
        return a

    def _flag_distance(self):
        # Height-based: robust to viewing angle and doesn't blow up at long range.
        if self.flag_height >= MIN_FLAG_HEIGHT:
            return K_FLAG_HEIGHT / self.flag_height
        # LiDAR fallback when height unavailable.
        # Index: 0 = front, CCW positive; (0.5-pos)*90 maps camera centre→index 0.
        if not self.lidar_data or len(self.lidar_data) < 360:
            return float('inf')
        idx      = int((0.5 - self.flag_pos) * 90) % 360
        neighbors = [
            d for i in range(-20, 21)
            if 0 <= (idx + i) % 360 < len(self.lidar_data)
            and not (d := self.lidar_data[(idx + i) % 360]) in (float('inf'), float('-inf'))
            and d > 0.0
        ]
        return min(neighbors) if neighbors else float('inf')

    def _sector_min(self, indices):
        vals = [
            self.lidar_data[i] for i in indices
            if self.lidar_data[i] not in (float('inf'), float('-inf'))
            and self.lidar_data[i] > 0.0
        ]
        return min(vals) if vals else float('inf')

    def _angular_span_in_front(self):
        indices = list(range(330, 360)) + list(range(0, 30))
        return sum(
            1 for i in indices
            if self.lidar_data[i] not in (float('inf'), float('-inf'))
            and 0.0 < self.lidar_data[i] < AVOID_DISTANCE
        )

    def _flag_visible(self) -> bool:
        if self._last_flag_time is None:
            return False
        return (self.get_clock().now() - self._last_flag_time).nanoseconds < 500_000_000

    def _open_side_turn(self):
        if not self.lidar_data or len(self.lidar_data) < 360:
            return self._last_turn_dir
        left  = [d for d in self.lidar_data[45:90]   if d != float('inf') and d > 0.0]
        right = [d for d in self.lidar_data[270:315] if d != float('inf') and d > 0.0]
        mean_left  = sum(left)  / len(left)  if left  else 0.0
        mean_right = sum(right) / len(right) if right else 0.0
        diff = mean_left - mean_right
        if abs(diff) > 0.15:
            self._last_turn_dir = +0.5 if diff > 0 else -0.5
        return self._last_turn_dir

    def _wall_follow(self):
        if not self.lidar_data or len(self.lidar_data) < 360:
            return Twist()

        dist_front = self._sector_min(list(range(0, 30)) + list(range(330, 360)))
        dist_left  = self._sector_min(list(range(60, 120)))
        span       = self._angular_span_in_front()

        error  = 0.3 - dist_left
        adjust = max(min(-error, MAX_ADJUST), -MAX_ADJUST)

        twist = Twist()
        if dist_front < AVOID_DISTANCE:
            if span >= WALL_SPAN:
                twist.linear.x  = 0.0
                twist.angular.z = -0.5
            else:
                t = (dist_front - STOP_DISTANCE) / (AVOID_DISTANCE - STOP_DISTANCE)
                t = max(0.0, min(1.0, t))
                twist.linear.x  = 0.3 * t
                twist.angular.z = -0.5 * (1.0 - t)
        elif dist_left < 1.5:
            twist.linear.x  = 0.5
            twist.angular.z = adjust
        else:
            twist.linear.x  = 0.25
            twist.angular.z = 0.5
        return twist

    # ── Grid / A* helpers ──────────────────────────────────────────────────────

    def _angle_diff(self, target: float, current: float) -> float:
        diff = target - current
        return (diff + math.pi) % (2 * math.pi) - math.pi

    def _world_to_grid(self, wx: float, wy: float):
        half = GRID_SIZE * GRID_RESOLUTION / 2.0
        gx   = int((wx + half) / GRID_RESOLUTION)
        gy   = int((wy + half) / GRID_RESOLUTION)
        return gx, gy

    def _grid_to_world(self, gx: int, gy: int):
        half = GRID_SIZE * GRID_RESOLUTION / 2.0
        wx   = gx * GRID_RESOLUTION - half + GRID_RESOLUTION / 2.0
        wy   = gy * GRID_RESOLUTION - half + GRID_RESOLUTION / 2.0
        return wx, wy

    def _grid_in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE

    def _inflate_obstacles(self, raw: np.ndarray, radius: int) -> np.ndarray:
        result = raw.copy()
        # Inflate both hard obstacles (100) and flag cells (50) — robot must not enter either
        for gy_o, gx_o in np.argwhere((raw == 100) | (raw == 50)):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = int(gy_o) + dy, int(gx_o) + dx
                    if 0 <= ny < raw.shape[0] and 0 <= nx < raw.shape[1]:
                        result[ny, nx] = 100
        return result

    def _plan_path_to_flag(self):
        if self.grid_map is None or self.flag_world_pos is None:
            return

        flag_x, flag_y = self.flag_world_pos
        dist_to_flag   = math.sqrt(
            (flag_x - self.robot_x) ** 2 + (flag_y - self.robot_y) ** 2
        )
        if dist_to_flag <= ASTAR_GOAL_OFFSET:
            # Robot is already inside the A* goal zone — visual servo handles it
            self.waypoints = []
            return

        start   = self._world_to_grid(self.robot_x, self.robot_y)
        bearing = math.atan2(flag_y - self.robot_y, flag_x - self.robot_x)
        goal_wx = flag_x - ASTAR_GOAL_OFFSET * math.cos(bearing)
        goal_wy = flag_y - ASTAR_GOAL_OFFSET * math.sin(bearing)
        goal    = self._world_to_grid(goal_wx, goal_wy)

        # If the goal cell is inside the obstacle inflation zone, search for the
        # nearest free neighbour (can happen when flag is close to a wall)
        g_gx, g_gy = goal
        if self._grid_in_bounds(g_gx, g_gy) and self.grid_map[g_gy, g_gx] == 100:
            found = False
            for r in range(1, 6):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        nx, ny = g_gx + dx, g_gy + dy
                        if self._grid_in_bounds(nx, ny) and self.grid_map[ny, nx] != 100:
                            goal  = (nx, ny)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                self.get_logger().warn('A* goal blocked — no free cell near flag; falling back to visual servo')
                self.waypoints = []
                return

        path_cells = self._astar(start, goal)
        if not path_cells:
            self.get_logger().warn('A* found no path — falling back to visual servo')
            self.waypoints = []
            return

        sampled = path_cells[::WAYPOINT_STRIDE]
        if sampled[-1] != path_cells[-1]:
            sampled.append(path_cells[-1])
        self.waypoints    = [self._grid_to_world(gx, gy) for gx, gy in sampled]
        self.waypoint_idx = 0
        self.get_logger().info(
            f'A* path: {len(path_cells)} cells → {len(self.waypoints)} waypoints'
        )

        now      = self.get_clock().now().to_msg()
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp    = now
        for gx, gy in path_cells:
            wx, wy = self._grid_to_world(gx, gy)
            ps     = PoseStamped()
            ps.header.frame_id    = 'map'
            ps.header.stamp       = now
            ps.pose.position.x    = wx
            ps.pose.position.y    = wy
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.path_pub.publish(path_msg)

    def _astar(self, start: tuple, goal: tuple) -> list:
        s_gx, s_gy = start
        g_gx, g_gy = goal
        if not self._grid_in_bounds(s_gx, s_gy) or not self._grid_in_bounds(g_gx, g_gy):
            return []

        open_set = []
        heapq.heappush(open_set, (0.0, s_gx, s_gy))
        came_from = {}
        g_score   = {(s_gx, s_gy): 0.0}

        def h(gx, gy):
            return math.sqrt((gx - g_gx) ** 2 + (gy - g_gy) ** 2)

        while open_set:
            _, cx, cy = heapq.heappop(open_set)
            if cx == g_gx and cy == g_gy:
                path = []
                pos  = (cx, cy)
                while pos in came_from:
                    path.append(pos)
                    pos = came_from[pos]
                path.append((s_gx, s_gy))
                path.reverse()
                return path
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                           (0,  1),  (1, -1), (1,  0), (1,  1)]:
                nx, ny = cx + dx, cy + dy
                if not self._grid_in_bounds(nx, ny):
                    continue
                if self.grid_map[ny, nx] == 100:
                    continue
                step        = math.sqrt(2) if dx != 0 and dy != 0 else 1.0
                penalty     = 0.3 if self.grid_map[ny, nx] < 0 else 0.0
                tentative_g = g_score[(cx, cy)] + step + penalty
                if tentative_g < g_score.get((nx, ny), float('inf')):
                    came_from[(nx, ny)] = (cx, cy)
                    g_score[(nx, ny)]   = tentative_g
                    heapq.heappush(open_set, (tentative_g + h(nx, ny), nx, ny))

        return []


def main(args=None):
    rclpy.init(args=args)
    node = RobotControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
