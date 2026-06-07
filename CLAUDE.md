# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace Overview

ROS 2 Humble workspace (`~/ros2_testing`) for the SSC0712 "Programação de Robôs Móveis" course at USP São Carlos. The `src/` directory contains three packages:

- **`prm`** — Final project (Group 5, last year): complete flag-collection mission with gripper, full state machine, wall-following, and base return. Reference implementation.
- **`prm_2026`** — Course template from the professor: simpler base node for student exercises. Used to launch the simulation and spawn the robot.
- **`robot_controller`** — My own package, based on `prm_2026` structure, implementing wall-following + flag detection mission.

## Build and Run

All colcon commands are run from the workspace root (`~/ros2_testing`), not from `src/`:

```bash
# Build a single package
colcon build --symlink-install --packages-select robot_controller

# Source after building (required in every new terminal)
source install/local_setup.bash
```

### Running the Simulation (three terminals)

**Terminal 1 — start Gazebo world:**
```bash
ros2 launch prm_2026 inicia_simulacao.launch.py world:=arena_cilindros.sdf
# Other worlds: empty_arena.sdf, arena_paredes.sdf
```

**Terminal 2 — spawn robot (after Gazebo is ready):**
```bash
ros2 launch prm_2026 carrega_robo.launch.py
```

**Terminal 3 — start mission:**
```bash
ros2 launch robot_controller start_mission.launch.py
```

**Manual keyboard control (alternative to auto mission):**
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## Architecture

### ROS Topic Map

| Topic | Type | Publisher → Subscriber |
|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | Gazebo → robot_control |
| `/imu` | `sensor_msgs/Imu` | Gazebo → robot_control |
| `/odom` | `nav_msgs/Odometry` | relay (from diff drive) → robot_control |
| `/model/prm_robot/pose` | `geometry_msgs/Pose` | Gazebo ground truth → robo_mapper |
| `/robot_cam/colored_map` | `sensor_msgs/Image` | Gazebo camera (BGR) → flag_detector |
| `/bandeira_detectada` | `std_msgs/String` | flag_detector → robot_control |
| `/cmd_vel` | `geometry_msgs/Twist` | robot_control → relay → diff drive |

**Topic relay pattern**: `/cmd_vel` is relayed to `/diff_drive_base_controller/cmd_vel_unstamped`; `/diff_drive_base_controller/odom` is relayed to `/odom`. Wired in `prm_2026/launch/carrega_robo.launch.py`.

### Camera

- Topic: `/robot_cam/colored_map` (BGR image)
- Horizontal FOV: **1.57 rad (90°)**, so half-FOV = 45°
- `bandeira_pos` is normalized centroid x ∈ [0, 1]: 0 = left edge, 0.5 = center, 1 = right edge
- Mapping to LiDAR index: `center_idx = int((0.5 - bandeira_pos) * 90) % 360`

### LiDAR

- Topic: `/scan`, 360 indices
- Index 0 = directly ahead, increases counter-clockwise
- Left side: indices 60–120; Front: 0–30 + 330–360; Right: 270–315

### Flag Detection Protocol

`flag_detector` publishes to `/bandeira_detectada` as:
```
detected:<pos_norm>:<area>
```
- `pos_norm` ∈ [0, 1]: normalized horizontal centroid position
- `area`: pixel area of detected blob (filtered < 200 px are ignored)
- Color searched: BGR lower `(217, 63, 0)` / upper `(237, 83, 10)` — centered on `(227, 73, 0)`
- **Unverified**: `prm_2026` template originally used `(171, 242, 0)` — flag color may differ in 2026 arena

### robot_controller State Machine (`robot_control.py`)

| State | Behaviour | Transition |
|---|---|---|
| `EXPLORANDO` | Left-wall following with span-based obstacle classification | → `BANDEIRA_DETECTADA` when flag detected |
| `BANDEIRA_DETECTADA` | One-tick transition | → `NAVIGANDO_PARA_BANDEIRA` immediately |
| `NAVIGANDO_PARA_BANDEIRA` | Visual servoing toward flag centroid; obstacle avoidance toward open side | → `PERTO_DA_BANDEIRA` when distance < 1.5 m |
| `PERTO_DA_BANDEIRA` | Full stop — mission complete | terminal |

**Key constants:**

| Constant | Value | Purpose |
|---|---|---|
| `DISTANCIA_PARAR` | 0.35 m | Emergency stop + turn (last resort) |
| `DISTANCIA_DESVIAR` | 0.70 m | Alert zone — start steering before obstacle |
| `SPAN_PAREDE` | 30 indices | Min front LiDAR indices to classify as wall vs. obstacle |
| `DISTANCIA_PARADA` | 1.5 m | Stop distance from flag |
| `K_DISTANCIA_AREA` | 20000.0 | Area-to-distance conversion: `dist = K / area` |

**Wall vs. obstacle classification** (`_angular_span_in_front`):
- Counts how many of the 60 front indices (330–360 + 0–30) read within `DISTANCIA_DESVIAR`
- ≥ 30 indices → wide arc → wall/corner → turn right (same as `prm`)
- < 30 indices → narrow arc → isolated obstacle → proportional slow + steer

**Flag distance estimation** (`_distancia_bandeira`):
- Primary: `K_DISTANCIA_AREA / area` when area > 200 px
- Fallback: minimum LiDAR reading in a ±20-index window around `int(bandeira_pos * 359)`

### prm Package State Machine (reference)

1. **EXPLORANDO** — left-wall following
2. **BANDEIRA_DETECTADA** — one-tick transition
3. **NAVIGANDO_PARA_BANDEIRA** — visual servoing toward flag
4. **POSICIONANDO_PARA_COLETA** — closes distance, opens gripper at 1.5 m
5. **RETORNANDO_PARA_BASE** — spin, wall-follow, direct nav to `(x=-8.0, y=-0.5)`
6. **FINALIZADO** — stop

## Known Issues

1. **Obstacle-flag navigation loop** *(unresolved)* — when an obstacle is near the path to the flag, `obstaculo_a_frente` (±30° front cone, threshold 0.70 m) triggers during `NAVIGANDO_PARA_BANDEIRA`. The dodge maneuver turns the robot, clearing the front cone, and navigation resumes immediately — looping. A `CIRCUMNAVIGATING` state with wall-follow was attempted twice but the exit condition was unreliable because `bandeira_pos` shifts as the robot turns, causing the LiDAR path-clear check to pass prematurely. Proposed fix (not yet implemented): minimum dwell time in `CIRCUMNAVIGATING` + `not obstaculo_a_frente` as exit condition.

2. **No tip-over recovery** *(reverted)* — IMU-based recovery (detect roll > 0.30 rad or pitch > 0.35 rad → back up → turn toward open side) was implemented and reverted at user request. Robot can be permanently tipped by obstacles.

3. **Wall-follow corner instability** *(partially addressed)* — the 0.70 m alert zone was interfering with normal wall-following at corners. Span-based classification partially fixes this but edge cases remain.

4. **Flag color unverified** — detector uses BGR `(227±10, 73±10, 0±10)` from last year's `prm` package. The `prm_2026` template used `(171, 242, 0)`. If the 2026 arena uses a different flag color, detection will silently fail.
