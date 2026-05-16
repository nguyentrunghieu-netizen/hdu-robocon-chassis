#!/usr/bin/env python3
"""
Camera LiDAR Web V7 — entry point
==================================
Cac chuc nang da duoc tach ra cac module rieng:
  config.py        - tat ca hang so cau hinh
  controllers.py   - Kalman1D, PID, SlewRateLimiter, AxisPD, AlignmentAutoTuner
  detector.py      - FaceDetector (YOLO)
  serial_manager.py- SerialManager, port utilities
  point_cloud.py   - PointCloudFilter, PersistentMap
  slam.py          - PoseEKF, LidarPoseManager
  planner.py       - AStarPlanner, WaypointStore
  app_state.py     - SharedAppState
  web_server.py    - WebHandler, ThreadedHTTPServer, update_web_frame, run_calibration_cli
  web_ui.py        - draw_dashboard, HTML_PAGE (da co truoc)
  kissicp.py       - LiDAR ICP core (da co truoc)
"""

import argparse
import math
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

from config import (
    # camera
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    CAMERA_FOCAL_PX, REAL_FACE_HEIGHT_M,
    TARGET_DISTANCE_M, DIST_DEADBAND_M,
    TARGET_BBOX_H,
    CENTER_DEADBAND,
    FORWARD_ALIGN_LIMIT, FORWARD_ALIGN_MIN_GAIN,
    # velocity limits
    VX_MAX, VY_MAX, OMEGA_MAX,
    VY_ALIGN_MAX, OMEGA_ALIGN_MAX,
    CLOSE_BACKOFF_GAIN, CLOSE_BACKOFF_MAX,
    # slew
    VX_SLEW_MPS_S, VY_SLEW_MPS_S, OMEGA_SLEW_RAD_S2,
    # alignment auto-tune thresholds
    ALIGN_HOLD_ENTER, ALIGN_HOLD_EXIT,
    ALIGN_STRAFE_ENTER, ALIGN_STRAFE_EXIT, ALIGN_RATE_HOLD,
    # PID / PD gains
    TURN_KP, TURN_KD, STRAFE_KP, STRAFE_KD,
    HOLD_KP, HOLD_KD, HOLD_OMEGA_MAX,
    TRACK_USE_IMU_HEADING, TRACK_IMU_STALE_S,
    TRACK_HEADING_KP, TRACK_HEADING_KI, TRACK_HEADING_KD,
    TRACK_HEADING_MAX, TRACK_HEADING_TARGET_RATE_RAD_S,
    VX_KP, VX_KI, VX_KD, VX_FF_GAIN,
    # control rate / history
    CONTROL_RATE, HIST_LEN,
    KALMAN_Q, KALMAN_R,
    # lost-target timing
    LOST_COAST_TIME, LOST_SOFT_STOP, LOST_HARD_RESET, LOST_TIMEOUT,
    # serial / lidar
    LIDAR_PORT, LIDAR_BAUDRATE, LIDAR_BAUDRATES,
    ARDUINO_PORT,
    MAP_FILE, MAP_AUTOSAVE_PERIOD_S,
    WAYPOINTS_FILE,
    # navigation
    NAV_LIDAR_STALE_S, NAV_GOAL_TOLERANCE_M,
    NAV_HEADING_TOLERANCE_DEG, NAV_PATH_HEADING_TOL_DEG, NAV_PATH_HEADING_HYST_DEG,
    NAV_VX_MAX, NAV_OMEGA_MAX, NAV_VX_MIN_MOVE, NAV_OMEGA_MIN_MOVE,
    NAV_HEADING_KP, NAV_TRANSLATION_KP,
    NAV_DECEL_DIST_M, NAV_FINAL_APPROACH_M,
    NAV_VX_RATE_LIMIT, NAV_OMEGA_RATE_LIMIT,
    # A*
    ASTAR_RESOLUTION_M, ASTAR_INFLATE_M,
    ASTAR_SUBWP_TOL_M, ASTAR_REPLAN_PERIOD_S,
    # obstacle avoidance
    OBS_SAFE_DIST_M, OBS_SLOW_DIST_M, OBS_CONE_HALF_DEG, OBS_MIN_POINTS,
)
from controllers import (
    Kalman1D, PID, SlewRateLimiter, AxisPD, AlignmentAutoTuner,
    wrap_angle_rad, signed_deadband,
)
from detector import FaceDetector
from serial_manager import SerialManager, find_serial_port, auto_detect_ports
from slam import LidarPoseManager
from planner import AStarPlanner, WaypointStore
from app_state import SharedAppState
from web_server import (
    WebHandler, ThreadedHTTPServer,
    update_web_frame, run_calibration_cli,
)
from web_ui import draw_dashboard


def main():
    parser = argparse.ArgumentParser(description='Camera based Web Test V7')
    parser.add_argument('--camera', type=int, default=CAMERA_INDEX)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--target-h', type=int, default=TARGET_BBOX_H)
    parser.add_argument('--model', type=str, default='yolov12n-face.pt')
    parser.add_argument('--conf', type=float, default=0.45)
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--classes', type=int, nargs='+', default=None)
    parser.add_argument('--kalman-r', type=float, default=KALMAN_R)
    parser.add_argument('--send-serial', action='store_true')
    parser.add_argument('--no-serial', action='store_true')
    parser.add_argument('--no-auto-detect', action='store_true')
    parser.add_argument('--serial-port', type=str, default=None)
    parser.add_argument('--no-lidar', action='store_true')
    parser.add_argument('--lidar-port', type=str, default=LIDAR_PORT)
    parser.add_argument('--lidar-baudrate', type=int, default=LIDAR_BAUDRATE)
    parser.add_argument('--lidar-baudrates', type=str, default=None)
    parser.add_argument('--map-file', type=str, default=MAP_FILE)
    parser.add_argument('--no-load-map', action='store_true')
    parser.add_argument('--map-autosave', type=float, default=MAP_AUTOSAVE_PERIOD_S)
    parser.add_argument('--waypoints-file', type=str, default=None)
    parser.add_argument('--config', type=str, default='robot_config.json')
    parser.add_argument('--calibrate', choices=['all', 'motors', 'wheel', 'rotation'],
                        default=None)
    parser.add_argument('--manual-distance', type=float, default=None)
    parser.add_argument('--manual-rotation', type=float, default=None)
    args = parser.parse_args()

    # --- CLI calibration mode ---
    if args.calibrate:
        run_calibration_cli(args, SharedAppState)
        return

    app_state = SharedAppState()
    app_state.config_path = args.config
    app_state.stats['target_bbox_h'] = float(args.target_h)
    app_state.stats['lidar_enabled'] = not args.no_lidar

    waypoints_file = args.waypoints_file
    if not waypoints_file:
        map_base, _ = os.path.splitext(args.map_file)
        waypoints_file = f'{map_base}_waypoints.json'
    app_state.waypoint_store = WaypointStore(waypoints_file)
    app_state.stats['waypoints'] = app_state.waypoint_store.list()
    app_state.stats['waypoints_file'] = waypoints_file

    if args.lidar_baudrates:
        lidar_baudrates = tuple(int(v.strip()) for v in args.lidar_baudrates.split(',') if v.strip())
    else:
        lidar_baudrates = (args.lidar_baudrate,) + tuple(
            b for b in LIDAR_BAUDRATES if b != args.lidar_baudrate
        )

    use_serial = not args.no_serial
    if args.send_serial:
        use_serial = True

    detected_lidar_port = None
    detected_arduino_port = None
    if not args.no_auto_detect and (use_serial or not args.no_lidar):
        print('\n=== AUTO-DETECT SERIAL PORTS ===')
        pref_lidar = args.lidar_port if (not args.no_lidar and args.lidar_port != LIDAR_PORT) else None
        pref_arduino = args.serial_port if (use_serial and args.serial_port) else None
        detected_lidar_port, detected_arduino_port = auto_detect_ports(
            preferred_lidar=pref_lidar,
            preferred_arduino=pref_arduino,
            lidar_baudrates=lidar_baudrates,
        )
        if detected_lidar_port and not args.no_lidar:
            args.lidar_port = detected_lidar_port
        if detected_arduino_port and use_serial and args.serial_port is None:
            args.serial_port = detected_arduino_port
        print()

    serial_mgr = None
    selected_serial_port = ''

    if use_serial:
        if detected_arduino_port:
            port = detected_arduino_port
        else:
            preferred_serial_ports = [args.serial_port]
            if args.serial_port is None and ARDUINO_PORT is not None:
                preferred_serial_ports.append(ARDUINO_PORT)
            port = find_serial_port(preferred_ports=preferred_serial_ports,
                                    exclude_ports=[args.lidar_port])
        selected_serial_port = port or ''
        if port:
            serial_mgr = SerialManager(port, config_path=args.config)
            if not serial_mgr.connect():
                serial_mgr = None
        else:
            print('[Serial] No Arduino port found. Running in monitor-only mode.')

    app_state.serial_mgr = serial_mgr
    app_state.stats['serial_enabled'] = serial_mgr is not None
    app_state.stats['serial_port'] = serial_mgr.port if serial_mgr is not None else selected_serial_port
    app_state.motor_enabled = serial_mgr is not None
    app_state.stats['motor_enabled'] = app_state.motor_enabled
    app_state.stats['motor_reason'] = (
        'Auto armed on startup' if serial_mgr is not None
        else 'Serial unavailable; check cable/port or use --serial-port'
    )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print('[Camera] Cannot open camera')
        return

    detector = FaceDetector(model_path=args.model, conf=args.conf,
                            imgsz=args.imgsz, classes=args.classes)

    ctrl_omega_turn  = AxisPD(TURN_KP, TURN_KD, -OMEGA_ALIGN_MAX, OMEGA_ALIGN_MAX)
    ctrl_vy_strafe   = AxisPD(STRAFE_KP, STRAFE_KD, -VY_ALIGN_MAX, VY_ALIGN_MAX)
    ctrl_omega_hold  = AxisPD(HOLD_KP, HOLD_KD, -HOLD_OMEGA_MAX, HOLD_OMEGA_MAX)
    pid_vx           = PID(VX_KP, VX_KI, VX_KD, -VX_MAX, VX_MAX, derivative_tau=0.10)
    vx_limiter       = SlewRateLimiter(VX_SLEW_MPS_S)
    vy_limiter       = SlewRateLimiter(VY_SLEW_MPS_S)
    omega_limiter    = SlewRateLimiter(OMEGA_SLEW_RAD_S2)
    align_mode       = 'TURN'
    align_tuner      = AlignmentAutoTuner()

    kf_cx = Kalman1D(q=KALMAN_Q, r=args.kalman_r)
    kf_bh = Kalman1D(q=KALMAN_Q, r=args.kalman_r * 2)
    frame_center_x = FRAME_WIDTH / 2.0

    hist_err_x   = deque(maxlen=HIST_LEN)
    hist_err_dist = deque(maxlen=HIST_LEN)
    hist_omega   = deque(maxlen=HIST_LEN)
    hist_vx      = deque(maxlen=HIST_LEN)
    track_heading_target = 0.0
    track_heading_integral = 0.0
    track_heading_initialized = False
    base_external_heading_active = False
    track_heading_deadband_rad = math.atan2(CENTER_DEADBAND * frame_center_x,
                                            CAMERA_FOCAL_PX)

    def set_base_heading_control(enabled):
        nonlocal base_external_heading_active

        enabled = bool(enabled and serial_mgr is not None)
        if enabled == base_external_heading_active:
            return

        if serial_mgr is not None:
            serial_mgr.send_heading_control_mode(enabled)
            mode_name = 'EXTERNAL' if enabled else 'INTERNAL'
            print(f'[Serial] Base heading control: {mode_name}')
        base_external_heading_active = enabled

    def vision_loop():
        fps_counter = 0
        fps_time = time.monotonic()
        while app_state.running:
            ret, frame = cap.read()
            if not ret:
                continue

            with app_state.lock:
                camera_tracking_enabled = app_state.camera_tracking_enabled

            result = detector.detect(frame) if camera_tracking_enabled else None
            with app_state.lock:
                if camera_tracking_enabled and result is not None:
                    cx, cy, bbox_h, bbox_w, bbox = result
                    app_state.ball_detected = True
                    app_state.last_seen_time = time.monotonic()
                    app_state.raw_cx = cx
                    app_state.raw_cy = cy
                    app_state.raw_bbox_h = bbox_h
                    app_state.raw_bbox_w = bbox_w
                    app_state.raw_bbox = bbox
                    app_state.new_measurement = True
                elif not camera_tracking_enabled:
                    app_state.ball_detected = False
                    app_state.new_measurement = False
                app_state.display_frame = frame

            if camera_tracking_enabled:
                fps_counter += 1
            now = time.monotonic()
            if now - fps_time >= 1.0:
                with app_state.lock:
                    app_state.fps_vision = (
                        fps_counter / (now - fps_time)
                        if app_state.camera_tracking_enabled else 0.0
                    )
                fps_counter = 0
                fps_time = now

    WebHandler.state = app_state
    httpd = ThreadedHTTPServer((args.host, args.port), WebHandler)
    web_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    web_thread.start()

    vision_thread = threading.Thread(target=vision_loop, daemon=True)
    vision_thread.start()

    lidar_mgr = None
    if args.no_lidar:
        app_state.stats['lidar_status'] = 'DISABLED'
    else:
        app_state.stats['lidar_port'] = args.lidar_port or ''
        lidar_mgr = LidarPoseManager(
            app_state, args.lidar_port, lidar_baudrates,
            map_file=args.map_file,
            autosave_period=args.map_autosave,
        )
        if args.no_load_map:
            lidar_mgr._skip_load = True
        app_state.lidar_mgr = lidar_mgr
        lidar_mgr.start()

    print('\n=== HDU SLAM ===')
    print(f'[Web] Open: http://{args.host if args.host != "0.0.0.0" else "<pi-ip>"}:{args.port}')
    print(f'[Control] {CONTROL_RATE} Hz | target bbox_h={args.target_h}px | target_dist={TARGET_DISTANCE_M:.2f}m')
    print(f'[Model] {args.model} | conf={args.conf} | imgsz={args.imgsz}')
    if lidar_mgr:
        print(f'[LiDAR] Pose enabled on {args.lidar_port} @ {lidar_baudrates}')
    else:
        print('[LiDAR] Disabled')
    if serial_mgr:
        print('[Serial] Sending commands to Arduino')
    else:
        print('[Serial] Monitor-only mode')
    print('Press Ctrl+C to stop.\n')

    ctrl_period = 1.0 / CONTROL_RATE
    last_kf_time = time.monotonic()
    ctrl_fps_counter = 0
    ctrl_fps_time = time.monotonic()
    lost_since = None
    last_navigation_active = False

    # nav phase machine state
    nav_phase = 'TURN_TO_PATH'
    last_vx_cmd = 0.0
    last_omega_cmd = 0.0
    astar_planner = AStarPlanner(ASTAR_RESOLUTION_M, ASTAR_INFLATE_M)

    try:
        while app_state.running:
            t_start = time.monotonic()
            dt = t_start - last_kf_time
            if dt <= 0:
                dt = ctrl_period
            last_kf_time = t_start

            with app_state.lock:
                ball_detected         = app_state.ball_detected
                camera_tracking_enabled = app_state.camera_tracking_enabled
                last_seen             = app_state.last_seen_time
                has_new               = app_state.new_measurement
                raw_cx                = app_state.raw_cx
                raw_cy                = app_state.raw_cy
                raw_bh                = app_state.raw_bbox_h
                raw_bw                = app_state.raw_bbox_w
                raw_bbox              = app_state.raw_bbox
                display_frame         = (None if app_state.display_frame is None
                                         else app_state.display_frame.copy())
                fps_vision            = app_state.fps_vision
                navigation_active     = app_state.navigation_active
                navigation_target     = (dict(app_state.navigation_target)
                                         if app_state.navigation_target else None)
                lidar_x               = float(app_state.stats.get('lidar_x', 0.0) or 0.0)
                lidar_y               = float(app_state.stats.get('lidar_y', 0.0) or 0.0)
                lidar_theta_deg       = float(app_state.stats.get('lidar_theta_deg', 0.0) or 0.0)
                lidar_status          = app_state.stats.get('lidar_status', 'DISABLED')
                lidar_last_update     = float(app_state.stats.get('lidar_last_update', 0.0) or 0.0)
                motor_enabled_snapshot = app_state.motor_enabled
                calibration_active_snapshot = bool(
                    getattr(app_state, 'calibration_active', False)
                    or app_state.stats.get('calibration_active', False)
                )
                app_state.new_measurement = False

            base_theta_deg = float(serial_mgr.robot_theta_deg) if serial_mgr is not None else 0.0
            base_wz_feedback = float(serial_mgr.measured_wz) if serial_mgr is not None else 0.0
            base_imu_ok = bool(serial_mgr.imu_ok) if serial_mgr is not None else False
            base_telm_age = (
                max(0.0, time.monotonic() - serial_mgr.last_telm_time)
                if serial_mgr is not None and serial_mgr.last_telm_time > 0.0
                else float('inf')
            )

            vx_target = 0.0; vx = 0.0
            vy_target = 0.0; vy = 0.0
            omega_target = 0.0; omega = 0.0
            err_x = 0.0; err_dist = 0.0; dist_m = 0.0
            vx_ff = 0.0; bearing_rate = 0.0
            omega_p = 0.0; omega_i = 0.0; omega_d = 0.0
            vy_p = 0.0; vy_d = 0.0
            align_gain_scale = align_tuner.gain_scale
            navigation_status = 'IDLE'
            navigation_reason = ''
            navigation_distance = 0.0
            navigation_bearing_error_deg = 0.0
            track_heading_error = 0.0
            track_camera_bearing = 0.0
            track_imu_active = False
            send_stop_now = False
            nav_control_active = bool(navigation_active and navigation_target)
            has_track_heading = (
                TRACK_USE_IMU_HEADING
                and serial_mgr is not None
                and base_imu_ok
                and base_telm_age <= TRACK_IMU_STALE_S
                and math.isfinite(base_theta_deg)
                and math.isfinite(base_wz_feedback)
            )

            set_base_heading_control(
                (not nav_control_active)
                and camera_tracking_enabled
                and has_track_heading
            )

            if not base_external_heading_active:
                track_heading_initialized = False
                track_heading_integral = 0.0

            if nav_control_active != last_navigation_active:
                ctrl_omega_turn.reset(); ctrl_vy_strafe.reset()
                ctrl_omega_hold.reset(); pid_vx.reset()
                align_tuner.reset()
                vx_limiter.reset(0.0); vy_limiter.reset(0.0); omega_limiter.reset(0.0)
                align_mode = 'TURN'
                lost_since = None
                nav_phase = 'TURN_TO_PATH'
                last_vx_cmd = 0.0; last_omega_cmd = 0.0
                track_heading_initialized = False
                track_heading_integral = 0.0
                app_state.nav_path_waypoints = []
                app_state.nav_path_idx = 0
                app_state.nav_path_target_id = None
                app_state.nav_last_plan_t = 0.0
                last_navigation_active = nav_control_active

            if nav_control_active:
                now_nav = time.monotonic()
                lidar_age = now_nav - lidar_last_update
                navigation_status = 'MOVING'
                if lidar_status != 'TRACKING' or lidar_age > NAV_LIDAR_STALE_S:
                    navigation_status = 'WAIT_LIDAR'
                    navigation_reason = 'LiDAR pose is stale'
                    vx_target = vy_target = omega_target = 0.0
                elif not motor_enabled_snapshot:
                    navigation_status = 'WAIT_MOTOR'
                    navigation_reason = 'Motor is stopped'
                    vx_target = vy_target = omega_target = 0.0
                else:
                    target_x = float(navigation_target.get('x', 0.0))
                    target_y = float(navigation_target.get('y', 0.0))
                    goal_theta_raw = navigation_target.get('theta', None)
                    has_goal_heading = goal_theta_raw is not None
                    goal_theta = float(goal_theta_raw) if has_goal_heading else 0.0

                    final_goal_x = target_x
                    final_goal_y = target_y

                    # ===== A* PATH MANAGEMENT =====
                    now_plan = time.monotonic()
                    need_replan = (
                        app_state.nav_path_target_id != navigation_target.get('id')
                        or (now_plan - app_state.nav_last_plan_t > ASTAR_REPLAN_PERIOD_S)
                        or (app_state.nav_path_idx >= len(app_state.nav_path_waypoints)
                            and math.hypot(final_goal_x - lidar_x, final_goal_y - lidar_y) > ASTAR_SUBWP_TOL_M)
                    )
                    if need_replan:
                        with app_state.lock:
                            map_pts = list(app_state.stats.get('lidar_map', []))
                        planned = astar_planner.plan((lidar_x, lidar_y),
                                                     (final_goal_x, final_goal_y),
                                                     map_pts) if map_pts else None
                        app_state.nav_path_waypoints = planned if planned else []
                        app_state.nav_path_idx = 0
                        app_state.nav_path_target_id = navigation_target.get('id')
                        app_state.nav_last_plan_t = now_plan
                        if planned:
                            print(f'[Nav] A* path planned: {len(planned)} waypoints to '
                                  f'({final_goal_x:.2f},{final_goal_y:.2f})')
                        else:
                            print('[Nav] A* no path found, going direct')

                    if app_state.nav_path_waypoints and app_state.nav_path_idx < len(app_state.nav_path_waypoints):
                        sub_wp = app_state.nav_path_waypoints[app_state.nav_path_idx]
                        if math.hypot(sub_wp[0] - lidar_x, sub_wp[1] - lidar_y) <= ASTAR_SUBWP_TOL_M:
                            app_state.nav_path_idx += 1
                        if app_state.nav_path_idx < len(app_state.nav_path_waypoints):
                            sub_wp = app_state.nav_path_waypoints[app_state.nav_path_idx]
                            target_x = sub_wp[0]
                            target_y = sub_wp[1]
                        else:
                            target_x = final_goal_x
                            target_y = final_goal_y
                    # ===== END A* PATH MANAGEMENT =====

                    dx_world = target_x - lidar_x
                    dy_world = target_y - lidar_y
                    navigation_distance = math.hypot(dx_world, dy_world)
                    final_distance = math.hypot(final_goal_x - lidar_x, final_goal_y - lidar_y)
                    theta = math.radians(lidar_theta_deg)

                    desired_heading = math.atan2(dy_world, dx_world)
                    heading_error = wrap_angle_rad(desired_heading - theta)
                    navigation_bearing_error_deg = math.degrees(heading_error)
                    abs_heading_err_deg = abs(navigation_bearing_error_deg)

                    # Phase transitions
                    if nav_phase == 'TURN_TO_PATH':
                        if final_distance <= NAV_GOAL_TOLERANCE_M:
                            nav_phase = 'TURN_TO_GOAL' if has_goal_heading else 'ARRIVED'
                        elif abs_heading_err_deg <= NAV_PATH_HEADING_TOL_DEG:
                            nav_phase = 'DRIVE'
                            print(f'[Nav] phase: TURN_TO_PATH -> DRIVE '
                                  f'(heading_err={navigation_bearing_error_deg:+.1f}deg, '
                                  f'dist={final_distance:.2f}m)')

                    elif nav_phase == 'DRIVE':
                        if final_distance <= NAV_GOAL_TOLERANCE_M:
                            nav_phase = 'TURN_TO_GOAL' if has_goal_heading else 'ARRIVED'
                            print(f'[Nav] phase: DRIVE -> '
                                  f'{"TURN_TO_GOAL" if has_goal_heading else "ARRIVED"}')
                        elif abs_heading_err_deg > NAV_PATH_HEADING_HYST_DEG:
                            nav_phase = 'TURN_TO_PATH'
                            print(f'[Nav] phase: DRIVE -> TURN_TO_PATH '
                                  f'(heading_err={navigation_bearing_error_deg:+.1f}deg)')

                    elif nav_phase == 'TURN_TO_GOAL':
                        heading_error = wrap_angle_rad(goal_theta - theta)
                        navigation_bearing_error_deg = math.degrees(heading_error)
                        abs_heading_err_deg = abs(navigation_bearing_error_deg)
                        if abs_heading_err_deg <= NAV_HEADING_TOLERANCE_DEG:
                            nav_phase = 'ARRIVED'

                    # Compute control per phase
                    if nav_phase == 'ARRIVED':
                        navigation_status = 'ARRIVED'
                        navigation_reason = f'Reached {navigation_target.get("label", "target")}'
                        vx_target = vy_target = omega_target = 0.0
                        send_stop_now = True
                        with app_state.lock:
                            app_state.navigation_active = False
                            app_state.navigation_target = navigation_target
                            app_state.motor_enabled = False
                            app_state.stats['motor_enabled'] = False
                            app_state.stats['motor_reason'] = navigation_reason
                            app_state.stats['navigation_active'] = False
                            app_state.stats['navigation_target'] = navigation_target
                            app_state.stats['navigation_status'] = navigation_status
                            app_state.stats['navigation_reason'] = navigation_reason
                            app_state.stats['navigation_phase'] = nav_phase

                    elif nav_phase in ('TURN_TO_PATH', 'TURN_TO_GOAL'):
                        navigation_status = nav_phase
                        vx_target = 0.0
                        vy_target = 0.0
                        omega_raw = NAV_HEADING_KP * heading_error
                        if abs(omega_raw) < NAV_OMEGA_MIN_MOVE and abs_heading_err_deg > NAV_HEADING_TOLERANCE_DEG:
                            omega_raw = NAV_OMEGA_MIN_MOVE * (1.0 if heading_error > 0 else -1.0)
                        omega_target = float(np.clip(
                            omega_raw,
                            -min(NAV_OMEGA_MAX, OMEGA_MAX),
                            min(NAV_OMEGA_MAX, OMEGA_MAX),
                        ))

                    elif nav_phase == 'DRIVE':
                        navigation_status = 'DRIVE'
                        vx_raw = NAV_TRANSLATION_KP * navigation_distance
                        if navigation_distance < NAV_DECEL_DIST_M:
                            decel_factor = navigation_distance / NAV_DECEL_DIST_M
                            vx_raw = min(vx_raw, NAV_VX_MAX * decel_factor)
                            if navigation_distance < NAV_FINAL_APPROACH_M:
                                vx_raw = min(vx_raw, NAV_VX_MAX * 0.4)
                        if vx_raw < NAV_VX_MIN_MOVE and navigation_distance > NAV_GOAL_TOLERANCE_M:
                            vx_raw = NAV_VX_MIN_MOVE
                        vx_target = float(np.clip(vx_raw, 0.0, min(NAV_VX_MAX, VX_MAX)))
                        vy_target = 0.0
                        omega_target = float(np.clip(
                            NAV_HEADING_KP * heading_error * 0.7,
                            -min(NAV_OMEGA_MAX * 0.7, OMEGA_MAX),
                            min(NAV_OMEGA_MAX * 0.7, OMEGA_MAX),
                        ))

                        # LOCAL OBSTACLE AVOIDANCE
                        if vx_target > 0.01:
                            with app_state.lock:
                                live_cloud = list(app_state.stats.get('lidar_cloud', []))
                            if len(live_cloud) >= OBS_MIN_POINTS:
                                cloud_world = np.array([[p[0], p[1]] for p in live_cloud],
                                                       dtype=np.float32)
                                theta_r = math.radians(lidar_theta_deg)
                                cos_r = math.cos(-theta_r)
                                sin_r = math.sin(-theta_r)
                                dx_w = cloud_world[:, 0] - lidar_x
                                dy_w = cloud_world[:, 1] - lidar_y
                                pts_rx = cos_r * dx_w - sin_r * dy_w
                                pts_ry = sin_r * dx_w + cos_r * dy_w
                                in_front = pts_rx > 0.15
                                cone_half = math.radians(OBS_CONE_HALF_DEG)
                                in_cone = np.abs(np.arctan2(pts_ry, pts_rx)) < cone_half
                                dist = np.hypot(pts_rx, pts_ry)
                                mask_safe = in_front & in_cone & (dist < OBS_SAFE_DIST_M)
                                if int(np.sum(mask_safe)) >= OBS_MIN_POINTS:
                                    vx_target = 0.0
                                    omega_target = 0.0
                                    navigation_status = 'OBSTACLE'
                                else:
                                    mask_slow = in_front & in_cone & (dist < OBS_SLOW_DIST_M)
                                    n_slow = int(np.sum(mask_slow))
                                    if n_slow >= OBS_MIN_POINTS:
                                        cand = dist[mask_slow]
                                        closest = float(cand.min()) if len(cand) else OBS_SLOW_DIST_M
                                        span = max(OBS_SLOW_DIST_M - OBS_SAFE_DIST_M, 0.01)
                                        factor = max(0.2, (closest - OBS_SAFE_DIST_M) / span)
                                        vx_target = min(vx_target, NAV_VX_MAX * factor)

                    else:
                        vx_target = vy_target = omega_target = 0.0

                    # Rate limiting
                    dt_safe = max(dt, 1e-3)
                    max_vx_change = NAV_VX_RATE_LIMIT * dt_safe
                    vx_target = max(last_vx_cmd - max_vx_change,
                                    min(vx_target, last_vx_cmd + max_vx_change))
                    last_vx_cmd = vx_target

                    max_omega_change = NAV_OMEGA_RATE_LIMIT * dt_safe
                    omega_target = max(last_omega_cmd - max_omega_change,
                                       min(omega_target, last_omega_cmd + max_omega_change))
                    last_omega_cmd = omega_target

                    with app_state.lock:
                        app_state.stats['navigation_phase'] = nav_phase

                err_dist = navigation_distance

            if not nav_control_active and not camera_tracking_enabled:
                ball_detected = False
                vx_target = vy_target = omega_target = 0.0
                lost_since = time.monotonic() - LOST_SOFT_STOP

            elif not nav_control_active and ball_detected:
                lost_since = None
                if has_new:
                    filt_cx = kf_cx.update(raw_cx, dt)
                    filt_bh = kf_bh.update(raw_bh, dt)
                else:
                    filt_cx = kf_cx.predict(dt)
                    filt_bh = kf_bh.predict(dt)

                err_x = (filt_cx - frame_center_x) / frame_center_x
                bh_safe = max(filt_bh, 5.0)
                dist_m = (REAL_FACE_HEIGHT_M * CAMERA_FOCAL_PX) / bh_safe
                err_dist_m = dist_m - TARGET_DISTANCE_M
                err_dist = float(np.clip(err_dist_m / max(TARGET_DISTANCE_M, 1e-3), -1.0, 1.0))

                bearing_rate = kf_cx.dx / CAMERA_FOCAL_PX
                track_camera_bearing = math.atan2(filt_cx - frame_center_x, CAMERA_FOCAL_PX)
                dist_rate = -REAL_FACE_HEIGHT_M * CAMERA_FOCAL_PX * kf_bh.dx / (bh_safe * bh_safe)
                vx_ff = VX_FF_GAIN * dist_rate

                align_err = abs(err_x)
                err_x_db = signed_deadband(err_x, CENTER_DEADBAND)

                prev_align_mode = align_mode
                if align_mode == 'TURN':
                    if align_err <= ALIGN_HOLD_ENTER and abs(bearing_rate) <= ALIGN_RATE_HOLD:
                        align_mode = 'HOLD'
                    elif align_err <= ALIGN_STRAFE_ENTER:
                        align_mode = 'STRAFE'
                elif align_mode == 'STRAFE':
                    if align_err >= ALIGN_STRAFE_EXIT:
                        align_mode = 'TURN'
                    elif align_err <= ALIGN_HOLD_ENTER and abs(bearing_rate) <= ALIGN_RATE_HOLD:
                        align_mode = 'HOLD'
                elif align_mode == 'HOLD':
                    if align_err >= ALIGN_STRAFE_EXIT:
                        align_mode = 'TURN'
                    elif align_err >= ALIGN_HOLD_EXIT:
                        align_mode = 'STRAFE'
                else:
                    align_mode = 'TURN'

                if align_mode != prev_align_mode:
                    ctrl_omega_turn.reset()
                    ctrl_vy_strafe.reset()
                    ctrl_omega_hold.reset()

                align_gain_scale, crossed = align_tuner.update(err_x, bearing_rate, dt)

                if base_external_heading_active and has_track_heading:
                    base_theta_rad = math.radians(base_theta_deg)
                    observed_heading = wrap_angle_rad(base_theta_rad + track_camera_bearing)
                    track_imu_active = True

                    if not track_heading_initialized:
                        track_heading_target = observed_heading
                        track_heading_integral = 0.0
                        track_heading_initialized = True

                    if align_mode == 'TURN':
                        max_target_step = TRACK_HEADING_TARGET_RATE_RAD_S * max(dt, 1e-3)
                        heading_delta = wrap_angle_rad(observed_heading - track_heading_target)
                        heading_delta = float(np.clip(heading_delta, -max_target_step, max_target_step))
                        track_heading_target = wrap_angle_rad(track_heading_target + heading_delta)

                    track_heading_error = wrap_angle_rad(track_heading_target - base_theta_rad)
                    if abs(track_heading_error) <= track_heading_deadband_rad:
                        track_heading_error = 0.0
                        if TRACK_HEADING_KI > 0.0:
                            track_heading_integral *= 0.85
                    elif TRACK_HEADING_KI > 0.0:
                        track_heading_integral += track_heading_error * dt
                        max_heading_integral = TRACK_HEADING_MAX / max(TRACK_HEADING_KI, 1e-6)
                        track_heading_integral = float(np.clip(
                            track_heading_integral,
                            -max_heading_integral,
                            max_heading_integral,
                        ))

                    imu_omega_cmd = (
                        TRACK_HEADING_KP * track_heading_error
                        + TRACK_HEADING_KI * track_heading_integral
                        - TRACK_HEADING_KD * base_wz_feedback
                    )

                if align_mode == 'TURN':
                    vy_target = 0.0
                    if track_imu_active:
                        omega_target = float(np.clip(
                            align_gain_scale * imu_omega_cmd,
                            -TRACK_HEADING_MAX,
                            TRACK_HEADING_MAX,
                        ))
                        omega_p = align_gain_scale * TRACK_HEADING_KP * track_heading_error
                        omega_i = align_gain_scale * TRACK_HEADING_KI * track_heading_integral
                        omega_d = -align_gain_scale * TRACK_HEADING_KD * base_wz_feedback
                    else:
                        omega_target = ctrl_omega_turn.compute(err_x_db, bearing_rate,
                                                               scale=align_gain_scale)
                        omega_p = ctrl_omega_turn.last_p
                        omega_d = ctrl_omega_turn.last_d
                elif align_mode == 'STRAFE':
                    vy_target = ctrl_vy_strafe.compute(err_x_db, bearing_rate,
                                                        scale=align_gain_scale)
                    vy_p = ctrl_vy_strafe.last_p
                    vy_d = ctrl_vy_strafe.last_d
                    if track_imu_active:
                        omega_target = float(np.clip(
                            imu_omega_cmd,
                            -HOLD_OMEGA_MAX,
                            HOLD_OMEGA_MAX,
                        ))
                        omega_p = TRACK_HEADING_KP * track_heading_error
                        omega_i = TRACK_HEADING_KI * track_heading_integral
                        omega_d = -TRACK_HEADING_KD * base_wz_feedback
                    else:
                        omega_target = ctrl_omega_hold.compute(err_x_db, bearing_rate, scale=1.0)
                        omega_p = ctrl_omega_hold.last_p
                        omega_d = ctrl_omega_hold.last_d
                else:
                    vy_target = 0.0
                    if track_imu_active:
                        omega_target = float(np.clip(
                            0.7 * imu_omega_cmd,
                            -HOLD_OMEGA_MAX,
                            HOLD_OMEGA_MAX,
                        ))
                        omega_p = 0.7 * TRACK_HEADING_KP * track_heading_error
                        omega_i = 0.7 * TRACK_HEADING_KI * track_heading_integral
                        omega_d = -0.7 * TRACK_HEADING_KD * base_wz_feedback
                        if align_err <= ALIGN_HOLD_ENTER and track_heading_error == 0.0:
                            omega_target = 0.0
                    else:
                        omega_target = ctrl_omega_hold.compute(err_x, bearing_rate, scale=0.7)
                        omega_p = ctrl_omega_hold.last_p
                        omega_d = ctrl_omega_hold.last_d
                        if align_err <= ALIGN_HOLD_ENTER:
                            omega_target = 0.0

                if crossed:
                    vy_target *= 0.65
                    omega_target *= 0.65

                if abs(err_dist_m) < DIST_DEADBAND_M:
                    vx_target = float(np.clip(vx_ff, -VX_MAX, VX_MAX))
                    pid_vx.reset()
                else:
                    vx_target = pid_vx.compute(err_dist, dt, feedforward=vx_ff)

                if err_dist_m < -DIST_DEADBAND_M:
                    backoff_speed = min(CLOSE_BACKOFF_MAX, CLOSE_BACKOFF_GAIN * (-err_dist_m))
                    vx_target = min(vx_target, -backoff_speed)

                if align_err >= FORWARD_ALIGN_LIMIT:
                    alignment = 0.0
                else:
                    scaled_align = align_err / max(FORWARD_ALIGN_LIMIT, 1e-6)
                    cos_gate = math.cos(0.5 * math.pi * scaled_align)
                    alignment = FORWARD_ALIGN_MIN_GAIN + (1.0 - FORWARD_ALIGN_MIN_GAIN) * (cos_gate * cos_gate)
                if vx_target > 0.0:
                    vx_target *= alignment

                seen_age = max(0.0, time.monotonic() - last_seen)
                visibility = float(np.clip(1.0 - seen_age / LOST_TIMEOUT, 0.0, 1.0))
                vx_target *= visibility
                vy_target *= visibility
                omega_target *= 0.35 + 0.65 * visibility

                if seen_age > LOST_TIMEOUT:
                    with app_state.lock:
                        app_state.ball_detected = False
                    ball_detected = False
                    lost_since = time.monotonic()

            if not nav_control_active and not ball_detected:
                track_heading_initialized = False
                track_heading_integral = 0.0
                now_lost = time.monotonic()
                if lost_since is None:
                    lost_since = now_lost
                lost_duration = now_lost - lost_since

                if lost_duration < LOST_COAST_TIME:
                    vx_target = vx_limiter.value
                    vy_target = vy_limiter.value
                    omega_target = omega_limiter.value
                elif lost_duration < LOST_SOFT_STOP:
                    vx_target = vy_target = omega_target = 0.0
                else:
                    vx_target = vy_target = omega_target = 0.0

                if lost_duration > LOST_HARD_RESET:
                    ctrl_omega_turn.reset(); ctrl_vy_strafe.reset()
                    ctrl_omega_hold.reset(); pid_vx.reset()
                    align_tuner.reset()
                    align_mode = 'TURN'
                    kf_cx.reset(); kf_bh.reset()

            vx    = vx_limiter.update(vx_target, dt)
            vy    = vy_limiter.update(vy_target, dt)
            omega = omega_limiter.update(omega_target, dt)

            hist_err_x.append(err_x)
            hist_err_dist.append(err_dist)
            hist_omega.append(omega)
            hist_vx.append(vx)

            ctrl_fps_counter += 1
            now = time.monotonic()
            if now - ctrl_fps_time >= 1.0:
                with app_state.lock:
                    app_state.fps_ctrl = ctrl_fps_counter / (now - ctrl_fps_time)
                ctrl_fps_counter = 0
                ctrl_fps_time = now

            if serial_mgr and calibration_active_snapshot:
                pass
            elif send_stop_now and serial_mgr:
                serial_mgr.send_stop()
                serial_mgr.read_feedback()
            elif serial_mgr and app_state.motor_enabled:
                serial_mgr.send_velocity(vx, vy, omega)
                serial_mgr.read_feedback()
            elif serial_mgr and not app_state.motor_enabled:
                ctrl_omega_turn.reset(); ctrl_vy_strafe.reset()
                ctrl_omega_hold.reset(); pid_vx.reset()
                align_tuner.reset(); align_mode = 'TURN'
                vx_limiter.reset(0.0); vy_limiter.reset(0.0); omega_limiter.reset(0.0)
                vx = vy = omega = 0.0
                serial_mgr.read_feedback()

            with app_state.lock:
                app_state.stats['navigation_active'] = (nav_control_active
                                                         and navigation_status != 'ARRIVED')
                app_state.stats['navigation_target'] = navigation_target
                app_state.stats['navigation_status'] = navigation_status
                app_state.stats['navigation_reason'] = navigation_reason
                app_state.stats['navigation_distance'] = navigation_distance
                app_state.stats['navigation_bearing_error_deg'] = navigation_bearing_error_deg

            if display_frame is not None:
                frame = display_frame
                if ball_detected:
                    x1, y1, x2, y2 = raw_bbox
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (int(raw_cx), int(raw_cy)), 4, (255, 150, 0), -1)

                    if kf_cx.initialized:
                        filt_y = raw_cy
                        cv2.circle(frame, (int(kf_cx.x), int(filt_y)), 8, (0, 0, 255), 2)
                        arrow_x = int(kf_cx.dx * 0.5)
                        arrow_y = int(kf_bh.dx * 0.15)
                        cv2.arrowedLine(
                            frame,
                            (int(kf_cx.x), int(filt_y)),
                            (int(kf_cx.x) + arrow_x, int(filt_y) + arrow_y),
                            (0, 0, 255), 2, tipLength=0.3,
                        )

                    if kf_bh.initialized:
                        filt_half_h = kf_bh.x / 2.0
                        filt_half_w = raw_bw / 2.0
                        fx1 = int(kf_cx.x - filt_half_w)
                        fy1 = int(raw_cy - filt_half_h)
                        fx2 = int(kf_cx.x + filt_half_w)
                        fy2 = int(raw_cy + filt_half_h)
                        for pos_x in range(fx1, fx2, 8):
                            cv2.line(frame, (pos_x, fy1), (min(pos_x + 4, fx2), fy1), (0, 255, 255), 1)
                            cv2.line(frame, (pos_x, fy2), (min(pos_x + 4, fx2), fy2), (0, 255, 255), 1)
                        for pos_y in range(fy1, fy2, 8):
                            cv2.line(frame, (fx1, pos_y), (fx1, min(pos_y + 4, fy2)), (0, 255, 255), 1)
                            cv2.line(frame, (fx2, pos_y), (fx2, min(pos_y + 4, fy2)), (0, 255, 255), 1)
                        label = f'face h={raw_bh:.0f}'
                        if kf_bh.initialized:
                            label += f'>{kf_bh.x:.0f}'
                        cv2.putText(frame, label, (int(x1), int(y1) - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                center_x = int(frame_center_x)
                center_y = int(FRAME_HEIGHT / 2)
                cv2.line(frame, (center_x - 25, center_y), (center_x + 25, center_y), (100, 100, 100), 1)
                cv2.line(frame, (center_x, center_y - 25), (center_x, center_y + 25), (100, 100, 100), 1)
                cv2.circle(frame, (center_x, center_y), 30, (100, 100, 100), 1)

                last_seen_age = max(0.0, time.monotonic() - last_seen) if last_seen > 0 else 0.0
                display_status = navigation_status if nav_control_active else (
                    'TRACKING' if ball_detected
                    else ('TRACKING_OFF' if not camera_tracking_enabled else 'SEARCHING')
                )
                info = {
                    'status': display_status,
                    'ball_detected': ball_detected,
                    'camera_tracking_enabled': camera_tracking_enabled,
                    'vx': vx, 'vy': vy, 'omega': omega,
                    'vx_target': vx_target, 'vy_target': vy_target, 'omega_target': omega_target,
                    'err_x': err_x, 'err_dist': err_dist, 'dist_m': dist_m,
                    'kf_cx':  kf_cx.x  if kf_cx.initialized  else 0.0,
                    'kf_bh':  kf_bh.x  if kf_bh.initialized  else 0.0,
                    'kf_dcx': kf_cx.dx if kf_cx.initialized  else 0.0,
                    'kf_dbh': kf_bh.dx if kf_bh.initialized  else 0.0,
                    'measured_vx':   serial_mgr.measured_vx   if serial_mgr else 0.0,
                    'measured_vy':   serial_mgr.measured_vy   if serial_mgr else 0.0,
                    'measured_wz':   serial_mgr.measured_wz   if serial_mgr else 0.0,
                    'robot_x':       serial_mgr.robot_x       if serial_mgr else 0.0,
                    'robot_y':       serial_mgr.robot_y       if serial_mgr else 0.0,
                    'robot_theta_deg': serial_mgr.robot_theta_deg if serial_mgr else 0.0,
                    'imu_ok':        serial_mgr.imu_ok        if serial_mgr else False,
                    'align_mode': align_mode,
                    'align_gain_scale': align_gain_scale,
                    'align_overshoots': align_tuner.overshoots,
                    'bearing_rate': bearing_rate,
                    'track_imu_active': track_imu_active,
                    'track_heading_error_deg': math.degrees(track_heading_error),
                    'track_heading_target_deg': math.degrees(track_heading_target) if track_heading_initialized else 0.0,
                    'track_camera_bearing_deg': math.degrees(track_camera_bearing),
                    'track_heading_source': 'imu' if track_imu_active else 'image_pd',
                    'omega_p': omega_p, 'omega_i': omega_i, 'omega_d': omega_d,
                    'vy_p': vy_p, 'vy_d': vy_d,
                    'vx_p': pid_vx.last_p, 'vx_i': pid_vx.last_i, 'vx_d': pid_vx.last_d,
                    'navigation_active': nav_control_active and navigation_status != 'ARRIVED',
                    'navigation_target': navigation_target,
                    'navigation_status': navigation_status,
                    'navigation_reason': navigation_reason,
                    'navigation_distance': navigation_distance,
                    'navigation_bearing_error_deg': navigation_bearing_error_deg,
                    'fps_vision': fps_vision,
                    'fps_ctrl': app_state.fps_ctrl,
                    'hist_err_x': list(hist_err_x),
                    'hist_err_dist': list(hist_err_dist),
                    'hist_omega': list(hist_omega),
                    'hist_vx': list(hist_vx),
                }
                draw_dashboard(frame, info, vx_max=VX_MAX, omega_max=OMEGA_MAX)
                update_web_frame(app_state, frame)

                with app_state.lock:
                    app_state.stats.update(info)
                    app_state.stats['serial_enabled'] = serial_mgr is not None
                    app_state.stats['motor_enabled'] = app_state.motor_enabled
                    app_state.stats['target_bbox_h'] = float(args.target_h)
                    app_state.stats['last_seen_age'] = last_seen_age

            elapsed = time.monotonic() - t_start
            sleep_time = ctrl_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print('\n[Exit] Ctrl+C')
    finally:
        app_state.running = False
        httpd.shutdown()
        httpd.server_close()
        if lidar_mgr:
            lidar_mgr.stop()
        if serial_mgr:
            set_base_heading_control(False)
            serial_mgr.send_stop()
            serial_mgr.close()
        cap.release()
        print('[Exit] Stopped.')


if __name__ == '__main__':
    main()
