#!/usr/bin/env python3
"""
Ball Chaser Web + LiDAR Localization
====================================

Phien ban nay giu nguyen pipeline camera tracking hien co va bo sung:
  - Doc LiDAR 2D tren Raspberry Pi
  - ICP scan matching giua cac scan lien tiep
  - Fusion odometry/IMU tu base voi delta pose tu LiDAR
  - Giam toc / chan lenh khi truoc mat co vat can
  - Hien thi pose hop nhat va trang thai LiDAR tren web

Toa do pose duoc tinh theo moc luc khoi dong, khong phai ban do toan cuc.
Muoi dung localization toan cuc/SLAM day du thi nen dua tiep len ROS2/cartographer.

Thu vien can co tren Pi:
  pip install numpy opencv-python pyserial ultralytics rplidar-roboticia

Vi du:
  python3 ball_chaser_web_lidar_v1.py --send-serial --serial-port /dev/ttyACM0 --lidar-port /dev/ttyUSB0
"""

import argparse
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from http import server
from socketserver import ThreadingMixIn

import cv2
import numpy as np

import ball_chaser_web_test_v3 as base

try:
    from rplidar import RPLidar
    LIDAR_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency at runtime
    RPLidar = None
    LIDAR_IMPORT_ERROR = exc


LIDAR_MIN_RANGE_M = 0.12
LIDAR_MAX_RANGE_M = 5.5
LIDAR_DOWNSAMPLE_M = 0.06
LIDAR_MAX_POINTS = 140
LIDAR_MATCH_MAX_DIST_M = 0.30
LIDAR_MATCH_MIN_POINTS = 24
LIDAR_MAX_TRANSLATION_STEP_M = 0.40
LIDAR_MAX_ROTATION_STEP_RAD = math.radians(25.0)
LIDAR_QUALITY_BLEND = 0.45
LIDAR_STALE_TIMEOUT = 1.0

OBSTACLE_STOP_FRONT_M = 0.32
OBSTACLE_SLOW_FRONT_M = 0.85
OBSTACLE_STOP_SIDE_M = 0.24
OBSTACLE_SLOW_SIDE_M = 0.55

PATH_LEN = 220
SCAN_HISTORY_LEN = 6


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def rotation_matrix(theta):
    ct = math.cos(theta)
    st = math.sin(theta)
    return np.array([[ct, -st], [st, ct]], dtype=np.float32)


def compose_delta(delta_a, delta_b):
    ax, ay, ath = delta_a
    bx, by, bth = delta_b
    rot = rotation_matrix(ath)
    translated = rot @ np.array([bx, by], dtype=np.float32)
    return (
        ax + float(translated[0]),
        ay + float(translated[1]),
        wrap_angle(ath + bth),
    )


def transform_points(points, pose_delta):
    if points.size == 0:
        return points
    dx, dy, theta = pose_delta
    rot = rotation_matrix(theta)
    transformed = points @ rot.T
    transformed[:, 0] += dx
    transformed[:, 1] += dy
    return transformed


def estimate_rigid_transform(source, target):
    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid

    h_mat = src_centered.T @ tgt_centered
    u_mat, _, vt_mat = np.linalg.svd(h_mat)
    rot = vt_mat.T @ u_mat.T
    if np.linalg.det(rot) < 0:
        vt_mat[1, :] *= -1.0
        rot = vt_mat.T @ u_mat.T

    theta = math.atan2(rot[1, 0], rot[0, 0])
    trans = tgt_centroid - rot @ src_centroid
    return float(trans[0]), float(trans[1]), theta


def clip_points(points, max_points):
    if points.shape[0] <= max_points:
        return points
    step = max(1, points.shape[0] // max_points)
    return points[::step][:max_points]


def voxel_downsample(points, voxel_size):
    if points.size == 0:
        return points
    grid = np.floor(points / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(grid, axis=0, return_index=True)
    sampled = points[np.sort(unique_idx)]
    return sampled


def world_to_local_delta(prev_pose, curr_pose):
    dx_world = curr_pose.x - prev_pose.x
    dy_world = curr_pose.y - prev_pose.y
    dtheta = wrap_angle(curr_pose.theta - prev_pose.theta)
    ct = math.cos(prev_pose.theta)
    st = math.sin(prev_pose.theta)
    dx_local = ct * dx_world + st * dy_world
    dy_local = -st * dx_world + ct * dy_world
    return dx_local, dy_local, dtheta


@dataclass
class OdomState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    stamp: float = 0.0


@dataclass
class ScanSnapshot:
    stamp: float
    points: np.ndarray
    front_min_m: float
    left_min_m: float
    right_min_m: float
    scan_rate_hz: float
    raw_count: int


class LidarManager:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.lidar = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_scan = None
        self.status = 'DISABLED'
        self.error = None
        self.scan_rate_hz = 0.0

    def start(self):
        if self.port is None:
            self.status = 'DISABLED'
            return False
        if RPLidar is None:
            self.status = 'IMPORT_ERROR'
            self.error = str(LIDAR_IMPORT_ERROR)
            return False

        try:
            self.lidar = RPLidar(self.port, baudrate=self.baudrate, timeout=1)
            self.status = 'CONNECTED'
        except Exception as exc:  # pragma: no cover - runtime IO
            self.status = 'CONNECT_ERROR'
            self.error = str(exc)
            self.lidar = None
            return False

        self.running = True
        self.thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.thread.start()
        return True

    def _scan_loop(self):
        last_time = time.monotonic()
        fps_counter = 0
        fps_time = last_time
        try:
            for scan in self.lidar.iter_scans(max_buf_meas=2048):
                if not self.running:
                    break

                points = []
                front_min = float('inf')
                left_min = float('inf')
                right_min = float('inf')

                for _, angle_deg, distance_mm in scan:
                    distance_m = distance_mm / 1000.0
                    if distance_m < LIDAR_MIN_RANGE_M or distance_m > LIDAR_MAX_RANGE_M:
                        continue

                    angle_rad = math.radians(angle_deg)
                    x_pos = distance_m * math.cos(angle_rad)
                    y_pos = distance_m * math.sin(angle_rad)
                    points.append((x_pos, y_pos))

                    if abs(angle_rad) <= math.radians(25.0):
                        front_min = min(front_min, distance_m)
                    if math.radians(55.0) <= angle_rad <= math.radians(125.0):
                        left_min = min(left_min, distance_m)
                    if -math.radians(125.0) <= angle_rad <= -math.radians(55.0):
                        right_min = min(right_min, distance_m)

                points_np = np.array(points, dtype=np.float32)
                if points_np.size == 0:
                    continue

                points_np = voxel_downsample(points_np, LIDAR_DOWNSAMPLE_M)
                points_np = clip_points(points_np, LIDAR_MAX_POINTS)
                now = time.monotonic()

                fps_counter += 1
                if now - fps_time >= 1.0:
                    self.scan_rate_hz = fps_counter / (now - fps_time)
                    fps_counter = 0
                    fps_time = now

                snapshot = ScanSnapshot(
                    stamp=now,
                    points=points_np,
                    front_min_m=front_min if math.isfinite(front_min) else 99.0,
                    left_min_m=left_min if math.isfinite(left_min) else 99.0,
                    right_min_m=right_min if math.isfinite(right_min) else 99.0,
                    scan_rate_hz=self.scan_rate_hz,
                    raw_count=len(scan),
                )
                with self.lock:
                    self.latest_scan = snapshot
                    self.status = 'RUNNING'
                    self.error = None
                last_time = now
        except Exception as exc:  # pragma: no cover - runtime IO
            with self.lock:
                self.status = 'RUNTIME_ERROR'
                self.error = str(exc)
        finally:
            self.running = False

    def get_latest_scan(self):
        with self.lock:
            return self.latest_scan, self.status, self.error

    def stop(self):
        self.running = False
        if self.lidar is not None:
            try:
                self.lidar.stop()
            except Exception:
                pass
            try:
                self.lidar.disconnect()
            except Exception:
                pass


@dataclass
class ICPMatchResult:
    valid: bool
    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_theta: float = 0.0
    rmse: float = 999.0
    match_ratio: float = 0.0
    iterations: int = 0


class ICPScanMatcher:
    def __init__(self, max_iterations=12, max_correspondence=LIDAR_MATCH_MAX_DIST_M):
        self.max_iterations = max_iterations
        self.max_correspondence = max_correspondence

    def match(self, prev_points, curr_points, initial_guess):
        if prev_points.shape[0] < LIDAR_MATCH_MIN_POINTS or curr_points.shape[0] < LIDAR_MATCH_MIN_POINTS:
            return ICPMatchResult(valid=False)

        estimate = tuple(initial_guess)
        best_rmse = float('inf')
        best_ratio = 0.0

        for iteration in range(self.max_iterations):
            curr_aligned = transform_points(curr_points.copy(), estimate)
            dists = np.linalg.norm(curr_aligned[:, None, :] - prev_points[None, :, :], axis=2)
            nn_idx = np.argmin(dists, axis=1)
            nn_dist = dists[np.arange(dists.shape[0]), nn_idx]
            mask = nn_dist < self.max_correspondence

            matched_count = int(mask.sum())
            match_ratio = matched_count / max(curr_points.shape[0], 1)
            if matched_count < LIDAR_MATCH_MIN_POINTS:
                return ICPMatchResult(valid=False, match_ratio=match_ratio)

            src = curr_aligned[mask]
            dst = prev_points[nn_idx[mask]]
            delta = estimate_rigid_transform(src, dst)
            estimate = compose_delta(delta, estimate)

            best_rmse = float(np.sqrt(np.mean(np.square(nn_dist[mask]))))
            best_ratio = match_ratio

            if abs(delta[0]) < 1e-3 and abs(delta[1]) < 1e-3 and abs(delta[2]) < math.radians(0.2):
                break

        dx, dy, dtheta = estimate
        if abs(dx) > LIDAR_MAX_TRANSLATION_STEP_M or abs(dy) > LIDAR_MAX_TRANSLATION_STEP_M:
            return ICPMatchResult(valid=False, rmse=best_rmse, match_ratio=best_ratio)
        if abs(dtheta) > LIDAR_MAX_ROTATION_STEP_RAD:
            return ICPMatchResult(valid=False, rmse=best_rmse, match_ratio=best_ratio)

        return ICPMatchResult(
            valid=True,
            delta_x=dx,
            delta_y=dy,
            delta_theta=wrap_angle(dtheta),
            rmse=best_rmse,
            match_ratio=best_ratio,
            iterations=iteration + 1,
        )


class PoseFusionTracker:
    def __init__(self):
        self.corr_x = 0.0
        self.corr_y = 0.0
        self.corr_theta = 0.0
        self.prev_scan_odom = None
        self.last_quality = 0.0
        self.last_rmse = 999.0
        self.match_count = 0

    def current_pose(self, odom):
        return OdomState(
            x=odom.x + self.corr_x,
            y=odom.y + self.corr_y,
            theta=wrap_angle(odom.theta + self.corr_theta),
            vx=odom.vx,
            vy=odom.vy,
            wz=odom.wz,
            stamp=odom.stamp,
        )

    def bootstrap(self, odom):
        if self.prev_scan_odom is None:
            self.prev_scan_odom = OdomState(**odom.__dict__)

    def apply_scan_match(self, odom, icp_result):
        if self.prev_scan_odom is None:
            self.prev_scan_odom = OdomState(**odom.__dict__)
            return
        if not icp_result.valid:
            self.prev_scan_odom = OdomState(**odom.__dict__)
            self.last_quality = 0.0
            self.last_rmse = icp_result.rmse
            return

        odom_dx, odom_dy, odom_dtheta = world_to_local_delta(self.prev_scan_odom, odom)
        err_dx = icp_result.delta_x - odom_dx
        err_dy = icp_result.delta_y - odom_dy
        err_dtheta = wrap_angle(icp_result.delta_theta - odom_dtheta)

        quality = float(np.clip(
            0.55 * icp_result.match_ratio +
            0.45 * np.clip(1.0 - icp_result.rmse / max(LIDAR_MATCH_MAX_DIST_M, 1e-6), 0.0, 1.0),
            0.0,
            1.0,
        ))
        gain = LIDAR_QUALITY_BLEND * quality

        theta_ref = wrap_angle(self.prev_scan_odom.theta + self.corr_theta)
        ct = math.cos(theta_ref)
        st = math.sin(theta_ref)
        err_x_world = ct * err_dx - st * err_dy
        err_y_world = st * err_dx + ct * err_dy

        self.corr_x += gain * err_x_world
        self.corr_y += gain * err_y_world
        self.corr_theta = wrap_angle(self.corr_theta + gain * err_dtheta)

        self.prev_scan_odom = OdomState(**odom.__dict__)
        self.last_quality = quality
        self.last_rmse = icp_result.rmse
        self.match_count += 1


class ObstacleGuard:
    def apply(self, vx, vy, omega, scan):
        front_scale = 1.0
        side_scale = 1.0

        if vx > 0.0:
            if scan.front_min_m <= OBSTACLE_STOP_FRONT_M:
                front_scale = 0.0
            elif scan.front_min_m < OBSTACLE_SLOW_FRONT_M:
                front_scale = (scan.front_min_m - OBSTACLE_STOP_FRONT_M) / max(OBSTACLE_SLOW_FRONT_M - OBSTACLE_STOP_FRONT_M, 1e-6)

        if vy > 0.0:
            if scan.left_min_m <= OBSTACLE_STOP_SIDE_M:
                side_scale = 0.0
            elif scan.left_min_m < OBSTACLE_SLOW_SIDE_M:
                side_scale = (scan.left_min_m - OBSTACLE_STOP_SIDE_M) / max(OBSTACLE_SLOW_SIDE_M - OBSTACLE_STOP_SIDE_M, 1e-6)
        elif vy < 0.0:
            if scan.right_min_m <= OBSTACLE_STOP_SIDE_M:
                side_scale = 0.0
            elif scan.right_min_m < OBSTACLE_SLOW_SIDE_M:
                side_scale = (scan.right_min_m - OBSTACLE_STOP_SIDE_M) / max(OBSTACLE_SLOW_SIDE_M - OBSTACLE_STOP_SIDE_M, 1e-6)

        vx *= float(np.clip(front_scale, 0.0, 1.0))
        vy *= float(np.clip(side_scale, 0.0, 1.0))
        return vx, vy, omega, front_scale, side_scale


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Camera + LiDAR Tracker</title>
  <style>
    :root {
      --bg: #0c1118;
      --panel: #151d27;
      --line: #2a3643;
      --text: #edf3f8;
      --muted: #95a8ba;
      --accent: #f0c36c;
      --ok: #88ef95;
      --warn: #ffd36d;
    }
    body {
      margin: 0;
      background: radial-gradient(circle at top left, #213246 0%, #0c1118 58%);
      color: var(--text);
      font-family: Consolas, 'Courier New', monospace;
    }
    .wrap {
      max-width: 1320px;
      margin: 0 auto;
      padding: 20px;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 16px;
    }
    .title {
      font-size: 26px;
      letter-spacing: 0.06em;
    }
    .hint {
      color: var(--muted);
      font-size: 13px;
    }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 12px;
    }
    .button {
      appearance: none;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #253241 0%, #151d27 100%);
      color: var(--text);
      padding: 10px 14px;
      border-radius: 999px;
      cursor: pointer;
      font: inherit;
      font-size: 13px;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 16px;
    }
    .panel {
      background: rgba(21, 29, 39, 0.92);
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
    }
    .stream {
      width: 100%;
      display: block;
      background: #05070a;
    }
    .stats {
      padding: 14px;
    }
    .stats h2 {
      margin: 0 0 12px;
      font-size: 14px;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .kv {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 8px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      font-size: 13px;
    }
    .label {
      color: var(--muted);
    }
    .ok {
      color: var(--ok);
    }
    .warn {
      color: var(--warn);
    }
    @media (max-width: 960px) {
      .grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div>
        <div class="title">Camera + LiDAR Tracker</div>
        <div class="hint">Tracker dung camera, pose duoc fusion tu odom base va LiDAR scan matching.</div>
        <div class="actions">
          <button class="button" id="startBtn">Start Motor</button>
          <button class="button" id="stopBtn">Stop Motor</button>
        </div>
      </div>
      <div class="hint" id="endpoint"></div>
    </div>
    <div class="grid">
      <div class="panel">
        <img class="stream" src="/stream.mjpg" alt="stream">
      </div>
      <div class="panel stats">
        <h2>Live State</h2>
        <div id="stats"></div>
      </div>
    </div>
  </div>
  <script>
    const statsEl = document.getElementById('stats');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    document.getElementById('endpoint').textContent = `${location.origin}/stream.mjpg`;

    function row(label, value, cls = '') {
      return `<div class="kv"><div class="label">${label}</div><div class="${cls}">${value}</div></div>`;
    }

    async function refresh() {
      try {
        const response = await fetch('/state');
        const state = await response.json();
        startBtn.disabled = !state.serial_enabled || state.motor_enabled;
        stopBtn.disabled = !state.serial_enabled || !state.motor_enabled;
        statsEl.innerHTML = [
          row('status', state.status, state.ball_detected ? 'ok' : 'warn'),
          row('serial', state.serial_enabled ? 'enabled' : 'disabled'),
          row('motor', state.motor_enabled ? 'armed' : 'stopped', state.motor_enabled ? 'ok' : 'warn'),
          row('lidar', state.lidar_status, state.lidar_ok ? 'ok' : 'warn'),
          row('pose x', state.pose_x.toFixed(3)),
          row('pose y', state.pose_y.toFixed(3)),
          row('pose th', state.pose_theta_deg.toFixed(1)),
          row('front obs', state.obstacle_front_m.toFixed(2)),
          row('left obs', state.obstacle_left_m.toFixed(2)),
          row('right obs', state.obstacle_right_m.toFixed(2)),
          row('lidar q', state.lidar_quality.toFixed(2)),
          row('lidar rmse', state.lidar_rmse.toFixed(3)),
          row('lidar fps', state.lidar_fps.toFixed(1)),
          row('vx', state.vx.toFixed(3)),
          row('vy', state.vy.toFixed(3)),
          row('omega', state.omega.toFixed(3)),
          row('dist m', state.dist_m.toFixed(3)),
          row('base x', state.robot_x.toFixed(3)),
          row('base y', state.robot_y.toFixed(3)),
          row('base th', state.robot_theta_deg.toFixed(1)),
          row('vision fps', state.fps_vision.toFixed(1)),
          row('control fps', state.fps_ctrl.toFixed(1)),
        ].join('');
      } catch (error) {
        statsEl.innerHTML = row('state', 'fetch error', 'warn');
      }
    }

    async function postAction(url) {
      await fetch(url, { method: 'POST' });
      await refresh();
    }

    startBtn.addEventListener('click', () => postAction('/api/motor/start'));
    stopBtn.addEventListener('click', () => postAction('/api/motor/stop'));

    refresh();
    setInterval(refresh, 300);
  </script>
</body>
</html>
"""


class SharedAppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.motor_enabled = False
        self.serial_mgr = None
        self.ball_detected = False
        self.last_seen_time = 0.0
        self.raw_cx = 0.0
        self.raw_cy = 0.0
        self.raw_bbox_h = 0.0
        self.raw_bbox_w = 0.0
        self.raw_bbox = (0.0, 0.0, 0.0, 0.0)
        self.new_measurement = False
        self.display_frame = None
        self.latest_jpeg = None
        self.fps_vision = 0.0
        self.fps_ctrl = 0.0
        self.stats = {
            'status': 'SEARCHING',
            'ball_detected': False,
            'serial_enabled': False,
            'motor_enabled': False,
            'motor_reason': 'Serial disabled',
            'vx': 0.0,
            'vy': 0.0,
            'omega': 0.0,
            'err_x': 0.0,
            'err_dist': 0.0,
            'dist_m': 0.0,
            'fps_vision': 0.0,
            'fps_ctrl': 0.0,
            'measured_vx': 0.0,
            'measured_vy': 0.0,
            'measured_wz': 0.0,
            'robot_x': 0.0,
            'robot_y': 0.0,
            'robot_theta_deg': 0.0,
            'pose_x': 0.0,
            'pose_y': 0.0,
            'pose_theta_deg': 0.0,
            'lidar_status': 'DISABLED',
            'lidar_ok': False,
            'lidar_quality': 0.0,
            'lidar_rmse': 999.0,
            'lidar_fps': 0.0,
            'obstacle_front_m': 99.0,
            'obstacle_left_m': 99.0,
            'obstacle_right_m': 99.0,
            'last_seen_age': 0.0,
            'hist_err_x': [],
            'hist_err_dist': [],
            'hist_omega': [],
            'hist_vx': [],
        }


class WebHandler(server.BaseHTTPRequestHandler):
    state = None

    def do_GET(self):
        if self.path == '/':
            self._send_html()
            return
        if self.path == '/state':
            self._send_state()
            return
        if self.path == '/stream.mjpg':
            self._send_stream()
            return
        self.send_error(404)

    def do_POST(self):
        if self.path == '/api/motor/start':
            self._set_motor_enabled(True)
            return
        if self.path == '/api/motor/stop':
            self._set_motor_enabled(False)
            return
        self.send_error(404)

    def log_message(self, fmt, *args):
        return

    def _send_html(self):
        body = HTML_PAGE.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_state(self):
        with self.state.lock:
            payload = dict(self.state.stats)
        body = json.dumps(payload).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self):
        self.send_response(200)
        self.send_header('Age', '0')
        self.send_header('Cache-Control', 'no-cache, private')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()

        last_payload = None
        try:
            while self.state.running:
                with self.state.lock:
                    payload = self.state.latest_jpeg
                if payload is None:
                    time.sleep(0.05)
                    continue
                if payload is last_payload:
                    time.sleep(0.03)
                    continue
                last_payload = payload
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n')
                self.wfile.write(f'Content-Length: {len(payload)}\r\n\r\n'.encode('ascii'))
                self.wfile.write(payload)
                self.wfile.write(b'\r\n')
        except (BrokenPipeError, ConnectionResetError):
            return

    def _set_motor_enabled(self, enabled):
        with self.state.lock:
            serial_ready = self.state.stats.get('serial_enabled', False)
            self.state.motor_enabled = enabled and serial_ready
            self.state.stats['motor_enabled'] = self.state.motor_enabled
            if not serial_ready:
                self.state.stats['motor_reason'] = 'Serial not connected'
            elif enabled:
                self.state.stats['motor_reason'] = 'Motor armed from web UI'
            else:
                self.state.stats['motor_reason'] = 'Stopped from web UI'
            payload = {
                'ok': True,
                'motor_enabled': self.state.motor_enabled,
                'reason': self.state.stats['motor_reason'],
            }
        if not enabled and self.state.serial_mgr is not None:
            self.state.serial_mgr.send_stop()

        body = json.dumps(payload).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ThreadedHTTPServer(ThreadingMixIn, server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def update_web_frame(state, frame):
    success, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not success:
        return
    with state.lock:
        state.latest_jpeg = encoded.tobytes()


def draw_topdown_overlay(frame, scan, fused_pose, path_history):
    inset_size = 220
    margin = 12
    x0 = frame.shape[1] - inset_size - margin
    y0 = frame.shape[0] - inset_size - margin
    cv2.rectangle(frame, (x0, y0), (x0 + inset_size, y0 + inset_size), (18, 24, 30), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + inset_size, y0 + inset_size), (70, 90, 105), 1)

    center = np.array([x0 + inset_size // 2, y0 + inset_size // 2], dtype=np.float32)
    scale = 45.0

    if scan is not None and scan.points.size:
        for point in scan.points:
            px = int(center[0] + point[1] * scale)
            py = int(center[1] - point[0] * scale)
            if x0 <= px < x0 + inset_size and y0 <= py < y0 + inset_size:
                frame[py, px] = (0, 220, 255)

    if len(path_history) >= 2:
        pts = []
        origin_x = fused_pose.x
        origin_y = fused_pose.y
        for pose in path_history:
            rel_x = pose[0] - origin_x
            rel_y = pose[1] - origin_y
            px = int(center[0] + rel_y * scale)
            py = int(center[1] - rel_x * scale)
            pts.append((px, py))
        for idx in range(1, len(pts)):
            cv2.line(frame, pts[idx - 1], pts[idx], (100, 255, 100), 1)

    cv2.circle(frame, tuple(center.astype(int)), 5, (255, 255, 255), -1)
    heading_tip = (
        int(center[0] + math.sin(fused_pose.theta) * 20.0),
        int(center[1] - math.cos(fused_pose.theta) * 20.0),
    )
    cv2.arrowedLine(frame, tuple(center.astype(int)), heading_tip, (0, 120, 255), 2, tipLength=0.35)
    cv2.putText(frame, 'LiDAR local map', (x0 + 8, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)


def draw_lidar_text(frame, info):
    x0 = 12
    y0 = frame.shape[0] - 90
    lines = [
        f"pose=({info['pose_x']:+.2f}, {info['pose_y']:+.2f}) th={info['pose_theta_deg']:+.1f}deg",
        f"lidar={info['lidar_status']} q={info['lidar_quality']:.2f} rmse={info['lidar_rmse']:.3f} fps={info['lidar_fps']:.1f}",
        f"obs front={info['obstacle_front_m']:.2f} left={info['obstacle_left_m']:.2f} right={info['obstacle_right_m']:.2f}",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(frame, line, (x0, y0 + idx * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)


def build_odom_state(serial_mgr):
    now = time.monotonic()
    if serial_mgr is None:
        return OdomState(stamp=now)
    return OdomState(
        x=serial_mgr.robot_x,
        y=serial_mgr.robot_y,
        theta=math.radians(serial_mgr.robot_theta_deg),
        vx=serial_mgr.measured_vx,
        vy=serial_mgr.measured_vy,
        wz=serial_mgr.measured_wz,
        stamp=serial_mgr.last_telm_time or now,
    )


def find_lidar_port():
    import glob

    candidates = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(description='Camera + LiDAR web tracker')
    parser.add_argument('--camera', type=int, default=base.CAMERA_INDEX)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--target-h', type=int, default=base.TARGET_BBOX_H)
    parser.add_argument('--model', type=str, default=base.YOLO_MODEL)
    parser.add_argument('--conf', type=float, default=base.YOLO_CONF)
    parser.add_argument('--imgsz', type=int, default=base.YOLO_IMGSZ)
    parser.add_argument('--classes', type=int, nargs='+', default=base.YOLO_CLASSES)
    parser.add_argument('--kalman-r', type=float, default=base.KALMAN_R)
    parser.add_argument('--send-serial', action='store_true')
    parser.add_argument('--no-serial', action='store_true')
    parser.add_argument('--serial-port', type=str, default=None)
    parser.add_argument('--lidar-port', type=str, default=None)
    parser.add_argument('--lidar-baud', type=int, default=115200)
    parser.add_argument('--no-lidar', action='store_true')
    args = parser.parse_args()

    app_state = SharedAppState()
    serial_mgr = None
    use_serial = not args.no_serial
    if args.send_serial:
        use_serial = True

    if use_serial:
        port = args.serial_port or base.find_serial_port()
        if port:
            serial_mgr = base.SerialManager(port)
            if not serial_mgr.connect():
                serial_mgr = None
        else:
            print('[Serial] No Arduino port found. Running in monitor-only mode.')
    app_state.serial_mgr = serial_mgr
    app_state.stats['serial_enabled'] = serial_mgr is not None
    app_state.motor_enabled = serial_mgr is not None
    app_state.stats['motor_enabled'] = app_state.motor_enabled
    app_state.stats['motor_reason'] = 'Auto armed on startup' if serial_mgr is not None else 'Serial unavailable; check cable/port or use --serial-port'

    lidar_mgr = None
    if not args.no_lidar:
        lidar_port = args.lidar_port or find_lidar_port()
        lidar_mgr = LidarManager(lidar_port, baudrate=args.lidar_baud)
        if lidar_port and lidar_mgr.start():
            print(f'[LiDAR] Connected on {lidar_port}')
        elif lidar_port:
            print(f'[LiDAR] Failed on {lidar_port}: {lidar_mgr.error}')
        else:
            print('[LiDAR] No LiDAR port found. Running without LiDAR.')

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, base.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, base.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print('[Camera] Cannot open camera')
        if lidar_mgr:
            lidar_mgr.stop()
        return

    detector = base.FaceDetector(model_path=args.model, conf=args.conf, imgsz=args.imgsz, classes=args.classes)
    pid_omega = base.PID(base.OMEGA_KP, base.OMEGA_KI, base.OMEGA_KD, -base.OMEGA_MAX, base.OMEGA_MAX, derivative_tau=0.06)
    pid_vy = base.PID(base.VY_KP, base.VY_KI, base.VY_KD, -base.VY_ALIGN_MAX, base.VY_ALIGN_MAX, derivative_tau=0.08)
    pid_vx = base.PID(base.VX_KP, base.VX_KI, base.VX_KD, -base.VX_MAX, base.VX_MAX, derivative_tau=0.10)
    vx_limiter = base.SlewRateLimiter(base.VX_SLEW_MPS_S)
    vy_limiter = base.SlewRateLimiter(base.VY_SLEW_MPS_S)
    omega_limiter = base.SlewRateLimiter(base.OMEGA_SLEW_RAD_S2)
    kf_cx = base.Kalman1D(q=base.KALMAN_Q, r=args.kalman_r)
    kf_bh = base.Kalman1D(q=base.KALMAN_Q, r=args.kalman_r * 2.0)
    frame_center_x = base.FRAME_WIDTH / 2.0

    hist_err_x = deque(maxlen=base.HIST_LEN)
    hist_err_dist = deque(maxlen=base.HIST_LEN)
    hist_omega = deque(maxlen=base.HIST_LEN)
    hist_vx = deque(maxlen=base.HIST_LEN)
    path_history = deque(maxlen=PATH_LEN)
    recent_scans = deque(maxlen=SCAN_HISTORY_LEN)

    matcher = ICPScanMatcher()
    fusion = PoseFusionTracker()
    obstacle_guard = ObstacleGuard()

    def vision_loop():
        fps_counter = 0
        fps_time = time.monotonic()
        while app_state.running:
            ret, frame = cap.read()
            if not ret:
                continue
            result = detector.detect(frame)
            with app_state.lock:
                if result is not None:
                    cx, cy, bbox_h, bbox_w, bbox = result
                    app_state.ball_detected = True
                    app_state.last_seen_time = time.monotonic()
                    app_state.raw_cx = cx
                    app_state.raw_cy = cy
                    app_state.raw_bbox_h = bbox_h
                    app_state.raw_bbox_w = bbox_w
                    app_state.raw_bbox = bbox
                    app_state.new_measurement = True
                app_state.display_frame = frame

            fps_counter += 1
            now = time.monotonic()
            if now - fps_time >= 1.0:
                with app_state.lock:
                    app_state.fps_vision = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

    WebHandler.state = app_state
    httpd = ThreadedHTTPServer((args.host, args.port), WebHandler)
    web_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    web_thread.start()
    vision_thread = threading.Thread(target=vision_loop, daemon=True)
    vision_thread.start()

    print('\n=== CAMERA + LiDAR TRACKER ===')
    print(f'[Web] Open: http://{args.host if args.host != "0.0.0.0" else "<pi-ip>"}:{args.port}')
    if serial_mgr:
        print('[Serial] Sending commands to Arduino')
    else:
        print('[Serial] Monitor-only mode')
    if lidar_mgr and lidar_mgr.status not in ('DISABLED', 'CONNECT_ERROR', 'IMPORT_ERROR'):
        print('[LiDAR] Scan matching + pose fusion enabled')
    else:
        print('[LiDAR] Running without active LiDAR correction')
    print('Press Ctrl+C to stop.\n')

    ctrl_period = 1.0 / base.CONTROL_RATE
    last_kf_time = time.monotonic()
    ctrl_fps_counter = 0
    ctrl_fps_time = time.monotonic()
    lost_since = None
    last_scan_stamp = 0.0

    try:
        while app_state.running:
            t_start = time.monotonic()
            dt = t_start - last_kf_time
            if dt <= 0:
                dt = ctrl_period
            last_kf_time = t_start

            if serial_mgr:
                serial_mgr.read_feedback()
            odom_state = build_odom_state(serial_mgr)
            fusion.bootstrap(odom_state)
            fused_pose = fusion.current_pose(odom_state)

            latest_scan = None
            lidar_status = 'DISABLED'
            lidar_error = None
            if lidar_mgr is not None:
                latest_scan, lidar_status, lidar_error = lidar_mgr.get_latest_scan()
                if latest_scan is not None and latest_scan.stamp > last_scan_stamp:
                    last_scan_stamp = latest_scan.stamp
                    recent_scans.append(latest_scan)
                    if len(recent_scans) >= 2:
                        prev_scan = recent_scans[-2]
                        curr_scan = recent_scans[-1]
                        odom_guess = world_to_local_delta(fusion.prev_scan_odom or odom_state, odom_state)
                        icp_result = matcher.match(prev_scan.points, curr_scan.points, odom_guess)
                        fusion.apply_scan_match(odom_state, icp_result)
                        fused_pose = fusion.current_pose(odom_state)
                elif latest_scan is not None and (t_start - latest_scan.stamp) > LIDAR_STALE_TIMEOUT:
                    lidar_status = 'STALE'

            with app_state.lock:
                ball_detected = app_state.ball_detected
                last_seen = app_state.last_seen_time
                has_new = app_state.new_measurement
                raw_cx = app_state.raw_cx
                raw_cy = app_state.raw_cy
                raw_bh = app_state.raw_bbox_h
                raw_bw = app_state.raw_bbox_w
                raw_bbox = app_state.raw_bbox
                display_frame = None if app_state.display_frame is None else app_state.display_frame.copy()
                fps_vision = app_state.fps_vision
                app_state.new_measurement = False

            vx_target = 0.0
            vy_target = 0.0
            omega_target = 0.0
            vx = 0.0
            vy = 0.0
            omega = 0.0
            err_x = 0.0
            err_dist = 0.0
            dist_m = 0.0
            vx_ff = 0.0
            omega_ff = 0.0

            if ball_detected:
                lost_since = None
                if has_new:
                    filt_cx = kf_cx.update(raw_cx, dt)
                    filt_bh = kf_bh.update(raw_bh, dt)
                else:
                    filt_cx = kf_cx.predict(dt)
                    filt_bh = kf_bh.predict(dt)

                err_x = (filt_cx - frame_center_x) / frame_center_x
                bh_safe = max(filt_bh, 5.0)
                dist_m = (base.REAL_FACE_HEIGHT_M * base.CAMERA_FOCAL_PX) / bh_safe
                err_dist_m = dist_m - base.TARGET_DISTANCE_M
                err_dist = float(np.clip(err_dist_m / max(base.TARGET_DISTANCE_M, 1e-3), -1.0, 1.0))

                bearing_rate = kf_cx.dx / base.CAMERA_FOCAL_PX
                omega_ff = -base.OMEGA_FF_GAIN * bearing_rate
                dist_rate = -base.REAL_FACE_HEIGHT_M * base.CAMERA_FOCAL_PX * kf_bh.dx / (bh_safe * bh_safe)
                vx_ff = base.VX_FF_GAIN * dist_rate

                align_err = abs(err_x)
                turn_blend = float(np.clip(
                    (align_err - base.SMALL_ERR_X_FOR_VY) /
                    max(base.LARGE_ERR_X_FOR_WZ - base.SMALL_ERR_X_FOR_VY, 1e-6),
                    0.0,
                    1.0,
                ))

                if align_err <= base.CENTER_DEADBAND:
                    pid_vy.reset()
                    pid_omega.reset()
                else:
                    settle_err_x = math.copysign(align_err - base.CENTER_DEADBAND, err_x)
                    settle_scale = float(np.clip(
                        (align_err - base.CENTER_DEADBAND) /
                        max(base.SMALL_ERR_X_FOR_VY - base.CENTER_DEADBAND, 1e-6),
                        0.0,
                        1.0,
                    ))
                    if turn_blend < 0.98:
                        vy_error = -settle_err_x * (0.25 + 0.75 * settle_scale)
                        vy_ff = -base.VY_ALIGN_DAMP_GAIN * bearing_rate
                        vy_cmd = pid_vy.compute(vy_error, dt, feedforward=vy_ff)
                        vy_target = float(np.clip(vy_cmd * (1.0 - turn_blend), -base.VY_ALIGN_MAX, base.VY_ALIGN_MAX))
                    else:
                        pid_vy.reset()

                    if turn_blend > 0.0:
                        omega_raw = -pid_omega.compute(err_x, dt, feedforward=-omega_ff)
                        omega_target = float(np.clip(omega_raw * turn_blend, -base.OMEGA_ALIGN_MAX, base.OMEGA_ALIGN_MAX))
                    else:
                        pid_omega.reset()

                if abs(err_dist_m) < base.DIST_DEADBAND_M:
                    vx_target = float(np.clip(vx_ff, -base.VX_MAX, base.VX_MAX))
                    pid_vx.reset()
                else:
                    vx_target = pid_vx.compute(err_dist, dt, feedforward=vx_ff)

                if err_dist_m < -base.DIST_DEADBAND_M:
                    backoff_speed = min(base.CLOSE_BACKOFF_MAX, base.CLOSE_BACKOFF_GAIN * (-err_dist_m))
                    vx_target = min(vx_target, -backoff_speed)

                if align_err >= base.FORWARD_ALIGN_LIMIT:
                    alignment = 0.0
                else:
                    scaled_align = align_err / max(base.FORWARD_ALIGN_LIMIT, 1e-6)
                    cos_gate = math.cos(0.5 * math.pi * scaled_align)
                    alignment = base.FORWARD_ALIGN_MIN_GAIN + (1.0 - base.FORWARD_ALIGN_MIN_GAIN) * (cos_gate * cos_gate)
                if vx_target > 0.0:
                    vx_target *= alignment

                seen_age = max(0.0, time.monotonic() - last_seen)
                visibility = float(np.clip(1.0 - seen_age / base.LOST_TIMEOUT, 0.0, 1.0))
                vx_target *= visibility
                vy_target *= visibility
                omega_target *= 0.35 + 0.65 * visibility

                if seen_age > base.LOST_TIMEOUT:
                    with app_state.lock:
                        app_state.ball_detected = False
                    ball_detected = False
                    lost_since = time.monotonic()

            if not ball_detected:
                now_lost = time.monotonic()
                if lost_since is None:
                    lost_since = now_lost
                lost_duration = now_lost - lost_since

                if lost_duration < base.LOST_COAST_TIME:
                    vx_target = vx_limiter.value
                    vy_target = vy_limiter.value
                    omega_target = omega_limiter.value
                else:
                    vx_target = 0.0
                    vy_target = 0.0
                    omega_target = 0.0

                if lost_duration > base.LOST_HARD_RESET:
                    pid_omega.reset()
                    pid_vy.reset()
                    pid_vx.reset()
                    kf_cx.reset()
                    kf_bh.reset()

            vx = vx_limiter.update(vx_target, dt)
            vy = vy_limiter.update(vy_target, dt)
            omega = omega_limiter.update(omega_target, dt)

            front_min = 99.0
            left_min = 99.0
            right_min = 99.0
            if latest_scan is not None:
                vx, vy, omega, _, _ = obstacle_guard.apply(vx, vy, omega, latest_scan)
                front_min = latest_scan.front_min_m
                left_min = latest_scan.left_min_m
                right_min = latest_scan.right_min_m

            if serial_mgr and app_state.motor_enabled:
                serial_mgr.send_velocity(vx, vy, omega)
            elif serial_mgr and not app_state.motor_enabled:
                pid_omega.reset()
                pid_vy.reset()
                pid_vx.reset()
                vx_limiter.reset(0.0)
                vy_limiter.reset(0.0)
                omega_limiter.reset(0.0)
                vx = 0.0
                vy = 0.0
                omega = 0.0

            fused_pose = fusion.current_pose(build_odom_state(serial_mgr))
            path_history.append((fused_pose.x, fused_pose.y, fused_pose.theta))

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

            if display_frame is not None:
                frame = display_frame
                if ball_detected:
                    x1, y1, x2, y2 = raw_bbox
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (int(raw_cx), int(raw_cy)), 4, (255, 150, 0), -1)
                    if kf_cx.initialized:
                        filt_y = raw_cy
                        cv2.circle(frame, (int(kf_cx.x), int(filt_y)), 8, (0, 0, 255), 2)
                    if kf_bh.initialized:
                        filt_half_h = kf_bh.x / 2.0
                        filt_half_w = raw_bw / 2.0
                        fx1 = int(kf_cx.x - filt_half_w)
                        fy1 = int(raw_cy - filt_half_h)
                        fx2 = int(kf_cx.x + filt_half_w)
                        fy2 = int(raw_cy + filt_half_h)
                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 1)

                center_x = int(frame_center_x)
                center_y = int(base.FRAME_HEIGHT / 2)
                cv2.line(frame, (center_x - 25, center_y), (center_x + 25, center_y), (100, 100, 100), 1)
                cv2.line(frame, (center_x, center_y - 25), (center_x, center_y + 25), (100, 100, 100), 1)
                cv2.circle(frame, (center_x, center_y), 30, (100, 100, 100), 1)

                last_seen_age = max(0.0, time.monotonic() - last_seen) if last_seen > 0 else 0.0
                lidar_ok = latest_scan is not None and lidar_status in ('RUNNING', 'CONNECTED')
                info = {
                    'status': 'TRACKING' if ball_detected else 'SEARCHING',
                    'ball_detected': ball_detected,
                    'vx': vx,
                    'vy': vy,
                    'omega': omega,
                    'err_x': err_x,
                    'err_dist': err_dist,
                    'dist_m': dist_m,
                    'measured_vx': serial_mgr.measured_vx if serial_mgr else 0.0,
                    'measured_vy': serial_mgr.measured_vy if serial_mgr else 0.0,
                    'measured_wz': serial_mgr.measured_wz if serial_mgr else 0.0,
                    'robot_x': serial_mgr.robot_x if serial_mgr else 0.0,
                    'robot_y': serial_mgr.robot_y if serial_mgr else 0.0,
                    'robot_theta_deg': serial_mgr.robot_theta_deg if serial_mgr else 0.0,
                    'pose_x': fused_pose.x,
                    'pose_y': fused_pose.y,
                    'pose_theta_deg': math.degrees(fused_pose.theta),
                    'lidar_status': lidar_status if lidar_error is None else f'{lidar_status}: {lidar_error}',
                    'lidar_ok': lidar_ok,
                    'lidar_quality': fusion.last_quality,
                    'lidar_rmse': fusion.last_rmse,
                    'lidar_fps': latest_scan.scan_rate_hz if latest_scan is not None else 0.0,
                    'obstacle_front_m': front_min,
                    'obstacle_left_m': left_min,
                    'obstacle_right_m': right_min,
                    'fps_vision': fps_vision,
                    'fps_ctrl': app_state.fps_ctrl,
                    'last_seen_age': last_seen_age,
                    'hist_err_x': list(hist_err_x),
                    'hist_err_dist': list(hist_err_dist),
                    'hist_omega': list(hist_omega),
                    'hist_vx': list(hist_vx),
                }
                base.draw_dashboard(frame, info)
                draw_topdown_overlay(frame, latest_scan, fused_pose, path_history)
                draw_lidar_text(frame, info)
                update_web_frame(app_state, frame)

                with app_state.lock:
                    app_state.stats.update(info)
                    app_state.stats['serial_enabled'] = serial_mgr is not None
                    app_state.stats['motor_enabled'] = app_state.motor_enabled
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
        if serial_mgr:
            serial_mgr.send_stop()
            serial_mgr.close()
        if lidar_mgr:
            lidar_mgr.stop()
        cap.release()
        print('[Exit] Stopped.')


if __name__ == '__main__':
    main()