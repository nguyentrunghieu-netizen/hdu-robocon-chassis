#!/usr/bin/env python3
"""
Ball Chaser Web Test
====================
Test pipeline vision + Kalman + PID tren Raspberry Pi va hien thi qua web.

Tinh nang:
  - Chay headless, khong can man hinh tren Pi
  - Stream MJPEG de xem frame da annotate tren trinh duyet
  - Trang web hien thong so PID, Kalman, FPS, bbox, err_x, err_dist
  - Tuy chon gui lenh Serial xuong Arduino de test thuc te

Su dung:
  python3 ball_chaser_web_test.py
  python3 ball_chaser_web_test.py --host 0.0.0.0 --port 8080
  python3 ball_chaser_web_test.py --send-serial --serial-port /dev/ttyACM0
"""

import argparse
import json
import math
import re
import sys
import threading
import time
from collections import deque
from http import server
from socketserver import ThreadingMixIn

import cv2
import numpy as np
import serial
from ultralytics import YOLO


# ======================== CAU HINH ========================

YOLO_MODEL = 'yolov12n-face.pt'
YOLO_CONF = 0.45
YOLO_IMGSZ = 320
YOLO_CLASSES = None

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CAMERA_FOCAL_PX = 500.0
REAL_FACE_HEIGHT_M = 0.22
TARGET_DISTANCE_M = 0.8
DIST_DEADBAND_M = 0.10

TARGET_BBOX_H = 120
MIN_BBOX_H = 30
DIST_DEADBAND = 0.12
CENTER_DEADBAND = 0.05
FORWARD_ALIGN_LIMIT = 0.28
FORWARD_ALIGN_MIN_GAIN = 0.18

VX_MAX = 0.35
VY_MAX = 0.25
OMEGA_MAX = 1.5

VY_ALIGN_MAX = 0.22
OMEGA_ALIGN_MAX = 0.28
VY_KP = 0.95
VY_KI = 0.04
VY_KD = 0.10
VY_ALIGN_DAMP_GAIN = 0.18
SMALL_ERR_X_FOR_VY = 0.12
LARGE_ERR_X_FOR_WZ = 0.30

CLOSE_BACKOFF_GAIN = 1.30
CLOSE_BACKOFF_MAX = 0.22

VX_SLEW_MPS_S = 0.90
VY_SLEW_MPS_S = 1.00
OMEGA_SLEW_RAD_S2 = 4.00

OMEGA_KP = 2.0
OMEGA_KI = 0.1
OMEGA_KD = 0.3

VX_KP = 0.8
VX_KI = 0.05
VX_KD = 0.15

OMEGA_FF_GAIN = 0.6
VX_FF_GAIN = 0.5

LOST_COAST_TIME = 0.3
LOST_SOFT_STOP = 0.8
LOST_HARD_RESET = 1.5

LOST_TIMEOUT = 0.8
CONTROL_RATE = 40

KALMAN_Q = 3.0
KALMAN_R = 1.0
KALMAN_VEL_DECAY = 0.95

HIST_LEN = 150


# ======================== KALMAN ========================

class Kalman1D:
    def __init__(self, q=KALMAN_Q, r=KALMAN_R):
        self.x = 0.0
        self.dx = 0.0
        self.P = [[1.0, 0.0], [0.0, 1.0]]
        self.q = q
        self.r = r
        self.initialized = False

    def reset(self):
        self.x = 0.0
        self.dx = 0.0
        self.P = [[1.0, 0.0], [0.0, 1.0]]
        self.initialized = False

    def init_with(self, value):
        self.x = value
        self.dx = 0.0
        self.P = [[1.0, 0.0], [0.0, 1.0]]
        self.initialized = True

    def predict(self, dt):
        if not self.initialized:
            return self.x

        self.dx *= KALMAN_VEL_DECAY
        self.x += dt * self.dx

        dt2 = dt * dt
        q = self.q
        self.P[0][0] += dt * (self.P[0][1] + self.P[1][0]) + dt2 * self.P[1][1] + q * dt2 * dt2 / 4.0
        self.P[0][1] += dt * self.P[1][1] + q * dt2 * dt / 2.0
        self.P[1][0] += dt * self.P[1][1] + q * dt2 * dt / 2.0
        self.P[1][1] += q * dt2
        return self.x

    def update(self, measurement, dt):
        if not self.initialized:
            self.init_with(measurement)
            return self.x

        self.predict(dt)

        innovation = measurement - self.x
        s_val = self.P[0][0] + self.r
        k0 = self.P[0][0] / s_val
        k1 = self.P[1][0] / s_val

        self.x += k0 * innovation
        self.dx += k1 * innovation

        p00, p01 = self.P[0][0], self.P[0][1]
        p10, p11 = self.P[1][0], self.P[1][1]
        self.P[0][0] = (1 - k0) * p00
        self.P[0][1] = (1 - k0) * p01
        self.P[1][0] = p10 - k1 * p00
        self.P[1][1] = p11 - k1 * p01
        return self.x


class PID:
    def __init__(self, kp, ki, kd, out_min, out_max, derivative_tau=0.08):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.derivative_tau = derivative_tau
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        self.d_state = 0.0
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        self.d_state = 0.0
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0

    def compute(self, error, dt=None, feedforward=0.0):
        now = time.monotonic()
        if dt is None:
            if self.prev_time is None:
                dt = 1.0 / CONTROL_RATE
            else:
                dt = now - self.prev_time
            if dt <= 0:
                dt = 0.001
        self.prev_time = now

        self.last_p = self.kp * error
        raw_d = (error - self.prev_error) / max(dt, 1e-3)
        alpha = dt / (self.derivative_tau + dt)
        self.d_state += alpha * (raw_d - self.d_state)
        self.last_d = self.kd * self.d_state

        if abs(error) < 0.02 and abs(self.prev_error) < 0.02:
            self.integral *= 0.85

        tentative_integral = self.integral + error * dt
        tentative_i = self.ki * tentative_integral
        u_pre = self.last_p + tentative_i + self.last_d + feedforward
        u_sat = float(np.clip(u_pre, self.out_min, self.out_max))
        allow_integral = abs(u_pre - u_sat) < 1e-6 or np.sign(error) != np.sign(u_pre - u_sat)

        if allow_integral:
            self.integral = tentative_integral

        max_integral = abs(self.out_max / (self.ki + 1e-9))
        self.integral = float(np.clip(self.integral, -max_integral, max_integral))
        self.last_i = self.ki * self.integral
        self.prev_error = error

        output = self.last_p + self.last_i + self.last_d + feedforward
        return float(np.clip(output, self.out_min, self.out_max))


class SlewRateLimiter:
    def __init__(self, rate_limit):
        self.rate_limit = rate_limit
        self.value = 0.0
        self.initialized = True

    def reset(self, value=0.0):
        self.value = float(value)
        self.initialized = True

    def update(self, target, dt):
        if not self.initialized:
            self.value = float(target)
            self.initialized = True
            return self.value

        max_delta = self.rate_limit * max(dt, 1e-3)
        delta = float(np.clip(target - self.value, -max_delta, max_delta))
        self.value += delta
        return self.value


class FaceDetector:
    def __init__(self, model_path=YOLO_MODEL, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, classes=YOLO_CLASSES):
        print(f"[YOLO] Model: {model_path} | conf={conf} | imgsz={imgsz}")
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.classes = classes

    def detect(self, frame):
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=self.classes,
            verbose=False,
        )

        if not results:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        best_idx = int(areas.argmax())

        x1, y1, x2, y2 = xyxy[best_idx]
        box_w = x2 - x1
        box_h = y2 - y1
        if box_h < MIN_BBOX_H:
            return None

        center_x = x1 + box_w / 2.0
        center_y = y1 + box_h / 2.0
        return (
            float(center_x),
            float(center_y),
            float(box_h),
            float(box_w),
            (float(x1), float(y1), float(x2), float(y2)),
        )


class SerialManager:
    _TELM_RE = re.compile(
        r'^T\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)'
        r'\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)'
    )

    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.measured_vx = 0.0
        self.measured_vy = 0.0
        self.measured_wz = 0.0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta_deg = 0.0
        self.last_telm_time = 0.0

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.05)
            time.sleep(2)
            self.ser.reset_input_buffer()
            print(f"[Serial] Connected: {self.port}")
            return True
        except serial.SerialException as exc:
            print(f"[Serial] Connect error {self.port}: {exc}")
            self.ser = None
            return False

    def send_velocity(self, vx, vy, omega):
        if self.ser and self.ser.is_open:
            # The robot base uses the opposite longitudinal sign from the vision controller.
            cmd = f"V {-vx:.3f} {vy:.3f} {omega:.3f}\n"
            try:
                self.ser.write(cmd.encode('ascii'))
            except serial.SerialException:
                self.ser = None

    def send_stop(self):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(b"S\n")
            except serial.SerialException:
                pass

    def read_feedback(self):
        if self.ser and self.ser.is_open:
            try:
                last_line = None
                while self.ser.in_waiting:
                    line = self.ser.readline().decode('ascii', errors='ignore').strip()
                    if line:
                        match = self._TELM_RE.match(line)
                        if match:
                            self.measured_vx = -float(match.group(1))
                            self.measured_vy = float(match.group(2))
                            self.measured_wz = float(match.group(3))
                            self.robot_x = float(match.group(4))
                            self.robot_y = float(match.group(5))
                            self.robot_theta_deg = float(match.group(6))
                            self.last_telm_time = time.monotonic()
                        last_line = line
                return last_line
            except serial.SerialException:
                self.ser = None
        return None

    def close(self):
        if self.ser:
            self.send_stop()
            self.ser.close()


def find_serial_port():
    import glob

    if sys.platform.startswith('linux'):
        candidates = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
    elif sys.platform == 'darwin':
        candidates = glob.glob('/dev/tty.usbmodem*') + glob.glob('/dev/tty.usbserial*')
    else:
        candidates = [f'COM{i}' for i in range(1, 20)]

    for port in candidates:
        try:
            test_ser = serial.Serial(port, 115200, timeout=1)
            test_ser.close()
            return port
        except (serial.SerialException, OSError):
            continue
    return None


# ======================== DASHBOARD ========================

def draw_bar(frame, x, y, width, height, value, max_val, color, label=""):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), 1)
    mid = x + width // 2
    cv2.line(frame, (mid, y), (mid, y + height), (100, 100, 100), 1)

    ratio = np.clip(value / (max_val + 1e-9), -1.0, 1.0)
    bar_w = int(abs(ratio) * width // 2)
    if ratio >= 0:
        cv2.rectangle(frame, (mid, y + 1), (mid + bar_w, y + height - 1), color, -1)
    else:
        cv2.rectangle(frame, (mid - bar_w, y + 1), (mid, y + height - 1), color, -1)

    if label:
        cv2.putText(frame, label, (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)


def draw_history_graph(frame, x, y, width, height, history, max_val, color, label=""):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), 1)
    mid_y = y + height // 2
    cv2.line(frame, (x, mid_y), (x + width, mid_y), (80, 80, 80), 1)

    if len(history) < 2:
        return

    count = len(history)
    points = []
    for idx, val in enumerate(history):
        px = x + int(idx * width / count)
        ratio = np.clip(val / (max_val + 1e-9), -1.0, 1.0)
        py = mid_y - int(ratio * height / 2)
        points.append((px, py))

    for idx in range(1, len(points)):
        cv2.line(frame, points[idx - 1], points[idx], color, 1)

    if label:
        cv2.putText(frame, label, (x + 3, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def draw_dashboard(frame, info):
    height, width = frame.shape[:2]
    panel_w = 220
    overlay = frame[:, :panel_w].copy()
    cv2.rectangle(frame, (0, 0), (panel_w, height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.3, frame[:, :panel_w], 0.7, 0, frame[:, :panel_w])

    y0 = 20
    dy = 16
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    white = (220, 220, 220)
    green = (0, 255, 0)
    red = (0, 0, 255)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)

    def put(text, y, color=white):
        cv2.putText(frame, text, (8, y), font, font_scale, color, 1)
        return y + dy

    y0 = put("=== CAMERA BASED WEB TEST ===", y0, cyan)
    y0 = put(f"Status: {info.get('status', '---')}", y0, green if info.get('ball_detected') else red)
    y0 += 5

    y0 = put("--- Velocity Command ---", y0, yellow)
    y0 = put(f"vx:    {info.get('vx', 0): .3f} m/s", y0)
    y0 = put(f"vy:    {info.get('vy', 0): .3f} m/s", y0)
    y0 = put(f"omega: {info.get('omega', 0): .3f} rad/s", y0)
    y0 += 5

    draw_bar(frame, 8, y0, 200, 12, info.get('vx', 0), VX_MAX, (0, 200, 0), 'vx')
    y0 += 28
    draw_bar(frame, 8, y0, 200, 12, info.get('omega', 0), OMEGA_MAX, (200, 100, 0), 'omega')
    y0 += 28

    y0 = put("--- Error ---", y0, yellow)
    y0 = put(f"err_x:    {info.get('err_x', 0): .3f}", y0)
    y0 = put(f"err_dist: {info.get('err_dist', 0): .3f}", y0)
    y0 += 5

    y0 = put("--- Kalman State ---", y0, yellow)
    y0 = put(f"filt cx:   {info.get('kf_cx', 0): .1f} px", y0)
    y0 = put(f"filt bh:   {info.get('kf_bh', 0): .1f} px", y0)
    y0 = put(f"vel cx:    {info.get('kf_dcx', 0): .1f} px/s", y0)
    y0 = put(f"vel bh:    {info.get('kf_dbh', 0): .1f} px/s", y0)
    y0 += 5

    y0 = put("--- PID Omega ---", y0, yellow)
    y0 = put(f"P: {info.get('omega_p', 0): .3f}  I: {info.get('omega_i', 0): .3f}", y0)
    y0 = put(f"D: {info.get('omega_d', 0): .3f}", y0)
    y0 += 3

    y0 = put("--- PID Vx ---", y0, yellow)
    y0 = put(f"P: {info.get('vx_p', 0): .3f}  I: {info.get('vx_i', 0): .3f}", y0)
    y0 = put(f"D: {info.get('vx_d', 0): .3f}", y0)
    y0 += 5

    y0 = put("--- FPS ---", y0, yellow)
    y0 = put(f"Vision:  {info.get('fps_vision', 0): .1f} Hz", y0)
    y0 = put(f"Control: {info.get('fps_ctrl', 0): .1f} Hz", y0)

    graph_x = width - 215
    graph_y = 10
    draw_history_graph(frame, graph_x, graph_y, 200, 60, info.get('hist_err_x', []), 1.0, (0, 200, 255), 'err_x')
    draw_history_graph(frame, graph_x, graph_y + 70, 200, 60, info.get('hist_err_dist', []), 1.0, (0, 255, 100), 'err_dist')
    draw_history_graph(frame, graph_x, graph_y + 140, 200, 60, info.get('hist_omega', []), OMEGA_MAX, (200, 100, 0), 'omega')
    draw_history_graph(frame, graph_x, graph_y + 210, 200, 60, info.get('hist_vx', []), VX_MAX, (0, 200, 0), 'vx')


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Camera Based Web Test</title>
  <style>
    :root {
      --bg: #0e1116;
      --panel: #171c23;
      --text: #eef3f8;
      --muted: #9cb0c1;
      --accent: #f3c969;
      --line: #283240;
    }
    body {
      margin: 0;
      background: radial-gradient(circle at top, #1b2633 0%, var(--bg) 55%);
      color: var(--text);
      font-family: Consolas, 'Courier New', monospace;
    }
    .wrap {
      max-width: 1280px;
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
            background: linear-gradient(180deg, #263140 0%, #171c23 100%);
            color: var(--text);
            padding: 10px 14px;
            border-radius: 999px;
            cursor: pointer;
            font: inherit;
            font-size: 13px;
        }
        .button[data-kind="start"] {
            border-color: #2d6f46;
        }
        .button[data-kind="stop"] {
            border-color: #7d443d;
        }
        .button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 16px;
    }
    .panel {
      background: rgba(23, 28, 35, 0.9);
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 18px 50px rgba(0, 0, 0, 0.3);
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
    .kv:last-child {
      border-bottom: none;
    }
    .label {
      color: var(--muted);
    }
    .ok {
      color: #83f28f;
    }
    .warn {
      color: #ffd16b;
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
        <div class="title">Camera Based Web Test</div>
        <div class="hint">Mo tren may khac trong cung mang LAN de xem stream tu Raspberry Pi.</div>
                <div class="actions">
                    <button class="button" data-kind="start" id="startBtn">Start Motor</button>
                    <button class="button" data-kind="stop" id="stopBtn">Stop Motor</button>
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
        const statusClass = state.ball_detected ? 'ok' : 'warn';
                startBtn.disabled = !state.serial_enabled || state.motor_enabled;
                stopBtn.disabled = !state.serial_enabled || !state.motor_enabled;
        statsEl.innerHTML = [
          row('status', state.status, statusClass),
          row('serial', state.serial_enabled ? 'enabled' : 'disabled'),
                    row('motor', state.motor_enabled ? 'armed' : 'stopped', state.motor_enabled ? 'ok' : 'warn'),
                    row('reason', state.motor_reason),
          row('vx', state.vx.toFixed(3)),
          row('omega', state.omega.toFixed(3)),
          row('err_x', state.err_x.toFixed(3)),
          row('err_dist', state.err_dist.toFixed(3)),
          row('dist m', state.dist_m.toFixed(3)),
          row('bbox_h', state.kf_bh.toFixed(1)),
          row('bbox target', state.target_bbox_h.toFixed(1)),
          row('base vx', state.measured_vx.toFixed(3)),
          row('base wz', state.measured_wz.toFixed(3)),
          row('vision fps', state.fps_vision.toFixed(1)),
          row('control fps', state.fps_ctrl.toFixed(1)),
          row('last seen', `${state.last_seen_age.toFixed(2)} s`),
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
            'vx_target': 0.0,
            'vy': 0.0,
            'omega': 0.0,
            'omega_target': 0.0,
            'err_x': 0.0,
            'err_dist': 0.0,
            'dist_m': 0.0,
            'kf_cx': 0.0,
            'kf_bh': 0.0,
            'kf_dcx': 0.0,
            'kf_dbh': 0.0,
            'measured_vx': 0.0,
            'measured_vy': 0.0,
            'measured_wz': 0.0,
            'robot_x': 0.0,
            'robot_y': 0.0,
            'robot_theta_deg': 0.0,
            'omega_p': 0.0,
            'omega_i': 0.0,
            'omega_d': 0.0,
            'vx_p': 0.0,
            'vx_i': 0.0,
            'vx_d': 0.0,
            'fps_vision': 0.0,
            'fps_ctrl': 0.0,
            'target_bbox_h': float(TARGET_BBOX_H),
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


def main():
    parser = argparse.ArgumentParser(description='Camera based Web Test')
    parser.add_argument('--camera', type=int, default=CAMERA_INDEX)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--target-h', type=int, default=TARGET_BBOX_H)
    parser.add_argument('--model', type=str, default=YOLO_MODEL)
    parser.add_argument('--conf', type=float, default=YOLO_CONF)
    parser.add_argument('--imgsz', type=int, default=YOLO_IMGSZ)
    parser.add_argument('--classes', type=int, nargs='+', default=YOLO_CLASSES)
    parser.add_argument('--kalman-r', type=float, default=KALMAN_R)
    parser.add_argument('--send-serial', action='store_true')
    parser.add_argument('--no-serial', action='store_true')
    parser.add_argument('--serial-port', type=str, default=None)
    args = parser.parse_args()

    app_state = SharedAppState()
    app_state.stats['target_bbox_h'] = float(args.target_h)

    serial_mgr = None
    use_serial = not args.no_serial
    if args.send_serial:
        use_serial = True

    if use_serial:
        port = args.serial_port or find_serial_port()
        if port:
            serial_mgr = SerialManager(port)
            if not serial_mgr.connect():
                serial_mgr = None
        else:
            print('[Serial] No Arduino port found. Running in monitor-only mode.')
    app_state.serial_mgr = serial_mgr
    app_state.stats['serial_enabled'] = serial_mgr is not None
    app_state.motor_enabled = serial_mgr is not None
    app_state.stats['motor_enabled'] = app_state.motor_enabled
    app_state.stats['motor_reason'] = 'Auto armed on startup' if serial_mgr is not None else 'Serial unavailable; check cable/port or use --serial-port'

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print('[Camera] Cannot open camera')
        return

    detector = FaceDetector(model_path=args.model, conf=args.conf, imgsz=args.imgsz, classes=args.classes)
    pid_omega = PID(OMEGA_KP, OMEGA_KI, OMEGA_KD, -OMEGA_MAX, OMEGA_MAX, derivative_tau=0.06)
    pid_vy = PID(VY_KP, VY_KI, VY_KD, -VY_ALIGN_MAX, VY_ALIGN_MAX, derivative_tau=0.08)
    pid_vx = PID(VX_KP, VX_KI, VX_KD, -VX_MAX, VX_MAX, derivative_tau=0.10)
    vx_limiter = SlewRateLimiter(VX_SLEW_MPS_S)
    vy_limiter = SlewRateLimiter(VY_SLEW_MPS_S)
    omega_limiter = SlewRateLimiter(OMEGA_SLEW_RAD_S2)

    kf_cx = Kalman1D(q=KALMAN_Q, r=args.kalman_r)
    kf_bh = Kalman1D(q=KALMAN_Q, r=args.kalman_r * 2)
    frame_center_x = FRAME_WIDTH / 2.0

    hist_err_x = deque(maxlen=HIST_LEN)
    hist_err_dist = deque(maxlen=HIST_LEN)
    hist_omega = deque(maxlen=HIST_LEN)
    hist_vx = deque(maxlen=HIST_LEN)

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

    print('\n=== CAMERA BASED WEB TEST ===')
    print(f'[Web] Open: http://{args.host if args.host != "0.0.0.0" else "<pi-ip>"}:{args.port}')
    print(f'[Control] {CONTROL_RATE} Hz | target bbox_h={args.target_h}px | target_dist={TARGET_DISTANCE_M:.2f}m')
    print(f'[Model] {args.model} | conf={args.conf} | imgsz={args.imgsz}')
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

    try:
        while app_state.running:
            t_start = time.monotonic()
            dt = t_start - last_kf_time
            if dt <= 0:
                dt = ctrl_period
            last_kf_time = t_start

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
            vx = 0.0
            vy_target = 0.0
            vy = 0.0
            omega_target = 0.0
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
                dist_m = (REAL_FACE_HEIGHT_M * CAMERA_FOCAL_PX) / bh_safe
                err_dist_m = dist_m - TARGET_DISTANCE_M
                err_dist = float(np.clip(err_dist_m / max(TARGET_DISTANCE_M, 1e-3), -1.0, 1.0))

                bearing_rate = kf_cx.dx / CAMERA_FOCAL_PX
                omega_ff = -OMEGA_FF_GAIN * bearing_rate
                dist_rate = -REAL_FACE_HEIGHT_M * CAMERA_FOCAL_PX * kf_bh.dx / (bh_safe * bh_safe)
                vx_ff = VX_FF_GAIN * dist_rate

                align_err = abs(err_x)
                turn_blend = float(np.clip((align_err - SMALL_ERR_X_FOR_VY) /
                                           max(LARGE_ERR_X_FOR_WZ - SMALL_ERR_X_FOR_VY, 1e-6), 0.0, 1.0))

                if align_err <= CENTER_DEADBAND:
                    pid_vy.reset()
                    pid_omega.reset()
                    vy_target = 0.0
                    omega_target = 0.0
                else:
                    settle_err_x = math.copysign(align_err - CENTER_DEADBAND, err_x)
                    settle_scale = float(np.clip((align_err - CENTER_DEADBAND) /
                                                 max(SMALL_ERR_X_FOR_VY - CENTER_DEADBAND, 1e-6), 0.0, 1.0))
                    if turn_blend >= 0.98:
                        pid_vy.reset()
                        vy_target = 0.0
                    else:
                        vy_error = -settle_err_x * (0.25 + 0.75 * settle_scale)
                        vy_ff = -VY_ALIGN_DAMP_GAIN * bearing_rate
                        vy_cmd = pid_vy.compute(vy_error, dt, feedforward=vy_ff)
                        vy_target = float(np.clip(vy_cmd * (1.0 - turn_blend), -VY_ALIGN_MAX, VY_ALIGN_MAX))

                    if turn_blend > 0.0:
                        err_x_ctrl = err_x
                        omega_raw = -pid_omega.compute(err_x_ctrl, dt, feedforward=-omega_ff)
                        omega_target = float(np.clip(omega_raw * turn_blend, -OMEGA_ALIGN_MAX, OMEGA_ALIGN_MAX))
                    else:
                        pid_omega.reset()
                        omega_target = 0.0

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

            if not ball_detected:
                now_lost = time.monotonic()
                if lost_since is None:
                    lost_since = now_lost
                lost_duration = now_lost - lost_since

                if lost_duration < LOST_COAST_TIME:
                    vx_target = vx_limiter.value
                    vy_target = vy_limiter.value
                    omega_target = omega_limiter.value
                elif lost_duration < LOST_SOFT_STOP:
                    vx_target = 0.0
                    vy_target = 0.0
                    omega_target = 0.0
                else:
                    vx_target = 0.0
                    vy_target = 0.0
                    omega_target = 0.0

                if lost_duration > LOST_HARD_RESET:
                    pid_omega.reset()
                    pid_vy.reset()
                    pid_vx.reset()
                    kf_cx.reset()
                    kf_bh.reset()

            vx = vx_limiter.update(vx_target, dt)
            vy = vy_limiter.update(vy_target, dt)
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

            if serial_mgr and app_state.motor_enabled:
                serial_mgr.send_velocity(vx, vy, omega)
                serial_mgr.read_feedback()
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
                serial_mgr.read_feedback()

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
                            (0, 0, 255),
                            2,
                            tipLength=0.3,
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
                    cv2.putText(frame, label, (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                center_x = int(frame_center_x)
                center_y = int(FRAME_HEIGHT / 2)
                cv2.line(frame, (center_x - 25, center_y), (center_x + 25, center_y), (100, 100, 100), 1)
                cv2.line(frame, (center_x, center_y - 25), (center_x, center_y + 25), (100, 100, 100), 1)
                cv2.circle(frame, (center_x, center_y), 30, (100, 100, 100), 1)

                last_seen_age = max(0.0, time.monotonic() - last_seen) if last_seen > 0 else 0.0
                info = {
                    'status': 'TRACKING' if ball_detected else 'SEARCHING',
                    'ball_detected': ball_detected,
                    'vx': vx,
                    'vy': vy,
                    'omega': omega,
                    'vx_target': vx_target,
                    'omega_target': omega_target,
                    'err_x': err_x,
                    'err_dist': err_dist,
                    'dist_m': dist_m,
                    'kf_cx': kf_cx.x if kf_cx.initialized else 0.0,
                    'kf_bh': kf_bh.x if kf_bh.initialized else 0.0,
                    'kf_dcx': kf_cx.dx if kf_cx.initialized else 0.0,
                    'kf_dbh': kf_bh.dx if kf_bh.initialized else 0.0,
                    'measured_vx': serial_mgr.measured_vx if serial_mgr else 0.0,
                    'measured_vy': serial_mgr.measured_vy if serial_mgr else 0.0,
                    'measured_wz': serial_mgr.measured_wz if serial_mgr else 0.0,
                    'robot_x': serial_mgr.robot_x if serial_mgr else 0.0,
                    'robot_y': serial_mgr.robot_y if serial_mgr else 0.0,
                    'robot_theta_deg': serial_mgr.robot_theta_deg if serial_mgr else 0.0,
                    'omega_p': pid_omega.last_p,
                    'omega_i': pid_omega.last_i,
                    'omega_d': pid_omega.last_d,
                    'vx_p': pid_vx.last_p,
                    'vx_i': pid_vx.last_i,
                    'vx_d': pid_vx.last_d,
                    'fps_vision': fps_vision,
                    'fps_ctrl': app_state.fps_ctrl,
                    'hist_err_x': list(hist_err_x),
                    'hist_err_dist': list(hist_err_dist),
                    'hist_omega': list(hist_omega),
                    'hist_vx': list(hist_vx),
                }
                draw_dashboard(frame, info)
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
        if serial_mgr:
            serial_mgr.send_stop()
            serial_mgr.close()
        cap.release()
        print('[Exit] Stopped.')


if __name__ == '__main__':
    main()