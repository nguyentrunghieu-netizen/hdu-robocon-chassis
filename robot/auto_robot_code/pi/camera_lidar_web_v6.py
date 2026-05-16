#!/usr/bin/env python3
"""
Ball Chaser Web Test V2
=======================
Test pipeline vision + Kalman + bo dieu khien can tam v2 tren Raspberry Pi va hien thi qua web.

Tinh nang:
  - Chay headless, khong can man hinh tren Pi
  - Stream MJPEG de xem frame da annotate tren trinh duyet
  - Trang web hien thong so PID, Kalman, FPS, bbox, err_x, err_dist
  - Tuy chon gui lenh Serial xuong Arduino de test thuc te

Su dung:
    python3 camera_lidar_web_v2.py
    python3 camera_lidar_web_v2.py --host 0.0.0.0 --port 8080
    python3 camera_lidar_web_v2.py --send-serial --serial-port /dev/ttyACM0
    python3 camera_lidar_web_v2.py --serial-port /dev/ttyACM0 --lidar-port /dev/ttyUSB0
"""

import argparse
import asyncio
import json
import math
import glob
import os
import re
import sys
import threading
import time
import traceback
from collections import deque
from http import server
from socketserver import ThreadingMixIn

import cv2
import numpy as np
import serial
from ultralytics import YOLO

try:
    import kissicp
    LIDAR_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional runtime dependency
    kissicp = None
    LIDAR_IMPORT_ERROR = exc


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
CENTER_SETTLE_BAND = 0.10
FORWARD_ALIGN_LIMIT = 0.28
FORWARD_ALIGN_MIN_GAIN = 0.18

VX_MAX = 0.35
VY_MAX = 0.25
OMEGA_MAX = 1.5

VY_ALIGN_MAX = 0.22
OMEGA_ALIGN_MAX = 0.28
OMEGA_ALIGN_MIN = 0.05
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
OMEGA_NEAR_CENTER_DAMP_GAIN = 0.22

ALIGN_HOLD_ENTER = 0.025
ALIGN_HOLD_EXIT = 0.045
ALIGN_STRAFE_ENTER = 0.075
ALIGN_STRAFE_EXIT = 0.135
ALIGN_RATE_HOLD = 0.035

TURN_KP = 1.85
TURN_KD = 0.42

STRAFE_KP = 0.95
STRAFE_KD = 0.24

HOLD_KP = 0.55
HOLD_KD = 0.18
HOLD_OMEGA_MAX = 0.10

ADAPT_GAIN_MIN = 0.70
ADAPT_GAIN_MAX = 1.15
ADAPT_OVERSHOOT_ERR = 0.040
ADAPT_OVERSHOOT_RATE = 0.030
ADAPT_DROP = 0.12
ADAPT_RECOVER_PER_S = 0.05

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
ARDUINO_PORT = '/dev/ttyUSB1' if sys.platform.startswith('linux') else None
LIDAR_PORT = '/dev/ttyUSB0' if sys.platform.startswith('linux') else 'COM14'
LIDAR_BAUDRATE = 460800
LIDAR_BAUDRATES = (460800, 256000, 115200)
LIDAR_PATH_LEN = 500

# ===== Point Cloud filtering config =====
# Voxel grid: chia khong gian thanh cac o vuong, moi o chi giu 1 diem
PC_VOXEL_SIZE_M = 0.05            # 5 cm/voxel - cang nho cang chi tiet, cang lon cang muot
# Statistical Outlier Removal: loai bo nhieu
PC_SOR_K = 8                      # so diem lan can de tinh trung binh
PC_SOR_STD_RATIO = 1.5            # nguong: mean + std_ratio * std
# Temporal accumulation: gop nhieu scan de cloud on dinh
PC_MAX_POINTS = 4000              # gioi han tong so diem trong cloud (de web khong lag)
PC_DECAY_PER_SEC = 0.65           # diem cu mo dan: alpha *= decay^dt (1.0 = khong mo)
PC_MIN_ALPHA = 0.15               # diem co alpha thap hon nay se bi xoa
PC_RANGE_MIN_M = 0.15             # bo cac diem qua gan (nhieu tu ban than robot)
PC_RANGE_MAX_M = 8.0              # bo cac diem qua xa (thuong khong tin cay)
PC_SEND_EVERY_N_SCANS = 1         # gui cloud len web moi N scans (tang neu can giam tai mang)

# ===== Persistent map (SLAM) config =====
MAP_FILE = 'lidar_map.npz'        # ten file luu/tai map
MAP_VOXEL_SIZE_M = 0.08           # voxel cho global map (lon hon cloud song -> nhe hon)
MAP_MAX_POINTS = 30000            # gioi han diem trong map (tranh quá nang)
MAP_AUTOSAVE_PERIOD_S = 30.0      # tu dong luu map moi N giay
MAP_MIN_OBS_TO_KEEP = 3.5         # nguong khi LUU file: diem phai obs >= N (loai vat di dong)
MAP_SEND_EVERY_N_SCANS = 5        # global map ban hon, gui len web tha thot
MAP_SEND_MAX_POINTS = 5000        # downsample khi gui ve client de tranh nghen mang
RELOC_MAX_TRIES = 5               # so lan thu localize voi map cu khi khoi dong
RELOC_MAX_MATCH_DIST_M = 2.0      # ICP cho relocalization: cho phep match xa hon

# ===== Saved map locations / waypoint navigation =====
WAYPOINTS_FILE = 'map_waypoints.json'
WAYPOINT_LABEL_MAX_LEN = 48

# --- Toleance ---
NAV_GOAL_TOLERANCE_M = 0.15        # ban kinh dat dich (m) - vao day la ARRIVED
NAV_GOAL_EXIT_M = 0.30             # ban kinh thoat ARRIVED (hysteresis, lon hon enter de chong dao dong)
NAV_HEADING_TOLERANCE_DEG = 8.0    # sai so heading chap nhan khi vao TURN_TO_GOAL
NAV_HEADING_EXIT_DEG = 15.0        # hysteresis heading: khi da arrived, chi quay lai neu lech > 15
NAV_LIDAR_STALE_S = 1.0            # neu pose qua N giay khong update -> dung
NAV_ARRIVED_HOLD_TICKS = 6         # so tick can on dinh truoc khi xac nhan ARRIVED (chong drift)

# --- Phase machine thresholds ---
# TURN_TO_PATH: quay mat ve huong goal truoc khi di
NAV_PATH_HEADING_TOL_DEG = 12.0    # neu lech > 12 deg -> quay tai cho
NAV_PATH_HEADING_HYST_DEG = 25.0   # khi dang DRIVE, lech > 25 -> quay lai TURN

# --- Velocity limits ---
NAV_VX_MAX = 0.25                  # toc do tien lui toi da (m/s)
NAV_OMEGA_MAX = 1.20               # toc do quay toi da (rad/s)
NAV_VX_MIN_MOVE = 0.05             # toc do toi thieu de robot kip thoat ma sat
NAV_OMEGA_MIN_MOVE = 0.12          # rad/s toi thieu khi quay (GIAM tu 0.25 de chong over-shoot)
NAV_OMEGA_FINE_TOLERANCE_DEG = 3.0 # khi gan tolerance, dung omega nho de fine-tune
NAV_GOAL_TURN_OMEGA_MAX = 0.32     # quay tai goal phai mem hon de tranh overshoot
NAV_GOAL_TURN_MIN_MOVE = 0.05      # toc do toi thieu nho hon khi chi chinh huong cuoi
NAV_GOAL_TURN_KP_SCALE = 0.45      # giam luc quay khi da den dich
NAV_GOAL_TURN_KD_SCALE = 0.20      # giu D-term nhe de dap dao dong, tranh giat

# --- P gains ---
NAV_HEADING_KP = 1.30              # P gain cho omega
NAV_HEADING_KD = 0.45              # D gain - dap dao dong (DERIVATIVE!)
NAV_HEADING_KI = 0.10              # I gain - khu sai so on dinh
NAV_HEADING_I_LIMIT = 0.40         # gioi han integral wind-up (rad/s)
NAV_TRANSLATION_KP = 0.65          # P gain cho vx (hieu = khoang cach den goal)

# --- Heading source ---
# Dung EKF heading de fuse LiDAR theta (world frame) voi IMU yaw tu Mega.
# IMU yaw ben Mega duoc zero-offset luc khoi dong nen frame khac voi LiDAR; EKF
# se hoc bias nay online va tra ve heading world muot hon cho navigation.
NAV_USE_IMU_HEADING = True
NAV_BASE_TELEM_STALE_S = 0.40
NAV_HEADING_EKF_IMU_R_DEG = 2.0
NAV_HEADING_EKF_LIDAR_R_DEG = 7.0
NAV_GOAL_HEADING_EKF_LIDAR_R_DEG = 20.0
NAV_HEADING_EKF_GYRO_Q_DEG_S = 12.0
NAV_HEADING_EKF_BIAS_Q_DEG_S = 0.25

# --- Deceleration / slow-down zone ---
NAV_DECEL_DIST_M = 0.45            # khi cach goal < N -> bat dau giam toc
NAV_FINAL_APPROACH_M = 0.20        # khi cach goal < N -> dung mode rat cham

# --- Rate limits (de mượt) ---
NAV_VX_RATE_LIMIT = 0.6            # m/s^2 - thay doi vx max
NAV_OMEGA_RATE_LIMIT = 3.0         # rad/s^2

# ===== Map: raycast occupancy =====
MAP_RAYCAST_FREE_DECREMENT = 0.8   # voxel duoc laser di qua -> tru obs_count (TANG: xoa nhieu nhanh hon)
MAP_RAYCAST_HIT_INCREMENT = 0.5    # voxel co diem hit -> cong obs_count (GIAM: can nhieu scan moi xac nhan)
MAP_OBS_MAX = 20.0                 # gioi han tren cua obs_count (tranh kho dong)
MAP_MIN_OBS_TO_DISPLAY = 8.0       # render: chi ve voxel obs >= nguong (TANG tu 2.5: chi giu diem ben vung)
MAP_RAYCAST_MAX_RANGE_M = 6.0      # khong raycast qua xa (giam tai CPU)
MAP_RAYCAST_DOWNSAMPLE = 4         # chi raycast 1/N tia moi scan (tang toc)

# ===== Stationary skip: bo qua khi robot dung yen =====
# Khi robot khong di chuyen, pose ICP van rung nhe -> map bi "vay" diem xam quanh robot.
# Giai phap: neu robot di chuyen qua it giua 2 scan -> bo qua update map.
MAP_STATIONARY_TRANS_M = 0.02      # neu di chuyen < 2cm giua 2 scan -> coi nhu dung yen
MAP_STATIONARY_ROT_DEG = 1.0       # neu xoay < 1 do giua 2 scan -> coi nhu dung yen


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


def signed_deadband(value, deadband):
    if abs(value) <= deadband:
        return 0.0
    return math.copysign(abs(value) - deadband, value)


def wrap_angle_rad(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class HeadingEKF:
    """
    EKF 1D cho heading navigation.

    State:
      x[0] = theta_world      -- heading robot trong world frame cua LiDAR
      x[1] = imu_to_world_bias -- bias cong vao yaw IMU de dua ve world frame

    Measurements:
      - LiDAR theta        : z_lidar = theta_world
      - IMU yaw tu Mega    : z_imu   = theta_world - bias

    Process:
      - Predict bang gyro wz tu Mega/BNO085.

    Muc tieu cua bo loc nay la giam dao dong quay tai goal:
      - LiDAR cho world frame dung voi x,y waypoint nhung rung khi robot dung yen.
      - IMU tren Mega muot va on dinh hon, nhung lech frame ban dau.
      - EKF hoc bias frame va fuse hai nguon heading theo xac suat.
    """

    def __init__(self):
        self.x = np.zeros(2, dtype=np.float64)
        self.P = np.diag([
            math.radians(20.0) ** 2,
            math.radians(25.0) ** 2,
        ]).astype(np.float64)
        self.initialized = False
        self.gyro_q_rad_s = math.radians(NAV_HEADING_EKF_GYRO_Q_DEG_S)
        self.bias_q_rad_s = math.radians(NAV_HEADING_EKF_BIAS_Q_DEG_S)
        self.imu_r_rad = math.radians(NAV_HEADING_EKF_IMU_R_DEG)

    def clear(self):
        self.x[:] = 0.0
        self.P[:] = 0.0
        self.P[0, 0] = math.radians(20.0) ** 2
        self.P[1, 1] = math.radians(25.0) ** 2
        self.initialized = False

    def reset(self, theta_world_rad, imu_theta_rad=None):
        theta_world_rad = wrap_angle_rad(theta_world_rad)
        bias_rad = 0.0
        if imu_theta_rad is not None and math.isfinite(imu_theta_rad):
            bias_rad = wrap_angle_rad(theta_world_rad - imu_theta_rad)
        self.x = np.array([theta_world_rad, bias_rad], dtype=np.float64)
        self.P = np.diag([
            math.radians(5.0) ** 2,
            math.radians(12.0) ** 2,
        ]).astype(np.float64)
        self.initialized = True

    def predict(self, gyro_wz_rad_s, dt):
        if not self.initialized:
            return

        dt = max(float(dt), 1e-3)
        if not math.isfinite(gyro_wz_rad_s):
            gyro_wz_rad_s = 0.0

        self.x[0] = wrap_angle_rad(self.x[0] + gyro_wz_rad_s * dt)

        q_theta = (self.gyro_q_rad_s * dt) ** 2
        q_bias = (self.bias_q_rad_s * dt) ** 2
        Q = np.diag([q_theta, q_bias])
        self.P = self.P + Q

    def _update(self, innovation, H, R):
        if not self.initialized:
            return

        H = np.asarray(H, dtype=np.float64).reshape(1, 2)
        S = float(H @ self.P @ H.T) + float(R)
        if S <= 1e-12:
            return

        K = (self.P @ H.T) / S
        self.x += K[:, 0] * innovation
        self.x[0] = wrap_angle_rad(self.x[0])
        self.x[1] = wrap_angle_rad(self.x[1])
        self.P = (np.eye(2, dtype=np.float64) - K @ H) @ self.P

    def update_lidar(self, lidar_theta_rad, lidar_r_deg):
        if not self.initialized:
            return

        innovation = wrap_angle_rad(lidar_theta_rad - self.x[0])
        self._update(
            innovation,
            H=[1.0, 0.0],
            R=math.radians(lidar_r_deg) ** 2,
        )

    def update_imu_yaw(self, imu_theta_rad):
        if not self.initialized:
            return

        predicted_imu = wrap_angle_rad(self.x[0] - self.x[1])
        innovation = wrap_angle_rad(imu_theta_rad - predicted_imu)
        self._update(
            innovation,
            H=[1.0, -1.0],
            R=self.imu_r_rad ** 2,
        )

    @property
    def theta_world_rad(self):
        return float(self.x[0])

    @property
    def bias_rad(self):
        return float(self.x[1])

    @property
    def heading_std_deg(self):
        return float(math.degrees(math.sqrt(max(0.0, self.P[0, 0]))))


class AxisPD:
    def __init__(self, kp, kd, out_min, out_max):
        self.kp = kp
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.last_p = 0.0
        self.last_d = 0.0

    def reset(self):
        self.last_p = 0.0
        self.last_d = 0.0

    def compute(self, error, error_rate, scale=1.0):
        self.last_p = -self.kp * error
        self.last_d = -self.kd * error_rate
        output = scale * (self.last_p + self.last_d)
        return float(np.clip(output, self.out_min, self.out_max))


class AlignmentAutoTuner:
    def __init__(self):
        self.gain_scale = 1.0
        self.last_sign = 0
        self.overshoots = 0
        self.stable_time = 0.0

    def reset(self):
        self.gain_scale = 1.0
        self.last_sign = 0
        self.overshoots = 0
        self.stable_time = 0.0

    def update(self, err_x, err_rate, dt):
        magnitude = abs(err_x)
        sign = 0
        if magnitude >= ADAPT_OVERSHOOT_ERR:
            sign = 1 if err_x > 0.0 else -1

        crossed = (
            self.last_sign != 0
            and sign != 0
            and sign != self.last_sign
            and abs(err_rate) > ADAPT_OVERSHOOT_RATE
        )

        if crossed:
            self.overshoots += 1
            self.gain_scale = max(ADAPT_GAIN_MIN, self.gain_scale * (1.0 - ADAPT_DROP))
            self.stable_time = 0.0
        elif magnitude < ALIGN_STRAFE_ENTER and abs(err_rate) < ALIGN_RATE_HOLD:
            self.stable_time += dt
            if self.stable_time > 0.35:
                self.gain_scale = min(ADAPT_GAIN_MAX, self.gain_scale + ADAPT_RECOVER_PER_S * dt)
        else:
            self.stable_time = 0.0

        if sign != 0:
            self.last_sign = sign

        return self.gain_scale, crossed


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


def find_serial_port(preferred_ports=None, exclude_ports=None):
    import glob

    preferred_ports = [port for port in (preferred_ports or []) if port]
    exclude_ports = set(exclude_ports or [])
    candidates = []

    if sys.platform.startswith('linux'):
        for port in preferred_ports:
            if port not in candidates:
                candidates.append(port)
        for port in sorted(glob.glob('/dev/ttyACM*')):
            if port not in candidates:
                candidates.append(port)
        for port in sorted(glob.glob('/dev/ttyUSB*')):
            if port not in candidates:
                candidates.append(port)
    elif sys.platform == 'darwin':
        for port in preferred_ports:
            if port not in candidates:
                candidates.append(port)
        for port in sorted(glob.glob('/dev/tty.usbmodem*') + glob.glob('/dev/tty.usbserial*')):
            if port not in candidates:
                candidates.append(port)
    else:
        for port in preferred_ports:
            if port not in candidates:
                candidates.append(port)
        for port in [f'COM{i}' for i in range(1, 20)]:
            if port not in candidates:
                candidates.append(port)

    for port in candidates:
        if port in exclude_ports:
            continue
        try:
            test_ser = serial.Serial(port, 115200, timeout=1)
            test_ser.close()
            return port
        except (serial.SerialException, OSError):
            continue
    return None


def probe_lidar_port(port, baudrates=(460800, 256000, 115200), timeout=0.5):
    """
    Thu xem `port` co phai la LiDAR LD/RPLidar khong.
    LiDAR phan hoi protocol khi gui RPLidar GET_INFO command (0xA5 0x50).
    Header phan hoi co dang: 0xA5 0x5A ...

    Return: True neu la LiDAR, False neu khong phai
    """
    for baud in baudrates:
        try:
            with serial.Serial(port, baud, timeout=timeout) as ser:
                try:
                    ser.dtr = False
                except (OSError, serial.SerialException):
                    pass
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                # Stop bat ky scan dang chay
                ser.write(b'\xA5\x25')
                ser.flush()
                time.sleep(0.1)
                ser.reset_input_buffer()
                # Gui GET_INFO request
                ser.write(b'\xA5\x50')
                ser.flush()
                time.sleep(0.15)
                # Doc 7 byte header: 0xA5 0x5A <size_lo> <size_hi> <size_xx> <size_xx> <type>
                resp = ser.read(7)
                # Cung ngó them them du lieu thoat ra dat dong
                # LD-LiDAR khong dap ung RPLidar protocol nhung lien tuc gui packet:
                # data thuong co byte 0x54 (header LD08) hoac 0xAA hoac stream lien tuc.
                # Voi LiDAR tan suat cao -> co data trong buffer.
                bytes_in_buf = ser.in_waiting
                # Heuristic 1: response RPLidar (header 0xA5 0x5A)
                if len(resp) >= 2 and resp[0] == 0xA5 and resp[1] == 0x5A:
                    return True
                # Heuristic 2: LD-LiDAR (LD06/LD08/LD19) - stream lien tuc, byte dau 0x54
                if len(resp) >= 2 and resp[0] == 0x54:
                    return True
                # Heuristic 3: lieu tuong tu - rat nhieu data den (LD lidar gui ~hang ngan byte/giay)
                if bytes_in_buf > 50 or (len(resp) >= 4 and resp.count(0x54) >= 1):
                    return True
        except (serial.SerialException, OSError):
            continue
        except Exception:
            continue
    return False


def probe_arduino_port(port, baudrate=115200, timeout=2.5):
    """
    Thu xem `port` co phai la Arduino Mega chay firmware mecanum_base khong.
    Arduino in "MECANUM_READY" sau khi setup (~2s sau khi mo port do auto-reset DTR),
    hoac in dong telemetry "T vx vy wz x y theta..." mỗi 100ms.

    Return: True neu la Arduino, False neu khong phai
    """
    try:
        # timeout = 2.5s du de cho Arduino reset (DTR low->high tu dong) + setup
        ser = serial.Serial(port, baudrate, timeout=0.3)
    except (serial.SerialException, OSError):
        return False
    try:
        # Doi Arduino restart va in du lieu
        time.sleep(0.5)  # ngan thoi gian dau de bo qua nhieu boot
        ser.reset_input_buffer()
        # Gui lenh stop "S" - Arduino se khong loi (bao gom firmware moi nhat)
        try:
            ser.write(b'S\n')
        except Exception:
            pass
        # Doc trong N giay, tim signature
        deadline = time.monotonic() + timeout
        buf = b''
        while time.monotonic() < deadline:
            try:
                chunk = ser.read(64)
            except Exception:
                break
            if chunk:
                buf += chunk
                # Signatures: "MECANUM_READY", dong "T " (telemetry), "[IMU]", "ODOM_RESET"
                if b'MECANUM_READY' in buf:
                    return True
                # Dong telemetry: bat dau bang "T " va co 6+ float space-separated
                # Don gian: chua "T " va co dau cham "."
                lines = buf.split(b'\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith(b'T ') and line.count(b' ') >= 5:
                        return True
                if b'[IMU]' in buf or b'ODOM_RESET' in buf:
                    return True
                # Neu da lon hon ~512 byte ma chua match -> chac la khong phai Arduino
                if len(buf) > 512:
                    return False
            else:
                time.sleep(0.05)
        return False
    finally:
        try:
            ser.close()
        except Exception:
            pass


def auto_detect_ports(preferred_lidar=None, preferred_arduino=None,
                      lidar_baudrates=(460800, 256000, 115200)):
    """
    Tu dong nhan dien dau la LiDAR, dau la Arduino.
    UU TIEN tim LiDAR truoc, sau do Arduino lay port con lai.

    Return: (lidar_port, arduino_port) - moi cai co the la None neu khong tim thay.
    """
    if not sys.platform.startswith('linux'):
        # Windows/macOS: tra ve preferred (khong probe)
        return preferred_lidar, preferred_arduino

    # Lay tat ca port USB serial
    all_ports = sorted(glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*'))
    if not all_ports:
        print('[Auto-detect] Khong tim thay port USB serial nao')
        return None, None

    print(f'[Auto-detect] Cac port USB phat hien: {all_ports}')

    # ===== BUOC 1: Thu PREFERRED port truoc (neu user chi dinh) =====
    lidar_port = None
    arduino_port = None
    tried = set()

    # Sap xep thu tu probe: preferred LiDAR truoc, sau do moi port khac
    probe_order = []
    if preferred_lidar and preferred_lidar in all_ports:
        probe_order.append(preferred_lidar)
    for p in all_ports:
        if p not in probe_order:
            probe_order.append(p)

    # ===== BUOC 2: Probe LiDAR truoc (uu tien) =====
    print('[Auto-detect] Tim LiDAR...')
    for port in probe_order:
        if port in tried:
            continue
        tried.add(port)
        print(f'[Auto-detect]   Probe LiDAR tren {port}...')
        if probe_lidar_port(port, baudrates=lidar_baudrates):
            lidar_port = port
            print(f'[Auto-detect]   -> LiDAR: {port} ✓')
            break
        else:
            print(f'[Auto-detect]   -> Khong phai LiDAR')

    # ===== BUOC 3: Probe Arduino o cac port con lai =====
    print('[Auto-detect] Tim Arduino...')
    arduino_candidates = [p for p in all_ports if p != lidar_port]
    # Uu tien preferred_arduino neu co
    if preferred_arduino and preferred_arduino in arduino_candidates:
        arduino_candidates.remove(preferred_arduino)
        arduino_candidates.insert(0, preferred_arduino)

    for port in arduino_candidates:
        print(f'[Auto-detect]   Probe Arduino tren {port}...')
        if probe_arduino_port(port):
            arduino_port = port
            print(f'[Auto-detect]   -> Arduino: {port} ✓')
            break
        else:
            print(f'[Auto-detect]   -> Khong phai Arduino')

    # ===== Tom tat =====
    print(f'[Auto-detect] Ket qua: LiDAR={lidar_port}, Arduino={arduino_port}')
    return lidar_port, arduino_port


class PointCloudFilter:
    """
    Bo loc point cloud hien dai gom 3 buoc:

    1) Range gating: bo cac diem qua gan/qua xa (ranh gioi do tin cay).
    2) Voxel-grid downsampling: gop cac diem nam trong cung 1 o luoi thanh 1 diem
       trung tam -> giam mat do, on dinh hon, giam tai bandwidth.
    3) Statistical Outlier Removal (SOR): voi moi diem, tinh khoang cach trung binh
       toi K diem gan nhat. Neu vuot qua mean_global + std_ratio * std_global thi
       diem do la outlier (nhieu) -> loai bo.

    Sau do co them buoc Temporal Accumulation: cac diem moi duoc them vao cloud
    voi alpha=1.0; sau moi scan, alpha cua cac diem cu giam dan theo decay; diem
    nao alpha thap hon nguong se bi xoa. Cach nay giup cloud:
      - On dinh (khong nhap nhay frame-to-frame).
      - Tu xoa cac doi tuong da di chuyen (vi diem o vi tri cu se mo dan).
    """

    def __init__(self,
                 voxel_size=PC_VOXEL_SIZE_M,
                 sor_k=PC_SOR_K,
                 sor_std_ratio=PC_SOR_STD_RATIO,
                 max_points=PC_MAX_POINTS,
                 decay_per_sec=PC_DECAY_PER_SEC,
                 min_alpha=PC_MIN_ALPHA,
                 range_min=PC_RANGE_MIN_M,
                 range_max=PC_RANGE_MAX_M):
        self.voxel_size = float(voxel_size)
        self.sor_k = int(sor_k)
        self.sor_std_ratio = float(sor_std_ratio)
        self.max_points = int(max_points)
        self.decay_per_sec = float(decay_per_sec)
        self.min_alpha = float(min_alpha)
        self.range_min = float(range_min)
        self.range_max = float(range_max)

        # Mang luu cloud tich luy: cot 0,1 = x,y (world); cot 2 = alpha (do tuoi)
        self._cloud = np.zeros((0, 3), dtype=np.float32)
        self._last_update_t = None

    def reset(self):
        self._cloud = np.zeros((0, 3), dtype=np.float32)
        self._last_update_t = None

    # ---------- Buoc 1: range gating ----------
    def _range_gate(self, pts_local):
        if pts_local.shape[0] == 0:
            return pts_local
        dists = np.linalg.norm(pts_local[:, :2], axis=1)
        mask = (dists >= self.range_min) & (dists <= self.range_max)
        return pts_local[mask]

    # ---------- Buoc 2: voxel downsample ----------
    def _voxel_downsample(self, pts):
        if pts.shape[0] == 0 or self.voxel_size <= 0.0:
            return pts
        keys = np.floor(pts[:, :2] / self.voxel_size).astype(np.int64)
        key1d = keys[:, 0] * 1000003 + keys[:, 1]
        order = np.argsort(key1d, kind='stable')
        sorted_keys = key1d[order]
        sorted_pts = pts[order]
        unique_mask = np.empty(sorted_keys.shape[0], dtype=bool)
        unique_mask[0] = True
        unique_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
        group_starts = np.flatnonzero(unique_mask)
        group_ends = np.append(group_starts[1:], sorted_keys.shape[0])
        sums = np.add.reduceat(sorted_pts[:, :2], group_starts, axis=0)
        counts = (group_ends - group_starts).reshape(-1, 1).astype(np.float32)
        means = sums / counts
        return means.astype(np.float32)

    # ---------- Buoc 3: Statistical Outlier Removal ----------
    def _sor(self, pts):
        n = pts.shape[0]
        k = self.sor_k
        if n <= k + 1:
            return pts
        if n > 1500:
            idx = np.random.choice(n, 1500, replace=False)
            pts_s = pts[idx]
        else:
            pts_s = pts
        diff = pts_s[:, None, :] - pts_s[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        kth = np.partition(dist2, k, axis=1)[:, 1:k + 1]
        mean_dist = np.sqrt(np.mean(kth, axis=1))
        global_mean = float(np.mean(mean_dist))
        global_std = float(np.std(mean_dist))
        thresh = global_mean + self.sor_std_ratio * global_std
        keep_mask = mean_dist <= thresh
        return pts_s[keep_mask]

    # ---------- Bien doi local -> world ----------
    @staticmethod
    def _transform_to_world(pts_local, pose):
        if pts_local.shape[0] == 0:
            return pts_local
        x0, y0, theta = float(pose[0]), float(pose[1]), float(pose[2])
        c = math.cos(theta)
        s = math.sin(theta)
        out = np.empty_like(pts_local[:, :2], dtype=np.float32)
        out[:, 0] = c * pts_local[:, 0] - s * pts_local[:, 1] + x0
        out[:, 1] = s * pts_local[:, 0] + c * pts_local[:, 1] + y0
        return out

    # ---------- Buoc 4: Temporal accumulation + decay ----------
    def _decay_old(self, now_t):
        if self._cloud.shape[0] == 0 or self._last_update_t is None:
            self._last_update_t = now_t
            return
        dt = max(0.0, now_t - self._last_update_t)
        self._last_update_t = now_t
        if dt <= 0.0 or self.decay_per_sec >= 1.0:
            return
        decay = self.decay_per_sec ** dt
        self._cloud[:, 2] *= decay
        keep = self._cloud[:, 2] >= self.min_alpha
        if not np.all(keep):
            self._cloud = self._cloud[keep]

    def _merge_into_cloud(self, new_world_pts):
        if new_world_pts.shape[0] == 0:
            return
        new_block = np.empty((new_world_pts.shape[0], 3), dtype=np.float32)
        new_block[:, :2] = new_world_pts
        new_block[:, 2] = 1.0
        self._cloud = np.vstack([self._cloud, new_block])
        self._consolidate_cloud()
        if self._cloud.shape[0] > self.max_points:
            order = np.argsort(-self._cloud[:, 2])
            self._cloud = self._cloud[order[:self.max_points]]

    def _consolidate_cloud(self):
        if self._cloud.shape[0] == 0 or self.voxel_size <= 0.0:
            return
        keys = np.floor(self._cloud[:, :2] / self.voxel_size).astype(np.int64)
        key1d = keys[:, 0] * 1000003 + keys[:, 1]
        order = np.argsort(key1d, kind='stable')
        sorted_keys = key1d[order]
        sorted_data = self._cloud[order]
        unique_mask = np.empty(sorted_keys.shape[0], dtype=bool)
        unique_mask[0] = True
        unique_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
        group_starts = np.flatnonzero(unique_mask)
        sums_xy = np.add.reduceat(sorted_data[:, :2] * sorted_data[:, 2:3],
                                  group_starts, axis=0)
        sums_a = np.add.reduceat(sorted_data[:, 2], group_starts, axis=0)
        max_a = np.maximum.reduceat(sorted_data[:, 2], group_starts, axis=0)
        sums_a_safe = np.where(sums_a > 1e-9, sums_a, 1.0).reshape(-1, 1)
        means_xy = sums_xy / sums_a_safe
        new_cloud = np.empty((means_xy.shape[0], 3), dtype=np.float32)
        new_cloud[:, :2] = means_xy
        new_cloud[:, 2] = np.minimum(max_a, 1.0)
        self._cloud = new_cloud

    # ---------- API chinh ----------
    def update(self, scan_points_local, pose, now_t=None):
        """
        scan_points_local: ndarray (N,2) hoac (N,>=2), toa do diem trong frame LiDAR.
        pose: (x, y, theta) trong world frame.
        Tra ve cloud da loc, tich luy: ndarray (M,3) voi cot [x, y, alpha].
        """
        if now_t is None:
            now_t = time.monotonic()
        self._decay_old(now_t)

        if scan_points_local is None or scan_points_local.shape[0] == 0:
            return self._cloud.copy()
        pts = np.asarray(scan_points_local, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return self._cloud.copy()
        pts = pts[:, :2]
        pts = self._range_gate(pts)
        if pts.shape[0] == 0:
            return self._cloud.copy()
        pts = self._voxel_downsample(pts)
        if pts.shape[0] == 0:
            return self._cloud.copy()
        pts = self._sor(pts)
        if pts.shape[0] == 0:
            return self._cloud.copy()
        world_pts = self._transform_to_world(pts, pose)
        self._merge_into_cloud(world_pts)
        return self._cloud.copy()

    @property
    def cloud(self):
        return self._cloud


class PersistentMap:
    """
    Global map ben vung dung de luu lai sau khi tat may va load lai khi khoi dong.

    KHAC voi PointCloudFilter (cloud song):
      - KHONG co decay theo thoi gian (map ton tai mai mai).
      - Su dung MO HINH OCCUPANCY GRID voi RAYCASTING:
          * Voxel co diem hit -> obs_count += MAP_RAYCAST_HIT_INCREMENT (1.0)
          * Voxel ma laser DI QUA (free space) -> obs_count -= MAP_RAYCAST_FREE_DECREMENT
        => Vat di dong (nguoi di qua) o lan dau co 1 voxel hit, nhung lan sau khi
           laser di xuyen qua se giam dan -> tu xoa khoi map.
        => Tuong tinh (vach, ban...) duoc hit lien tuc -> obs_count tang den MAP_OBS_MAX
           -> rat ben vung.
      - Voxel size LON hon cloud song (8cm) de map nhe va tong quat.

    Cau truc noi tai:
      - self._grid: dict {voxel_key (int64): [x_center, y_center, obs_count]}
        Truy cap O(1) cho update tung voxel.
      - Khi can ndarray (de save / send / ICP): dung self._as_array().

    Format file .npz:
      - 'points': mang (N, 2) toa do x, y trong world frame
      - 'obs_count': mang (N,) so lan quan sat tinh tich luy
      - 'last_pose': mang (3,) [x, y, theta] pose cuoi cung khi luu
      - 'meta': dict {'voxel_size': ..., 'saved_at': ..., 'note': ...}
    """

    # Hang so giai thich gia tri obs_count:
    #   < 0          -> day la free space, voxel se khong duoc render
    #   0 - MIN_DISP -> chua chac, an
    #   >= MIN_DISP  -> diem ben vung, render len map

    KEY_SCALE = 1000003   # nhan x_idx voi so nay khi tao key 1D - du de tranh xung

    def __init__(self,
                 voxel_size=MAP_VOXEL_SIZE_M,
                 max_points=MAP_MAX_POINTS,
                 min_obs_to_keep=MAP_MIN_OBS_TO_KEEP):
        self.voxel_size = float(voxel_size)
        self.inv_voxel_size = 1.0 / self.voxel_size
        self.max_points = int(max_points)
        self.min_obs_to_keep = float(min_obs_to_keep)
        # Dict {key_int: [obs_count]} - chi luu obs (toa do tinh tu key)
        self._grid = {}
        self.last_pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.last_save_t = 0.0
        self.loaded_from_file = False
        self.source_file = None

    # ---------- Helpers chuyen doi key <-> index ----------
    def _xy_to_key(self, ix, iy):
        # Numpy scalars -> Python int
        return int(ix) * self.KEY_SCALE + int(iy)

    def _key_to_xy(self, key):
        ix = key // self.KEY_SCALE
        iy = key - ix * self.KEY_SCALE
        # Dau am
        if iy > self.KEY_SCALE // 2:
            iy -= self.KEY_SCALE
            ix += 1
        elif iy < -self.KEY_SCALE // 2:
            iy += self.KEY_SCALE
            ix -= 1
        return ix, iy

    def _world_to_idx(self, x, y):
        return int(math.floor(x * self.inv_voxel_size)), \
               int(math.floor(y * self.inv_voxel_size))

    def _as_array(self, only_visible=False, min_obs=None):
        """Tra ve ndarray (N, 3): cot [x, y, obs_count].
        Neu only_visible: chi tra cac voxel co obs >= min_obs (mac dinh = MIN_DISPLAY)."""
        if not self._grid:
            return np.zeros((0, 3), dtype=np.float32)
        if min_obs is None:
            min_obs = MAP_MIN_OBS_TO_DISPLAY if only_visible else None
        items = []
        vs = self.voxel_size
        # Trung tam o = (idx + 0.5) * voxel
        for key, obs in self._grid.items():
            if min_obs is not None and obs < min_obs:
                continue
            ix, iy = self._key_to_xy(key)
            x = (ix + 0.5) * vs
            y = (iy + 0.5) * vs
            items.append((x, y, obs))
        if not items:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array(items, dtype=np.float32)

    @property
    def points_xy(self):
        """Tra ve cac diem co obs_count >= MIN_DISPLAY de lam target ICP."""
        arr = self._as_array(only_visible=True)
        if arr.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return arr[:, :2].copy()

    @property
    def size(self):
        """So voxel hien thi duoc (sau filter min_obs_to_display)."""
        if not self._grid:
            return 0
        return sum(1 for v in self._grid.values() if v >= MAP_MIN_OBS_TO_DISPLAY)

    @property
    def total_voxels(self):
        return len(self._grid)

    def reset(self):
        self._grid = {}
        self.last_pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.loaded_from_file = False
        self.source_file = None

    # ---------- Bresenham 2D: liet ke voxel doc tia ----------
    def _bresenham(self, ix0, iy0, ix1, iy1):
        """Tra ve list cac (ix, iy) tu (ix0,iy0) den (ix1,iy1) (KHONG bao gom diem cuoi).
        Day la cac voxel ma tia laser di qua (free space)."""
        cells = []
        dx = abs(ix1 - ix0)
        dy = abs(iy1 - iy0)
        sx = 1 if ix0 < ix1 else -1
        sy = 1 if iy0 < iy1 else -1
        err = dx - dy
        x, y = ix0, iy0
        # Gioi han so buoc de tranh raycast cuc dai (vd loi chia 0)
        max_steps = dx + dy + 2
        steps = 0
        while True:
            if x == ix1 and y == iy1:
                break
            cells.append((x, y))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            steps += 1
            if steps > max_steps:
                break
        return cells

    # ---------- Update chinh: raycast occupancy ----------
    def add_scan_with_raycast(self, world_pts, sensor_xy):
        """
        Cap nhat map voi raycasting:
          - Voi moi diem hit P: lay duong di tu sensor -> P, giam obs cua cac voxel
            tren duong (free space), tang obs cua voxel chua P (hit).

        world_pts: (N, 2) cac diem hit trong world frame.
        sensor_xy: (2,) vi tri lidar trong world frame.
        """
        if world_pts is None or world_pts.shape[0] == 0:
            return

        sx, sy = float(sensor_xy[0]), float(sensor_xy[1])
        ix0, iy0 = self._world_to_idx(sx, sy)
        max_range_idx2 = (MAP_RAYCAST_MAX_RANGE_M * self.inv_voxel_size) ** 2

        downsample = max(1, int(MAP_RAYCAST_DOWNSAMPLE))

        for i in range(0, world_pts.shape[0], downsample):
            px = float(world_pts[i, 0])
            py = float(world_pts[i, 1])
            ix1, iy1 = self._world_to_idx(px, py)

            # Bo qua tia qua dai (giam tai)
            d2 = (ix1 - ix0) ** 2 + (iy1 - iy0) ** 2
            if d2 > max_range_idx2:
                continue

            # 1) Free cells: tat ca voxel doc tia tru voxel hit
            free_cells = self._bresenham(ix0, iy0, ix1, iy1)
            for (cx, cy) in free_cells:
                key = self._xy_to_key(cx, cy)
                cur = self._grid.get(key, 0.0)
                new = cur - MAP_RAYCAST_FREE_DECREMENT
                # Khong cho di xuong qua sau (-3 = chac chan free)
                if new < -3.0:
                    new = -3.0
                self._grid[key] = new

            # 2) Hit cell: tang obs
            key_hit = self._xy_to_key(ix1, iy1)
            cur = self._grid.get(key_hit, 0.0)
            new = cur + MAP_RAYCAST_HIT_INCREMENT
            if new > MAP_OBS_MAX:
                new = MAP_OBS_MAX
            self._grid[key_hit] = new

        # Don dep dinh ky: bo cac voxel hoan toan free
        if len(self._grid) > self.max_points * 1.5:
            self._prune()

    def _prune(self):
        """Xoa cac voxel co obs_count rat thap (chac chan la free space) de tiet kiem RAM."""
        before = len(self._grid)
        self._grid = {k: v for k, v in self._grid.items() if v > -1.5}
        # Neu van qua dong sau prune -> giu top max_points theo obs cao nhat
        if len(self._grid) > self.max_points:
            items = sorted(self._grid.items(), key=lambda kv: -kv[1])[:self.max_points]
            self._grid = dict(items)
        after = len(self._grid)
        if before != after:
            print(f'[Map] pruned voxels: {before} -> {after}')

    # ---------- Backward-compat: gọi add_scan_world van OK ----------
    def add_scan_world(self, world_pts, sensor_xy=None):
        """Tuong thich nguoc - neu khong co sensor_xy thi khong raycast, chi them hit."""
        if world_pts is None or world_pts.shape[0] == 0:
            return
        if sensor_xy is not None:
            self.add_scan_with_raycast(world_pts, sensor_xy)
            return
        # Fallback: chi them hit (khong raycast)
        for i in range(world_pts.shape[0]):
            ix, iy = self._world_to_idx(float(world_pts[i, 0]), float(world_pts[i, 1]))
            key = self._xy_to_key(ix, iy)
            cur = self._grid.get(key, 0.0)
            self._grid[key] = min(cur + MAP_RAYCAST_HIT_INCREMENT, MAP_OBS_MAX)

    def save(self, file_path, pose=None, note=''):
        """Luu map ra file .npz. Chi luu cac voxel co obs_count >= min_obs_to_keep."""
        if not self._grid:
            print(f'[Map] save skipped: empty map')
            return False
        try:
            # Trich xuat tat ca voxel co obs >= nguong (de loai vat di dong/free)
            arr = self._as_array(only_visible=True, min_obs=self.min_obs_to_keep)
            if arr.shape[0] == 0:
                print(f'[Map] save skipped: no points pass min_obs={self.min_obs_to_keep}')
                return False
            if pose is not None:
                self.last_pose = np.array(
                    [float(pose[0]), float(pose[1]), float(pose[2])],
                    dtype=np.float32)
            meta = {
                'voxel_size': float(self.voxel_size),
                'saved_at': time.time(),
                'note': str(note),
                'total_points': int(arr.shape[0]),
                'total_voxels_in_memory': len(self._grid),
                'min_obs_to_keep': float(self.min_obs_to_keep),
            }
            np.savez_compressed(
                str(file_path) + '.tmp',
                points=arr[:, :2].astype(np.float32),
                obs_count=arr[:, 2].astype(np.float32),
                last_pose=self.last_pose,
                meta=np.array([json.dumps(meta)], dtype=object),
            )
            tmp_path = str(file_path) + '.tmp.npz'
            os.replace(tmp_path, file_path)
            self.last_save_t = time.monotonic()
            print(f'[Map] SAVED {arr.shape[0]} pts -> {file_path} '
                  f'(from {len(self._grid)} voxels in memory, '
                  f'pose=[{self.last_pose[0]:+.2f},{self.last_pose[1]:+.2f},'
                  f'{math.degrees(self.last_pose[2]):+.1f}deg])')
            return True
        except Exception as exc:
            print(f'[Map] SAVE FAILED: {type(exc).__name__}: {exc!r}')
            return False

    def load(self, file_path):
        """Load map tu file .npz. Tra ve True neu thanh cong.
        Cac diem load len duoc dat obs_count nhu trong file (da pass nguong khi save)."""
        try:
            if not os.path.isfile(file_path):
                print(f'[Map] No existing map at {file_path}')
                return False
            data = np.load(file_path, allow_pickle=True)
            pts = np.asarray(data['points'], dtype=np.float32)
            obs = np.asarray(data['obs_count'], dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2:
                raise ValueError(f'Bad points shape: {pts.shape}')
            if obs.shape[0] != pts.shape[0]:
                obs = np.full(pts.shape[0], float(self.min_obs_to_keep),
                              dtype=np.float32)

            self._grid = {}
            for i in range(pts.shape[0]):
                ix, iy = self._world_to_idx(float(pts[i, 0]), float(pts[i, 1]))
                key = self._xy_to_key(ix, iy)
                # Voi voxel da trung, lay max obs
                cur = self._grid.get(key, 0.0)
                self._grid[key] = max(cur, float(obs[i]))

            if 'last_pose' in data.files:
                lp = np.asarray(data['last_pose'], dtype=np.float32).flatten()
                if lp.shape[0] >= 3:
                    self.last_pose = lp[:3].copy()

            meta_text = ''
            if 'meta' in data.files:
                try:
                    meta_text = str(data['meta'][0])
                except Exception:
                    pass

            self.loaded_from_file = True
            self.source_file = str(file_path)
            print(f'[Map] LOADED {pts.shape[0]} pts ({len(self._grid)} unique voxels) '
                  f'from {file_path}')
            print(f'[Map]   last_pose=[{self.last_pose[0]:+.2f},'
                  f'{self.last_pose[1]:+.2f},'
                  f'{math.degrees(self.last_pose[2]):+.1f}deg]')
            if meta_text:
                print(f'[Map]   meta={meta_text}')
            return True
        except Exception as exc:
            print(f'[Map] LOAD FAILED: {type(exc).__name__}: {exc!r}')
            self.reset()
            return False

    def downsample_for_send(self, max_points):
        """Lay 1 mau con cua map de gui ve client (chi cac voxel hien thi duoc)."""
        arr = self._as_array(only_visible=True)
        if arr.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if arr.shape[0] <= max_points:
            return arr
        # Lay top max_points theo obs_count
        order = np.argsort(-arr[:, 2])
        return arr[order[:max_points]]


class WaypointStore:
    def __init__(self, file_path):
        self.file_path = str(file_path)
        self.lock = threading.Lock()
        self._waypoints = []
        self.load()

    def list(self):
        with self.lock:
            return [dict(wp) for wp in self._waypoints]

    def get(self, waypoint_id):
        waypoint_id = str(waypoint_id)
        with self.lock:
            for wp in self._waypoints:
                if wp.get('id') == waypoint_id:
                    return dict(wp)
        return None

    def add(self, label, x, y, theta=None):
        """Them mot waypoint. theta = None nghia la khong dat huong (giu huong cuoi)."""
        with self.lock:
            clean_label = self._clean_label(label, len(self._waypoints) + 1)
            stamp = int(time.time() * 1000)
            waypoint = {
                'id': f'wp_{stamp}_{len(self._waypoints) + 1}',
                'label': clean_label,
                'x': float(x),
                'y': float(y),
                # theta: None hoac float (radians). Khi None, robot khong quay them tai goal.
                'theta': float(theta) if theta is not None else None,
                'created_at': time.time(),
            }
            self._waypoints.append(waypoint)
            self._save_locked()
            return dict(waypoint)

    def delete(self, waypoint_id):
        waypoint_id = str(waypoint_id)
        with self.lock:
            before = len(self._waypoints)
            self._waypoints = [
                wp for wp in self._waypoints
                if wp.get('id') != waypoint_id
            ]
            deleted = len(self._waypoints) != before
            if deleted:
                self._save_locked()
            return deleted

    def load(self):
        with self.lock:
            self._waypoints = []
            if not os.path.isfile(self.file_path):
                return
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                items = raw.get('waypoints', raw) if isinstance(raw, dict) else raw
                if not isinstance(items, list):
                    raise ValueError('Waypoint file must contain a list')
                loaded = []
                for idx, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    # theta co the la None hoac float (radians)
                    theta_raw = item.get('theta', None)
                    theta_val = None
                    if theta_raw is not None:
                        try:
                            theta_val = float(theta_raw)
                        except (TypeError, ValueError):
                            theta_val = None
                    loaded.append({
                        'id': str(item.get('id') or f'wp_loaded_{idx + 1}'),
                        'label': self._clean_label(item.get('label'), idx + 1),
                        'x': float(item.get('x', 0.0)),
                        'y': float(item.get('y', 0.0)),
                        'theta': theta_val,
                        'created_at': float(item.get('created_at', 0.0) or 0.0),
                    })
                self._waypoints = loaded
                print(f'[Waypoints] LOADED {len(loaded)} points from {self.file_path}')
            except Exception as exc:
                print(f'[Waypoints] LOAD FAILED: {type(exc).__name__}: {exc!r}')
                self._waypoints = []

    def _save_locked(self):
        try:
            parent = os.path.dirname(os.path.abspath(self.file_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp_path = f'{self.file_path}.tmp'
            payload = {
                'version': 1,
                'saved_at': time.time(),
                'frame': 'lidar_map_world',
                'waypoints': self._waypoints,
            }
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.write('\n')
            os.replace(tmp_path, self.file_path)
        except Exception as exc:
            print(f'[Waypoints] SAVE FAILED: {type(exc).__name__}: {exc!r}')
            raise

    @staticmethod
    def _clean_label(label, fallback_idx):
        text = re.sub(r'\s+', ' ', str(label or '').strip())
        if not text:
            text = f'Point {fallback_idx}'
        return text[:WAYPOINT_LABEL_MAX_LEN]


class LidarPoseManager:
    def __init__(self, app_state, port=LIDAR_PORT, baudrates=None,
                 map_file=MAP_FILE, autosave_period=MAP_AUTOSAVE_PERIOD_S):
        self.app_state = app_state
        self.port = port
        self.baudrates = tuple(baudrates or LIDAR_BAUDRATES)
        self.thread = None
        self.lidar = None
        self.path = deque(maxlen=LIDAR_PATH_LEN)
        self.last_console_log = 0.0
        self.last_status = None
        # Bo loc point cloud va dem scan de gui dinh ky len web
        self.pc_filter = PointCloudFilter()
        self._pc_scan_counter = 0
        # Persistent global map
        self.map = PersistentMap()
        self.map_file = map_file
        self.autosave_period = float(autosave_period)
        self._map_send_counter = 0
        self._save_request = False  # co nut bam tu web
        # Pose lan cuoi update map (de bo qua khi robot dung yen)
        self._last_map_update_pose = None

    def _list_visible_serial_ports(self):
        if sys.platform.startswith('linux'):
            ports = sorted(glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*'))
        elif sys.platform == 'darwin':
            ports = sorted(glob.glob('/dev/tty.usbmodem*') + glob.glob('/dev/tty.usbserial*'))
        else:
            ports = [f'COM{i}' for i in range(1, 30)]

        visible = []
        for port in ports:
            try:
                if sys.platform.startswith('linux') or sys.platform == 'darwin':
                    visible.append(port)
                else:
                    test_ser = serial.Serial(port, 115200, timeout=0.05)
                    test_ser.close()
                    visible.append(port)
            except (serial.SerialException, OSError):
                continue
        return visible

    def _candidate_ports(self):
        visible = self._list_visible_serial_ports()
        if not visible:
            return [self.port]

        ordered = []
        for port in (self.port, *visible):
            if port and port not in ordered:
                ordered.append(port)
        return ordered

    def start(self):
        if kissicp is None:
            print(f'[LiDAR] DISABLED: cannot import kissicp: {LIDAR_IMPORT_ERROR}')
        else:
            print(f'[LiDAR] STARTING: port={self.port} baudrates={self.baudrates}')
            print(f'[LiDAR] kissicp module: {getattr(kissicp, "__file__", "unknown")}')
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()

    def stop(self):
        if self.lidar is not None:
            try:
                self.lidar.stop_event.set()
            except Exception:
                pass
        # Luu map lan cuoi truoc khi tat (snapshot tot nhat)
        try:
            if self.map.size > 50:
                self.map.save(self.map_file, pose=self.map.last_pose,
                              note='shutdown')
        except Exception as exc:
            print(f'[Map] final save failed: {type(exc).__name__}: {exc!r}')

    def _thread_main(self):
        try:
            asyncio.run(self._run_async())
        except Exception as exc:
            error_text = f'{type(exc).__name__}: {exc!r}'
            self._update_status('ERROR', lidar_error_text=error_text)
            print(f'[LiDAR] ERROR: {error_text}')
            traceback.print_exc(limit=6)

    def _print_serial_devices(self):
        visible = self._list_visible_serial_ports()
        print(f'[LiDAR] Visible serial ports: {visible if visible else "none"}')
        if self.port not in visible and (sys.platform.startswith('linux') or sys.platform == 'darwin'):
            print(f'[LiDAR] WARNING: configured port {self.port} is not in visible serial ports')

    def _prepare_lidar_port(self, port, baudrate):
        try:
            with serial.Serial(port, baudrate, timeout=0.2) as test_ser:
                try:
                    test_ser.dtr = False
                except (OSError, serial.SerialException):
                    pass
                test_ser.reset_input_buffer()
                test_ser.reset_output_buffer()
                test_ser.write(b'\xA5\x25')  # RPLidar STOP
                test_ser.flush()
                time.sleep(0.08)
                test_ser.reset_input_buffer()
                print(f'[LiDAR] Prepared serial buffer on {port} @ {baudrate}')
        except Exception as exc:
            print(f'[LiDAR] Preflight failed on {port} @ {baudrate}: {type(exc).__name__}: {exc!r}')

    def _update_status(self, status, **extra):
        now = time.monotonic()
        with self.app_state.lock:
            self.app_state.stats.update({
                'lidar_enabled': kissicp is not None,
                'lidar_status': status,
                'lidar_last_update': now,
            })
            self.app_state.stats.update(extra)
        note = extra.get('lidar_error_text', '')
        if status != self.last_status or now - self.last_console_log >= 1.0:
            self.last_status = status
            self.last_console_log = now
            if note:
                print(f'[LiDAR] {status}: {note}')
            else:
                print(f'[LiDAR] {status}')

    def _update_pose(self, pose, step, error, matches, overlap, scan_count, accepted,
                     rejected, hard_resets, point_count, map_points):
        now = time.monotonic()
        self.path.append((float(pose[0]), float(pose[1])))
        with self.app_state.lock:
            self.app_state.stats.update({
                'lidar_enabled': True,
                'lidar_status': 'TRACKING',
                'lidar_x': float(pose[0]),
                'lidar_y': float(pose[1]),
                'lidar_theta_deg': float(math.degrees(pose[2])),
                'lidar_step_dx': float(step[0]),
                'lidar_step_dy': float(step[1]),
                'lidar_step_dtheta_deg': float(math.degrees(step[2])),
                'lidar_error': float(error),
                'lidar_matches': int(matches),
                'lidar_overlap': float(overlap),
                'lidar_scans': int(scan_count),
                'lidar_accepted': int(accepted),
                'lidar_rejected': int(rejected),
                'lidar_hard_resets': int(hard_resets),
                'lidar_points': int(point_count),
                'lidar_map_points': int(map_points),
                'lidar_path': [[x, y] for x, y in self.path],
                'lidar_last_update': now,
                'lidar_error_text': '',
            })
        if now - self.last_console_log >= 1.0:
            self.last_console_log = now
            print(
                '[LiDAR] TRACKING '
                f'scan={scan_count} accepted={accepted} rejected={rejected} '
                f'x={pose[0]:+.3f}m y={pose[1]:+.3f}m '
                f'theta={math.degrees(pose[2]):+.1f}deg '
                f'err={error:.3f} overlap={overlap:.2f} pts={point_count}'
            )

    def _publish_cloud(self):
        """Day cloud da loc vao app_state.stats de gui ve trinh duyet."""
        cloud = self.pc_filter.cloud
        if cloud is None or cloud.shape[0] == 0:
            payload = []
        else:
            payload = [
                [round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 2)]
                for p in cloud
            ]
        with self.app_state.lock:
            self.app_state.stats['lidar_cloud'] = payload
            self.app_state.stats['lidar_cloud_count'] = len(payload)

    def _publish_map(self, force=False):
        """Day global map xuong client (downsample neu can)."""
        self._map_send_counter += 1
        if not force and self._map_send_counter % MAP_SEND_EVERY_N_SCANS != 0:
            return
        sub = self.map.downsample_for_send(MAP_SEND_MAX_POINTS)
        if sub.shape[0] == 0:
            payload = []
        else:
            # Chi gui [x, y] - khong can obs_count o client
            payload = [
                [round(float(p[0]), 3), round(float(p[1]), 3)]
                for p in sub
            ]
        with self.app_state.lock:
            self.app_state.stats['lidar_map'] = payload
            self.app_state.stats['lidar_map_total'] = self.map.size
            self.app_state.stats['lidar_map_loaded'] = self.map.loaded_from_file

    def request_save(self):
        """Cho phep web/UI yeu cau luu map ngay."""
        self._save_request = True

    def _try_save_if_due(self, pose):
        """Luu map neu da den chu ky tu dong hoac co yeu cau thu cong."""
        now = time.monotonic()
        due = self.autosave_period > 0 and (now - self.map.last_save_t) >= self.autosave_period
        if not (due or self._save_request):
            return
        # Khong luu khi map qua nho (chua co du data)
        if self.map.size < 50 and not self._save_request:
            self.map.last_save_t = now  # delay lan kiem tra ke tiep
            return
        ok = self.map.save(self.map_file, pose=pose,
                           note='auto' if due else 'manual')
        with self.app_state.lock:
            self.app_state.stats['lidar_map_last_save'] = time.time() if ok else 0.0
            self.app_state.stats['lidar_map_save_ok'] = bool(ok)
        self._save_request = False

    def _try_relocalize(self, first_scan_points):
        """
        Khi co map cu da load, ICP scan dau tien voi map cu de tim pose ban dau.
        Tra ve pose (x, y, theta) trong frame map cu, hoac None neu that bai.

        Diem khoi dau cho ICP la `last_pose` luu trong file (vi tri robot luc tat
        may lan truoc - thuong robot khoi dong gan vi tri do).
        """
        if not self.map.loaded_from_file or self.map.size < 50:
            return None
        if first_scan_points is None or first_scan_points.shape[0] < kissicp.ICP_MIN_MATCHES:
            return None

        init_pose = (float(self.map.last_pose[0]),
                     float(self.map.last_pose[1]),
                     float(self.map.last_pose[2]))

        target = self.map.points_xy
        # ICP voi nguong match noi long de chiu loi pose ban dau
        for attempt in range(RELOC_MAX_TRIES):
            try:
                result, fail_reason = kissicp.icp_2d(
                    first_scan_points,
                    target,
                    init_delta=init_pose,
                    max_match_dist=RELOC_MAX_MATCH_DIST_M,
                )
            except Exception as exc:
                print(f'[Map] reloc ICP exception: {type(exc).__name__}: {exc!r}')
                return None
            if result is not None:
                new_pose, error, matches, overlap = result
                print(f'[Map] RELOCALIZED at attempt {attempt+1}: '
                      f'pose=[{new_pose[0]:+.2f},{new_pose[1]:+.2f},'
                      f'{math.degrees(new_pose[2]):+.1f}deg] '
                      f'err={error:.3f} matches={matches} overlap={overlap:.2f}')
                return new_pose
            print(f'[Map] reloc attempt {attempt+1}/{RELOC_MAX_TRIES} failed: {fail_reason}')
        print(f'[Map] RELOC FAILED after {RELOC_MAX_TRIES} attempts -> using identity pose')
        return None

    async def _run_async(self):
        if kissicp is None:
            self._update_status('DISABLED', lidar_error_text=str(LIDAR_IMPORT_ERROR))
            return

        self._update_status('CONNECTING', lidar_error_text='')
        self._print_serial_devices()
        candidate_ports = self._candidate_ports()
        print(f'[LiDAR] Candidate ports: {candidate_ports}')
        init_errors = []
        active_port = self.port
        for port in candidate_ports:
            for baudrate in self.baudrates:
                print(f'[LiDAR] Opening serial device {port} @ {baudrate}')
                self._prepare_lidar_port(port, baudrate)
                try:
                    self.lidar = kissicp.RPLidar(port, baudrate)
                    active_port = port
                    with self.app_state.lock:
                        self.app_state.stats['lidar_baudrate'] = baudrate
                        self.app_state.stats['lidar_port'] = port
                    print(f'[LiDAR] RPLidar initialized on {port} at {baudrate} baud')
                    break
                except Exception as exc:
                    init_errors.append(f'{port}@{baudrate}: {type(exc).__name__}: {exc!r}')
                    print(f'[LiDAR] Init failed on {port} @ {baudrate}: {type(exc).__name__}: {exc!r}')
                    self.lidar = None
            if self.lidar is not None:
                break

        if self.lidar is None:
            raise RuntimeError(
                f'Cannot initialize RPLidar on candidate ports {candidate_ports}. Tried: ' + '; '.join(init_errors)
            )
        self.port = active_port
        self._update_status('CONNECTED', lidar_error_text='')
        print('[LiDAR] Connected. Waiting for full 360-degree scans...')

        try:
            await asyncio.gather(
                self.lidar.simple_scan(make_return_dict=False),
                self._process_scans(self.lidar),
            )
        finally:
            try:
                self.lidar.stop_event.set()
                time.sleep(0.2)
                self.lidar.reset()
            except Exception:
                pass
            self._update_status('STOPPED')

    async def _process_scans(self, lidar):
        local_map = kissicp.LocalMap(
            kissicp.LOCAL_MAP_NUM_SCANS,
            kissicp.LOCAL_MAP_VOXEL_M,
            kissicp.LOCAL_MAP_RANGE_M,
        )
        # ===== Load map cu (neu co) =====
        skip_load = getattr(self, '_skip_load', False)
        loaded = False if skip_load else self.map.load(self.map_file)
        if skip_load:
            print('[Map] --no-load-map: skipping load, starting fresh')
        if loaded:
            with self.app_state.lock:
                self.app_state.stats['lidar_map_loaded'] = True
                self.app_state.stats['lidar_map_total'] = self.map.size
            # Day map cu len web ngay tu dau
            self._publish_map(force=True)
            pose = (float(self.map.last_pose[0]),
                    float(self.map.last_pose[1]),
                    float(self.map.last_pose[2]))
        else:
            pose = (0.0, 0.0, 0.0)

        velocity_per_scan = (0.0, 0.0, 0.0)
        current_scan = []
        prev_angle = None
        consecutive_failures = 0
        scan_count = 0
        accepted = 0
        rejected = 0
        hard_resets = 0
        relocalized = not loaded  # neu khong co map cu, coi nhu khong can relocalize

        self.path.clear()
        self.path.append((pose[0], pose[1]))
        # Khoi tao thoi diem cuoi cua autosave de tinh tu luc khoi dong
        self.map.last_save_t = time.monotonic()

        while self.app_state.running and not lidar.stop_event.is_set():
            try:
                point = await asyncio.wait_for(lidar.output_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            angle = float(point['a_deg'])
            wrapped = prev_angle is not None and prev_angle > 300.0 and angle < 60.0
            prev_angle = angle

            if not (wrapped and len(current_scan) > 60):
                current_scan.append(point)
                continue

            scan_count += 1
            scan_buf = current_scan
            current_scan = [point]
            points = kissicp.scan_to_points(scan_buf, velocity_per_scan)

            if points.shape[0] < kissicp.ICP_MIN_MATCHES:
                self._update_status(
                    'LOW_POINTS',
                    lidar_scans=scan_count,
                    lidar_points=int(points.shape[0]),
                    lidar_error_text='too few valid points',
                )
                if time.monotonic() - self.last_console_log >= 1.0:
                    self.last_console_log = time.monotonic()
                    print(f'[LiDAR] LOW_POINTS scan={scan_count} valid_points={points.shape[0]}')
                continue

            # ===== Relocalization voi map cu (chi lan dau) =====
            if not relocalized:
                reloc_pose = self._try_relocalize(points)
                if reloc_pose is not None:
                    pose = reloc_pose
                    self.path.clear()
                    self.path.append((pose[0], pose[1]))
                    with self.app_state.lock:
                        self.app_state.stats['lidar_map_relocalized'] = True
                else:
                    # Khong reloc duoc -> reset goc toa do, bat dau map moi
                    print('[Map] Relocalization failed, starting fresh map at origin')
                    self.map.reset()
                    pose = (0.0, 0.0, 0.0)
                    with self.app_state.lock:
                        self.app_state.stats['lidar_map_relocalized'] = False
                        self.app_state.stats['lidar_map_loaded'] = False
                        self.app_state.stats['lidar_map_total'] = 0
                relocalized = True

            if len(local_map.scans_world) == 0:
                local_map.add_scan(points, pose)
                print(f'[LiDAR] Local map initialized: scan={scan_count} '
                      f'pts={points.shape[0]} pose=[{pose[0]:+.2f},{pose[1]:+.2f},'
                      f'{math.degrees(pose[2]):+.1f}deg]')
                self._update_pose(
                    pose, (0.0, 0.0, 0.0), 0.0, 0, 1.0,
                    scan_count, accepted, rejected, hard_resets,
                    points.shape[0], points.shape[0],
                )
                continue

            match_thresh = kissicp.adaptive_threshold(velocity_per_scan)
            target_map = local_map.query_near(pose)
            if target_map.shape[0] < kissicp.ICP_MIN_MATCHES:
                local_map.add_scan(points, pose)
                continue

            predicted_pose = kissicp.apply_robot_delta(pose, velocity_per_scan)
            result, fail_reason = kissicp.icp_2d(
                points,
                target_map,
                init_delta=predicted_pose,
                max_match_dist=match_thresh,
            )

            if result is None:
                consecutive_failures += 1
                rejected += 1
                if consecutive_failures >= kissicp.MAX_CONSECUTIVE_FAILURES:
                    local_map.scans_world.clear()
                    local_map.add_scan(points, pose)
                    consecutive_failures = 0
                    hard_resets += 1
                    velocity_per_scan = (0.0, 0.0, 0.0)
                    status = 'RESET'
                else:
                    status = 'ICP_FAIL'
                self._update_status(
                    status,
                    lidar_scans=scan_count,
                    lidar_rejected=rejected,
                    lidar_hard_resets=hard_resets,
                    lidar_points=int(points.shape[0]),
                    lidar_error_text=str(fail_reason),
                )
                if time.monotonic() - self.last_console_log >= 1.0:
                    self.last_console_log = time.monotonic()
                    print(
                        f'[LiDAR] {status} scan={scan_count} reason={fail_reason} '
                        f'rejected={rejected} hard_resets={hard_resets}'
                    )
                continue

            consecutive_failures = 0
            new_pose, error, matches, overlap = result
            step = kissicp.relative_delta(pose, new_pose)

            if not kissicp.velocity_is_plausible(step):
                rejected += 1
                self._update_status(
                    'REJECTED',
                    lidar_scans=scan_count,
                    lidar_rejected=rejected,
                    lidar_points=int(points.shape[0]),
                    lidar_error_text='implausible velocity',
                )
                if time.monotonic() - self.last_console_log >= 1.0:
                    self.last_console_log = time.monotonic()
                    print(f'[LiDAR] REJECTED scan={scan_count}: implausible velocity')
                continue

            pose = new_pose
            velocity_per_scan = step
            local_map.add_scan(points, pose)
            accepted += 1
            map_points = sum(scan.shape[0] for scan in local_map.scans_world)

            # ===== Cap nhat point cloud da loc =====
            # `points` la mang (N, >=2) trong frame LiDAR (local). Loc + tich luy.
            try:
                self.pc_filter.update(points, pose)
            except Exception as filter_exc:
                print(f'[LiDAR] PC filter error: {type(filter_exc).__name__}: {filter_exc!r}')

            self._pc_scan_counter += 1
            send_cloud_now = (self._pc_scan_counter % max(1, PC_SEND_EVERY_N_SCANS) == 0)
            if send_cloud_now:
                self._publish_cloud()

            # ===== Cap nhat persistent global map =====
            # Lay diem da loc trong cloud song (cot 0,1) -> chuyen ve world ->
            # them vao map. Vi cloud song dung pose moi nhat luc filter, ta lay
            # voxel filter cua scan hien tai roi transform de tranh trung lap.
            #
            # QUAN TRONG: Bo qua update map khi robot dung yen.
            # Pose ICP rung nhe (jitter) khi dung yen -> cung 1 buc tuong se duoc
            # ghi vao nhieu voxel khac nhau qua nhieu scan -> tao "vong tron xam"
            # gia quanh robot. Bang cach skip khi dung yen, map chi cap nhat khi
            # co thong tin moi (robot di chuyen sang vi tri khac).
            skip_map_update = False
            if self._last_map_update_pose is not None:
                dx = float(pose[0]) - self._last_map_update_pose[0]
                dy = float(pose[1]) - self._last_map_update_pose[1]
                dtheta = float(pose[2]) - self._last_map_update_pose[2]
                # Chuan hoa goc ve [-pi, pi]
                while dtheta > math.pi:
                    dtheta -= 2 * math.pi
                while dtheta < -math.pi:
                    dtheta += 2 * math.pi
                trans_dist = math.sqrt(dx * dx + dy * dy)
                rot_deg = math.degrees(abs(dtheta))
                if (trans_dist < MAP_STATIONARY_TRANS_M and
                        rot_deg < MAP_STATIONARY_ROT_DEG):
                    skip_map_update = True

            try:
                if skip_map_update:
                    # Robot dung yen -> chi cap nhat last_pose, khong sua map
                    self.map.last_pose = np.array(
                        [float(pose[0]), float(pose[1]), float(pose[2])],
                        dtype=np.float32)
                else:
                    # Loc nhanh scan hien tai roi them vao map (giong cloud nhung khong decay)
                    pts_local = points[:, :2] if points.ndim == 2 else points
                    # Range gate + voxel cua chinh map (8cm)
                    dists = np.linalg.norm(pts_local, axis=1)
                    pts_local = pts_local[(dists >= PC_RANGE_MIN_M) &
                                          (dists <= PC_RANGE_MAX_M)]
                    if pts_local.shape[0] > 0:
                        # Voxel theo MAP_VOXEL_SIZE_M de gop diem trong scan
                        keys = np.floor(pts_local / MAP_VOXEL_SIZE_M).astype(np.int64)
                        key1d = keys[:, 0] * 1000003 + keys[:, 1]
                        _, idx_unique = np.unique(key1d, return_index=True)
                        pts_local = pts_local[idx_unique]
                        # Bien doi sang world frame
                        c, s = math.cos(pose[2]), math.sin(pose[2])
                        world_pts = np.empty_like(pts_local)
                        world_pts[:, 0] = c * pts_local[:, 0] - s * pts_local[:, 1] + pose[0]
                        world_pts[:, 1] = s * pts_local[:, 0] + c * pts_local[:, 1] + pose[1]
                        # RAYCAST OCCUPANCY: pass pose lam sensor_xy de update
                        # ca free space lan hit -> vat di dong tu xoa
                        self.map.add_scan_with_raycast(
                            world_pts.astype(np.float32),
                            sensor_xy=(pose[0], pose[1]),
                        )
                    # Cap nhat pose lan cuoi update map
                    self._last_map_update_pose = (
                        float(pose[0]), float(pose[1]), float(pose[2]))
                    # Cap nhat last_pose moi scan de save dung pose hien tai
                    self.map.last_pose = np.array(
                        [float(pose[0]), float(pose[1]), float(pose[2])],
                        dtype=np.float32)
            except Exception as map_exc:
                print(f'[Map] update error: {type(map_exc).__name__}: {map_exc!r}')

            # Day map len web (it thuong xuyen hon cloud)
            self._publish_map()

            # Auto-save (hoac save thu cong tu web)
            self._try_save_if_due(pose)

            self._update_pose(
                pose, step, error, matches, overlap, scan_count, accepted,
                rejected, hard_resets, points.shape[0], map_points,
            )


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

    y0 = put("=== CAMERA BASED WEB TEST V2 ===", y0, cyan)
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

    y0 = put("--- Align V2 ---", y0, yellow)
    y0 = put(f"mode:  {info.get('align_mode', '---')}", y0)
    y0 = put(f"gain:  {info.get('align_gain_scale', 1.0): .2f}", y0)
    y0 = put(f"cross: {info.get('align_overshoots', 0)}", y0)
    y0 = put(f"rate:  {info.get('bearing_rate', 0): .3f}", y0)
    y0 += 5

    y0 = put("--- PID Omega ---", y0, yellow)
    y0 = put(f"P: {info.get('omega_p', 0): .3f}  I: {info.get('omega_i', 0): .3f}", y0)
    y0 = put(f"D: {info.get('omega_d', 0): .3f}", y0)
    y0 += 3

    y0 = put("--- PD Vy ---", y0, yellow)
    y0 = put(f"P: {info.get('vy_p', 0): .3f}", y0)
    y0 = put(f"D: {info.get('vy_d', 0): .3f}", y0)
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
    <title>Camera Based Web Test V2</title>
  <style>
    :root {
      --bg: #10110f;
      --panel: #181a17;
      --panel-2: #20231f;
      --text: #f3f5ef;
      --muted: #aeb6aa;
      --accent: #2fc5a8;
      --accent-2: #f0b84d;
      --danger: #d96a5f;
      --line: #30362f;
    }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, "Segoe UI", Arial, sans-serif;
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
            background: var(--panel-2);
            color: var(--text);
            padding: 10px 14px;
            border-radius: 8px;
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
        .button.compact {
            padding: 8px 10px;
            font-size: 12px;
        }
        .button.full {
            width: 100%;
        }
        .button.active {
            border-color: var(--accent);
            color: var(--accent);
        }
        .button.danger {
            border-color: var(--danger);
            color: #ffd8d3;
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
    .side {
      display: grid;
      gap: 16px;
      align-content: start;
    }
    .panel {
      background: rgba(24, 26, 23, 0.95);
      border: 1px solid var(--line);
      border-radius: 8px;
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
    .mapbox {
      padding: 14px;
    }
    .mapbox h2 {
      margin: 0 0 12px;
      font-size: 14px;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .section-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .section-head h2 {
      margin-bottom: 4px;
    }
    .map-status,
    .map-selection,
    .empty {
      color: var(--muted);
      font-size: 12px;
    }
    .map-toolbar,
    .waypoint-actions,
    .waypoint-editor {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .waypoint-editor {
      margin-top: 10px;
      flex-wrap: wrap;
    }
    .heading-input {
      width: 90px;
      flex: none;
    }
    .heading-hint {
      font-size: 11px;
      color: var(--muted);
      width: 100%;
      margin-top: -2px;
    }
    .text-input {
      min-width: 0;
      flex: 1;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0d0f0c;
      color: var(--text);
      padding: 10px 12px;
      font: inherit;
      font-size: 13px;
    }
    .map-selection {
      margin-top: 8px;
      min-height: 16px;
    }
    .waypoint-list {
      display: grid;
      gap: 8px;
      margin: 12px 0;
      max-height: 220px;
      overflow: auto;
    }
    .waypoint-item {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 10px;
      border: 1px solid rgba(255, 255, 255, 0.07);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.03);
    }
    .waypoint-title {
      font-size: 13px;
      font-weight: 700;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .waypoint-meta {
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
    }
    .mini-button {
      appearance: none;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #11130f;
      color: var(--text);
      padding: 7px 9px;
      cursor: pointer;
      font: inherit;
      font-size: 12px;
    }
    .mini-button.go {
      border-color: var(--accent);
      color: var(--accent);
    }
    .mini-button.delete {
      border-color: var(--danger);
      color: #ffd8d3;
    }
    #lidarMap {
      width: 100%;
      aspect-ratio: 1 / 1;
      display: block;
      background: #070a0f;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    #lidarMap.is-marking {
      cursor: crosshair;
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
        <div class="title">Camera Based Web Test V2</div>
        <div class="hint">Mo tren may khac trong cung mang LAN de xem stream tu Raspberry Pi.</div>
                <div class="actions">
                    <button class="button" data-kind="start" id="startBtn">Start Motor</button>
                    <button class="button" data-kind="stop" id="stopBtn">Stop Motor</button>
                    <button class="button" data-kind="save" id="saveMapBtn">Save Map</button>
                </div>
      </div>
      <div class="hint" id="endpoint"></div>
    </div>
    <div class="grid">
      <div class="panel">
        <img class="stream" src="/stream.mjpg" alt="stream">
      </div>
      <div class="side">
        <div class="panel stats">
          <h2>Live State</h2>
          <div id="stats"></div>
        </div>
        <div class="panel mapbox">
          <div class="section-head">
            <div>
              <h2>Map Locations</h2>
              <div class="map-status" id="mapMeta">-</div>
            </div>
            <div class="map-toolbar">
              <button class="button compact" id="addPointBtn">Add Point</button>
              <button class="button compact" id="cancelDraftBtn">Clear</button>
            </div>
          </div>
          <canvas id="lidarMap" width="420" height="420"></canvas>
          <div class="waypoint-editor">
            <input class="text-input" id="waypointLabel" maxlength="48" placeholder="Label">
            <input class="text-input heading-input" id="waypointHeading" type="number" step="1" placeholder="deg">
            <button class="button compact" id="saveWaypointBtn">Save Point</button>
            <div class="heading-hint">Heading deg (0 = +X, 90 = +Y). Bo trong = giu huong hien tai cua robot khi den noi.</div>
          </div>
          <div class="map-selection" id="mapSelection">Draft: -</div>
          <div class="waypoint-list" id="waypointList"></div>
          <button class="button compact full danger" id="cancelNavBtn">Cancel Navigation</button>
        </div>
      </div>
    </div>
  </div>
  <script>
    const statsEl = document.getElementById('stats');
        const lidarCanvas = document.getElementById('lidarMap');
        const lidarCtx = lidarCanvas.getContext('2d');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const saveMapBtn = document.getElementById('saveMapBtn');
        const mapMeta = document.getElementById('mapMeta');
        const addPointBtn = document.getElementById('addPointBtn');
        const cancelDraftBtn = document.getElementById('cancelDraftBtn');
        const waypointLabel = document.getElementById('waypointLabel');
        const waypointHeading = document.getElementById('waypointHeading');
        const saveWaypointBtn = document.getElementById('saveWaypointBtn');
        const waypointList = document.getElementById('waypointList');
        const mapSelection = document.getElementById('mapSelection');
        const cancelNavBtn = document.getElementById('cancelNavBtn');
        let latestState = {};
        let addMode = false;
        let draftPoint = null;
        let mapView = { scale: 1, cx: 0, cy: 0, pad: 18, hitboxes: [] };
    document.getElementById('endpoint').textContent = `${location.origin}/stream.mjpg`;

    function row(label, value, cls = '') {
      return `<div class="kv"><div class="label">${label}</div><div class="${cls}">${value}</div></div>`;
    }

    function num(value, digits = 3) {
      const n = Number(value);
      return Number.isFinite(n) ? n.toFixed(digits) : '-';
    }

    function escapeHtml(value) {
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function drawLidarMap(state) {
      const ctx = lidarCtx;
      const w = lidarCanvas.width;
      const h = lidarCanvas.height;
      const pad = 18;
      const cx = w / 2;
      const cy = h / 2;
      const path = Array.isArray(state.lidar_path) ? state.lidar_path : [];
      const cloud = Array.isArray(state.lidar_cloud) ? state.lidar_cloud : [];
      const gmap = Array.isArray(state.lidar_map) ? state.lidar_map : [];
      const waypoints = Array.isArray(state.waypoints) ? state.waypoints : [];
      const navTarget = state.navigation_target || null;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#070a0f';
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = '#1f2a37';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 8; i++) {
        const p = pad + i * (w - 2 * pad) / 8;
        ctx.beginPath();
        ctx.moveTo(p, pad);
        ctx.lineTo(p, h - pad);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(pad, p);
        ctx.lineTo(w - pad, p);
        ctx.stroke();
      }

      // Tinh maxAbs bao gom ca cloud + gmap + path + pose hien tai
      let maxAbs = 0.5;
      for (const p of path) {
        maxAbs = Math.max(maxAbs, Math.abs(p[0] || 0), Math.abs(p[1] || 0));
      }
      for (const wp of waypoints) {
        maxAbs = Math.max(maxAbs, Math.abs(wp.x || 0), Math.abs(wp.y || 0));
      }
      if (draftPoint) {
        maxAbs = Math.max(maxAbs, Math.abs(draftPoint.x || 0), Math.abs(draftPoint.y || 0));
      }
      if (navTarget) {
        maxAbs = Math.max(maxAbs, Math.abs(navTarget.x || 0), Math.abs(navTarget.y || 0));
      }
      maxAbs = Math.max(maxAbs, Math.abs(state.lidar_x || 0), Math.abs(state.lidar_y || 0));
      // Cloud co the rat lon -> lay percentile xap xi (max 95%) de tranh phong to qua muc
      if (cloud.length > 0) {
        let cloudMax = 0;
        for (const p of cloud) {
          const v = Math.max(Math.abs(p[0]), Math.abs(p[1]));
          if (v > cloudMax) cloudMax = v;
        }
        // Khong cho cloud lam scale qua nho (giu pose ro)
        maxAbs = Math.max(maxAbs, Math.min(cloudMax, 8.0));
      }
      // Tinh ca global map: dung 90th percentile de tranh outlier dan
      if (gmap.length > 0) {
        const xs = gmap.map(p => Math.max(Math.abs(p[0]), Math.abs(p[1]))).sort((a, b) => a - b);
        const idx90 = Math.floor(xs.length * 0.9);
        const mapMax = xs[Math.min(idx90, xs.length - 1)] || 0;
        maxAbs = Math.max(maxAbs, Math.min(mapMax, 12.0));
      }
      const scale = (w / 2 - pad) / maxAbs;
      const toPx = (x, y) => [cx + x * scale, cy - y * scale];
      mapView = { scale, cx, cy, pad, hitboxes: [], toPx };

      // ===== Ve global map TRUOC (mau xam, lop nen) =====
      if (gmap.length > 0) {
        ctx.fillStyle = 'rgba(180, 180, 200, 0.55)';
        for (const p of gmap) {
          const px = cx + (p[0] || 0) * scale;
          const py = cy - (p[1] || 0) * scale;
          if (px < pad || px > w - pad || py < pad || py > h - pad) continue;
          ctx.fillRect(px - 1, py - 1, 2, 2);
        }
      }

      // ===== Ve point cloud song (mau cyan, lop tren) =====
      // Moi diem co alpha (cot 2): diem moi sang, diem cu mo dan.
      if (cloud.length > 0) {
        for (const p of cloud) {
          const px = cx + (p[0] || 0) * scale;
          const py = cy - (p[1] || 0) * scale;
          if (px < pad || px > w - pad || py < pad || py > h - pad) continue;
          const a = Math.max(0.1, Math.min(1.0, p[2] || 1.0));
          // Mau xanh cyan voi do trong suot theo alpha
          ctx.fillStyle = `rgba(0, 220, 255, ${a.toFixed(2)})`;
          ctx.fillRect(px - 1, py - 1, 2, 2);
        }
      }

      ctx.strokeStyle = '#2d4258';
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.stroke();

      if (path.length > 1) {
        ctx.strokeStyle = '#f3c969';
        ctx.lineWidth = 2;
        ctx.beginPath();
        path.forEach((p, idx) => {
          const [px, py] = toPx(p[0] || 0, p[1] || 0);
          if (idx === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      }

      const drawPin = (x, y, label, color, active, id, theta) => {
        const [px, py] = toPx(x || 0, y || 0);
        if (px < pad || px > w - pad || py < pad || py > h - pad) return;
        if (id) mapView.hitboxes.push({ id, x: px, y: py });
        ctx.save();
        // Ve mui ten heading neu co theta
        if (theta !== null && theta !== undefined && isFinite(theta)) {
          const arrowLen = 18;
          // Trong he world: theta = 0 -> +X. Tren canvas y dao chieu, nen ve theta=0 -> phai
          const ax = px + Math.cos(theta) * arrowLen;
          const ay = py - Math.sin(theta) * arrowLen;
          ctx.strokeStyle = color;
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(ax, ay);
          ctx.stroke();
          // Dau mui ten
          const headSize = 5;
          const angle = Math.atan2(ay - py, ax - px);
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(ax - headSize * Math.cos(angle - Math.PI / 6),
                     ay - headSize * Math.sin(angle - Math.PI / 6));
          ctx.moveTo(ax, ay);
          ctx.lineTo(ax - headSize * Math.cos(angle + Math.PI / 6),
                     ay - headSize * Math.sin(angle + Math.PI / 6));
          ctx.stroke();
        }
        ctx.fillStyle = color;
        ctx.strokeStyle = active ? '#ffffff' : '#0b0d0a';
        ctx.lineWidth = active ? 3 : 2;
        ctx.beginPath();
        ctx.arc(px, py, active ? 8 : 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.font = '12px "Segoe UI", Arial, sans-serif';
        const fullLabel = String(label || 'Point');
        const safeLabel = fullLabel.length > 18 ? `${fullLabel.slice(0, 17)}.` : fullLabel;
        const labelW = Math.min(120, ctx.measureText(safeLabel).width + 14);
        const lx = Math.min(px + 10, w - pad - labelW);
        const ly = Math.max(pad + 18, py - 9);
        ctx.fillStyle = 'rgba(13, 15, 12, 0.86)';
        ctx.fillRect(lx, ly - 14, labelW, 20);
        ctx.fillStyle = '#f3f5ef';
        ctx.fillText(safeLabel, lx + 7, ly);
        ctx.restore();
      };

      if (navTarget) {
        const [tx, ty] = toPx(navTarget.x || 0, navTarget.y || 0);
        const [rxLine, ryLine] = toPx(state.lidar_x || 0, state.lidar_y || 0);
        ctx.strokeStyle = state.navigation_active ? '#2fc5a8' : '#f0b84d';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 5]);
        ctx.beginPath();
        ctx.moveTo(rxLine, ryLine);
        ctx.lineTo(tx, ty);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      for (const wp of waypoints) {
        const active = state.navigation_active && navTarget && navTarget.id === wp.id;
        drawPin(wp.x, wp.y, wp.label, active ? '#2fc5a8' : '#f0b84d', active, wp.id,
                wp.theta != null ? wp.theta : null);
      }
      if (draftPoint) {
        drawPin(draftPoint.x, draftPoint.y, 'Draft', '#d96a5f', true, null,
                draftPoint.theta != null ? draftPoint.theta : null);
      }

      const x = state.lidar_x || 0;
      const y = state.lidar_y || 0;
      // Uu tien filtered theta (filtered LiDAR theta) khi dang nav, fallback sang LiDAR raw
      let thetaDeg;
      if (state.heading_effective_deg !== undefined && state.heading_effective_deg !== null) {
        thetaDeg = state.heading_effective_deg;
      } else {
        thetaDeg = state.lidar_theta_deg || 0;
      }
      const theta = thetaDeg * Math.PI / 180;
      const [rx, ry] = toPx(x, y);
      // Mui ten heading TO HON va RO RANG hon
      const headingLen = 42;
      const hx = rx + Math.cos(theta) * headingLen;
      const hy = ry - Math.sin(theta) * headingLen;

      // 1) Vong tron BAN KINH THUC TE 0.5m va 1.0m quanh robot
      // Bien doi tu met sang pixel theo `scale` da tinh o tren
      const ring1Radius = 0.5 * scale;  // 0.5 met thuc te
      const ring2Radius = 1.0 * scale;  // 1.0 met thuc te
      // Vong 0.5m
      ctx.strokeStyle = 'rgba(131, 242, 143, 0.22)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(rx, ry, ring1Radius, 0, Math.PI * 2);
      ctx.stroke();
      // Vong 1.0m
      ctx.strokeStyle = 'rgba(131, 242, 143, 0.15)';
      ctx.beginPath();
      ctx.arc(rx, ry, ring2Radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      // Label "0.5m" va "1.0m" o phia tren-phai cua moi vong
      ctx.fillStyle = 'rgba(131, 242, 143, 0.55)';
      ctx.font = '9px Consolas, monospace';
      ctx.fillText('0.5m', rx + ring1Radius * 0.71 + 2, ry - ring1Radius * 0.71);
      ctx.fillText('1.0m', rx + ring2Radius * 0.71 + 2, ry - ring2Radius * 0.71);

      // 2) Vach mui ten chi 4 huong la ban (kich thuoc co dinh theo pixel)
      const compassR1 = headingLen + 4;
      const compassR2 = headingLen + 10;
      for (let deg = 0; deg < 360; deg += 90) {
        const rad = deg * Math.PI / 180;
        const tx1 = rx + Math.cos(rad) * compassR1;
        const ty1 = ry - Math.sin(rad) * compassR1;
        const tx2 = rx + Math.cos(rad) * compassR2;
        const ty2 = ry - Math.sin(rad) * compassR2;
        ctx.strokeStyle = 'rgba(131, 242, 143, 0.4)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(tx1, ty1);
        ctx.lineTo(tx2, ty2);
        ctx.stroke();
        // Label do (chi cho 4 huong chinh)
        ctx.fillStyle = 'rgba(131, 242, 143, 0.55)';
        ctx.font = '10px Consolas, monospace';
        const labelR = headingLen + 18;
        const lx = rx + Math.cos(rad) * labelR - 6;
        const ly = ry - Math.sin(rad) * labelR + 4;
        ctx.fillText(`${deg}`, lx, ly);
      }

      // 3) Diem robot
      ctx.fillStyle = '#83f28f';
      ctx.beginPath();
      ctx.arc(rx, ry, 6, 0, Math.PI * 2);
      ctx.fill();

      // 3) Mui ten heading lon (truc + dau mui ten)
      ctx.strokeStyle = '#83f28f';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(rx, ry);
      ctx.lineTo(hx, hy);
      ctx.stroke();
      // Dau mui ten
      const arrowHead = 9;
      const arrowAngle = Math.atan2(hy - ry, hx - rx);
      ctx.beginPath();
      ctx.moveTo(hx, hy);
      ctx.lineTo(
        hx - arrowHead * Math.cos(arrowAngle - Math.PI / 6),
        hy - arrowHead * Math.sin(arrowAngle - Math.PI / 6)
      );
      ctx.moveTo(hx, hy);
      ctx.lineTo(
        hx - arrowHead * Math.cos(arrowAngle + Math.PI / 6),
        hy - arrowHead * Math.sin(arrowAngle + Math.PI / 6)
      );
      ctx.stroke();

      // 4) Hien thi so do ngay canh mui ten (chuan hoa ve [-180, 180])
      let displayDeg = thetaDeg;
      while (displayDeg > 180) displayDeg -= 360;
      while (displayDeg < -180) displayDeg += 360;
      const headingText = `${displayDeg >= 0 ? '+' : ''}${displayDeg.toFixed(1)} deg`;
      // Background cho de doc
      ctx.font = 'bold 13px Consolas, monospace';
      const textW = ctx.measureText(headingText).width + 10;
      const textBgX = hx + 6;
      const textBgY = hy - 9;
      ctx.fillStyle = 'rgba(13, 15, 12, 0.85)';
      ctx.fillRect(textBgX, textBgY, textW, 18);
      ctx.fillStyle = '#83f28f';
      ctx.fillText(headingText, textBgX + 5, textBgY + 13);

      // 5) Bang HEADING TO o goc tren ben trai canvas (de nguoi dung set huong tiep theo)
      const panelW = 130;
      const panelH = 50;
      const panelX = pad;
      const panelY = h - panelH - pad;
      ctx.fillStyle = 'rgba(13, 15, 12, 0.88)';
      ctx.fillRect(panelX, panelY, panelW, panelH);
      ctx.strokeStyle = '#83f28f';
      ctx.lineWidth = 1;
      ctx.strokeRect(panelX, panelY, panelW, panelH);
      ctx.fillStyle = '#9cb0c1';
      ctx.font = '10px Consolas, monospace';
      ctx.fillText('HEADING', panelX + 8, panelY + 14);
      ctx.fillStyle = '#83f28f';
      ctx.font = 'bold 22px Consolas, monospace';
      ctx.fillText(`${displayDeg.toFixed(1)}°`, panelX + 8, panelY + 38);
      // Hien thi compass direction (N/E/S/W) - he world: 0deg = +X = East
      let compass = 'E';
      if (displayDeg > 45 && displayDeg <= 135) compass = 'N';
      else if (displayDeg > 135 || displayDeg <= -135) compass = 'W';
      else if (displayDeg > -135 && displayDeg <= -45) compass = 'S';
      ctx.fillStyle = '#9cb0c1';
      ctx.font = 'bold 16px Consolas, monospace';
      ctx.fillText(compass, panelX + panelW - 24, panelY + 38);

      ctx.fillStyle = '#9cb0c1';
      ctx.font = '12px Consolas, monospace';
      ctx.fillText(`${maxAbs.toFixed(1)} m`, pad, h - 8);
      ctx.fillText(state.lidar_status || 'DISABLED', pad, 16);
      ctx.fillText(`live: ${cloud.length}`, w - pad - 80, 16);
      ctx.fillText(`map: ${state.lidar_map_total || 0}`, w - pad - 80, 32);
      if (state.lidar_map_loaded) {
        ctx.fillStyle = state.lidar_map_relocalized ? '#83f28f' : '#f3c969';
        ctx.fillText(state.lidar_map_relocalized ? 'RELOC OK' : 'RELOC...', pad, 32);
      }
      // Hien thi nav phase neu dang navigation
      if (state.navigation_active) {
        ctx.fillStyle = '#2fc5a8';
        ctx.font = 'bold 13px Consolas, monospace';
        ctx.fillText(`NAV: ${state.navigation_phase || '?'}`, pad, h - panelH - pad - 8);
      }
    }

    function setAddMode(enabled) {
      addMode = Boolean(enabled);
      addPointBtn.classList.toggle('active', addMode);
      lidarCanvas.classList.toggle('is-marking', addMode);
    }

    function updateDraftText(text) {
      if (text) {
        mapSelection.textContent = text;
      } else if (draftPoint) {
        mapSelection.textContent = `Draft: ${num(draftPoint.x)}, ${num(draftPoint.y)} m`;
      } else {
        mapSelection.textContent = 'Draft: -';
      }
    }

    function renderWaypointList(state) {
      const waypoints = Array.isArray(state.waypoints) ? state.waypoints : [];
      if (!waypoints.length) {
        waypointList.innerHTML = '<div class="empty">No saved locations</div>';
        return;
      }
      waypointList.innerHTML = waypoints.map((wp) => {
        const active = state.navigation_active && state.navigation_target && state.navigation_target.id === wp.id;
        const headingTxt = (wp.theta != null && isFinite(wp.theta))
          ? `, ${(wp.theta * 180 / Math.PI).toFixed(0)}deg`
          : ', free heading';
        return `
          <div class="waypoint-item">
            <div>
              <div class="waypoint-title">${escapeHtml(wp.label)}${active ? ' - active' : ''}</div>
              <div class="waypoint-meta">${num(wp.x)}, ${num(wp.y)} m${headingTxt}</div>
            </div>
            <div class="waypoint-actions">
              <button class="mini-button go" data-go="${escapeHtml(wp.id)}">Go</button>
              <button class="mini-button delete" data-delete="${escapeHtml(wp.id)}">X</button>
            </div>
          </div>
        `;
      }).join('');
    }

    function canvasPoint(event) {
      const rect = lidarCanvas.getBoundingClientRect();
      return {
        px: (event.clientX - rect.left) * lidarCanvas.width / rect.width,
        py: (event.clientY - rect.top) * lidarCanvas.height / rect.height,
      };
    }

    function worldFromCanvas(point) {
      return {
        x: (point.px - mapView.cx) / mapView.scale,
        y: (mapView.cy - point.py) / mapView.scale,
      };
    }

    function waypointHit(point) {
      let best = null;
      let bestDist = 15;
      for (const hit of mapView.hitboxes || []) {
        const dist = Math.hypot(point.px - hit.x, point.py - hit.y);
        if (dist < bestDist) {
          best = hit;
          bestDist = dist;
        }
      }
      return best;
    }

    async function postJson(url, payload = {}) {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      let data = {};
      try {
        data = await response.json();
      } catch (error) {
        data = {};
      }
      return { ...data, httpOk: response.ok };
    }

    async function startNavigation(id) {
      const result = await postJson('/api/navigation/start', { id });
      updateDraftText(result.ok ? 'Navigation: started' : `Navigation: ${result.reason || 'failed'}`);
      await refresh();
    }

    async function deleteWaypoint(id) {
      await postJson('/api/waypoints/delete', { id });
      await refresh();
    }

    async function cancelNavigation() {
      await postJson('/api/navigation/cancel');
      await refresh();
    }

    async function saveWaypoint() {
      if (!draftPoint) return;
      saveWaypointBtn.disabled = true;
      // Parse heading: bo trong = null (khong dat huong)
      const headingRaw = waypointHeading.value.trim();
      const payload = {
        label: waypointLabel.value,
        x: draftPoint.x,
        y: draftPoint.y,
      };
      if (headingRaw !== '') {
        const headingNum = parseFloat(headingRaw);
        if (!isNaN(headingNum) && isFinite(headingNum)) {
          payload.theta_deg = headingNum;
        }
      }
      const result = await postJson('/api/waypoints', payload);
      if (result.ok) {
        draftPoint = null;
        waypointLabel.value = '';
        waypointHeading.value = '';
        setAddMode(false);
        updateDraftText('Saved point');
      } else {
        updateDraftText(`Save failed: ${result.reason || 'invalid point'}`);
      }
      saveWaypointBtn.disabled = false;
      await refresh();
    }

    async function refresh() {
      try {
        const response = await fetch('/state');
        const state = await response.json();
        latestState = state;
        const statusClass = (state.navigation_active || state.ball_detected || state.status === 'ARRIVED') ? 'ok' : 'warn';
        const waypoints = Array.isArray(state.waypoints) ? state.waypoints : [];
        const navTarget = state.navigation_target || null;
                startBtn.disabled = !state.serial_enabled || state.motor_enabled;
                stopBtn.disabled = !state.serial_enabled || !state.motor_enabled;
                cancelNavBtn.disabled = !state.navigation_active;
                saveWaypointBtn.disabled = !draftPoint;
                mapMeta.textContent = `${state.lidar_status || 'DISABLED'} | ${waypoints.length} saved`;
        statsEl.innerHTML = [
          row('status', state.status, statusClass),
          row('serial', state.serial_enabled ? 'enabled' : 'disabled'),
                    row('motor', state.motor_enabled ? 'armed' : 'stopped', state.motor_enabled ? 'ok' : 'warn'),
                    row('reason', state.motor_reason),
          row('vx / vy', `${num(state.vx)} / ${num(state.vy)}`),
          row('omega', num(state.omega)),
          row('err_x', num(state.err_x)),
          row('err_dist', num(state.err_dist)),
          row('dist m', num(state.dist_m)),
          row('bbox_h', num(state.kf_bh, 1)),
          row('bbox target', num(state.target_bbox_h, 1)),
          row('base vx', num(state.measured_vx)),
          row('base wz', num(state.measured_wz)),
          row('lidar', state.lidar_status, state.lidar_status === 'TRACKING' ? 'ok' : 'warn'),
          row('lidar pose', `${num(state.lidar_x)}, ${num(state.lidar_y)}, ${num(state.lidar_theta_deg, 1)} deg`),
          row('lidar baud', state.lidar_baudrate || '-'),
          row('lidar scans', `${state.lidar_accepted}/${state.lidar_scans}`),
          row('lidar err', `${num(state.lidar_error)} m`),
          row('cloud pts', `${state.lidar_cloud_count || 0}`),
          row('map pts', `${state.lidar_map_total || 0}`),
          row('map file', state.lidar_map_loaded ? 'loaded' : 'fresh', state.lidar_map_loaded ? 'ok' : 'warn'),
          row('reloc', state.lidar_map_loaded ? (state.lidar_map_relocalized ? 'success' : 'pending') : 'n/a',
              state.lidar_map_relocalized ? 'ok' : 'warn'),
          row('last save', state.lidar_map_last_save ? new Date(state.lidar_map_last_save * 1000).toLocaleTimeString() : '-'),
          row('waypoints', waypoints.length),
          row('nav', state.navigation_status || 'IDLE', state.navigation_active ? 'ok' : ''),
          row('phase', state.navigation_phase || 'IDLE', state.navigation_active ? 'ok' : ''),
          row('goal', navTarget ? navTarget.label : '-'),
          row('goal dist', `${num(state.navigation_distance)} m`),
          row('heading err', `${num(state.navigation_bearing_error_deg, 1)} deg`),
          row('lidar note', state.lidar_error_text || '-'),
          row('vision fps', num(state.fps_vision, 1)),
          row('control fps', num(state.fps_ctrl, 1)),
          row('last seen', `${num(state.last_seen_age, 2)} s`),
        ].join('');
        drawLidarMap(state);
        renderWaypointList(state);
        updateDraftText();
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
        saveMapBtn.addEventListener('click', async () => {
          saveMapBtn.disabled = true;
          saveMapBtn.textContent = 'Saving...';
          try {
            const r = await fetch('/api/map/save', { method: 'POST' });
            const j = await r.json();
            saveMapBtn.textContent = j.ok ? 'Saved!' : 'Failed';
          } catch (e) {
            saveMapBtn.textContent = 'Error';
          }
          setTimeout(() => {
            saveMapBtn.textContent = 'Save Map';
            saveMapBtn.disabled = false;
          }, 1500);
        });

        addPointBtn.addEventListener('click', () => {
          setAddMode(!addMode);
        });

        cancelDraftBtn.addEventListener('click', () => {
          draftPoint = null;
          setAddMode(false);
          updateDraftText();
          drawLidarMap(latestState);
        });

        saveWaypointBtn.addEventListener('click', saveWaypoint);
        cancelNavBtn.addEventListener('click', cancelNavigation);

        lidarCanvas.addEventListener('click', async (event) => {
          const point = canvasPoint(event);
          const hit = waypointHit(point);
          if (hit && !addMode) {
            await startNavigation(hit.id);
            return;
          }
          draftPoint = worldFromCanvas(point);
          // Sync heading hien tai vao draft (de live preview)
          updateDraftHeading();
          setAddMode(true);
          updateDraftText();
          drawLidarMap(latestState);
        });

        // Khi user nhap so vao input theta -> cap nhat draftPoint.theta de live preview
        function updateDraftHeading() {
          if (!draftPoint) return;
          const raw = waypointHeading.value.trim();
          if (raw === '') {
            draftPoint.theta = null;
          } else {
            const num = parseFloat(raw);
            draftPoint.theta = (isNaN(num) || !isFinite(num)) ? null : num * Math.PI / 180;
          }
        }
        waypointHeading.addEventListener('input', () => {
          updateDraftHeading();
          drawLidarMap(latestState);
        });

        waypointList.addEventListener('click', async (event) => {
          const goBtn = event.target.closest('[data-go]');
          const deleteBtn = event.target.closest('[data-delete]');
          if (goBtn) {
            await startNavigation(goBtn.dataset.go);
          } else if (deleteBtn) {
            await deleteWaypoint(deleteBtn.dataset.delete);
          }
        });

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
        self.lidar_mgr = None
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
        self.waypoint_store = None
        self.navigation_active = False
        self.navigation_target = None
        self.stats = {
            'status': 'SEARCHING',
            'ball_detected': False,
            'serial_enabled': False,
            'motor_enabled': False,
            'motor_reason': 'Serial disabled',
            'vx': 0.0,
            'vx_target': 0.0,
            'vy': 0.0,
            'vy_target': 0.0,
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
            'heading_source': 'lidar',
            'heading_bias_deg': 0.0,
            'heading_std_deg': 0.0,
            'lidar_enabled': False,
            'lidar_status': 'DISABLED',
            'lidar_error_text': '',
            'lidar_baudrate': 0,
            'lidar_x': 0.0,
            'lidar_y': 0.0,
            'lidar_theta_deg': 0.0,
            'heading_effective_deg': 0.0,
            'lidar_step_dx': 0.0,
            'lidar_step_dy': 0.0,
            'lidar_step_dtheta_deg': 0.0,
            'lidar_error': 0.0,
            'lidar_matches': 0,
            'lidar_overlap': 0.0,
            'lidar_scans': 0,
            'lidar_accepted': 0,
            'lidar_rejected': 0,
            'lidar_hard_resets': 0,
            'lidar_points': 0,
            'lidar_map_points': 0,
            'lidar_path': [],
            'lidar_cloud': [],
            'lidar_cloud_count': 0,
            'lidar_map': [],
            'lidar_map_total': 0,
            'lidar_map_loaded': False,
            'lidar_map_relocalized': False,
            'lidar_map_last_save': 0.0,
            'lidar_map_save_ok': False,
            'lidar_last_update': 0.0,
            'waypoints': [],
            'waypoints_file': WAYPOINTS_FILE,
            'navigation_active': False,
            'navigation_target': None,
            'navigation_status': 'IDLE',
            'navigation_phase': 'IDLE',
            'navigation_reason': '',
            'navigation_distance': 0.0,
            'navigation_bearing_error_deg': 0.0,
            'align_mode': 'TURN',
            'align_gain_scale': 1.0,
            'align_overshoots': 0,
            'bearing_rate': 0.0,
            'omega_p': 0.0,
            'omega_i': 0.0,
            'omega_d': 0.0,
            'vy_p': 0.0,
            'vy_d': 0.0,
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
        if self.path == '/api/map/save':
            self._request_map_save()
            return
        if self.path == '/api/waypoints':
            self._create_waypoint()
            return
        if self.path == '/api/waypoints/delete':
            self._delete_waypoint()
            return
        if self.path == '/api/navigation/start':
            self._start_navigation()
            return
        if self.path == '/api/navigation/cancel':
            self._cancel_navigation()
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

    def _read_json_body(self):
        try:
            length = int(self.headers.get('Content-Length', '0') or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        raw = self.rfile.read(min(length, 65536))
        try:
            return json.loads(raw.decode('utf-8'))
        except Exception:
            return {}

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _set_motor_enabled(self, enabled):
        with self.state.lock:
            serial_ready = self.state.stats.get('serial_enabled', False)
            self.state.motor_enabled = enabled and serial_ready
            self.state.stats['motor_enabled'] = self.state.motor_enabled
            if not enabled:
                self.state.navigation_active = False
                self.state.navigation_target = None
                self.state.stats['navigation_active'] = False
                self.state.stats['navigation_target'] = None
                self.state.stats['navigation_status'] = 'CANCELLED'
                self.state.stats['navigation_phase'] = 'IDLE'
                self.state.stats['navigation_reason'] = 'Stopped from web UI'
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

    def _request_map_save(self):
        lidar_mgr = self.state.lidar_mgr
        ok = False
        reason = ''
        if lidar_mgr is None:
            reason = 'LiDAR not enabled'
        else:
            try:
                lidar_mgr.request_save()
                ok = True
                reason = 'Save requested; will run on next scan'
            except Exception as exc:
                reason = f'{type(exc).__name__}: {exc!r}'
        payload = {'ok': ok, 'reason': reason}
        body = json.dumps(payload).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _create_waypoint(self):
        store = self.state.waypoint_store
        if store is None:
            self._send_json({'ok': False, 'reason': 'Waypoint store not ready'}, 503)
            return
        data = self._read_json_body()
        try:
            x = float(data.get('x'))
            y = float(data.get('y'))
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError('Coordinates must be finite')
        except Exception:
            self._send_json({'ok': False, 'reason': 'Invalid waypoint coordinates'}, 400)
            return
        # theta la TUY CHON: client co the gui dang radians (theta) hoac do (theta_deg)
        theta_val = None
        if 'theta' in data and data.get('theta') is not None:
            try:
                theta_val = float(data.get('theta'))
                if not math.isfinite(theta_val):
                    theta_val = None
            except (TypeError, ValueError):
                theta_val = None
        elif 'theta_deg' in data and data.get('theta_deg') is not None:
            try:
                theta_deg = float(data.get('theta_deg'))
                if math.isfinite(theta_deg):
                    theta_val = math.radians(theta_deg)
            except (TypeError, ValueError):
                theta_val = None
        try:
            waypoint = store.add(data.get('label', ''), x, y, theta=theta_val)
            waypoints = store.list()
            with self.state.lock:
                self.state.stats['waypoints'] = waypoints
            self._send_json({'ok': True, 'waypoint': waypoint, 'waypoints': waypoints})
        except Exception as exc:
            self._send_json({'ok': False, 'reason': f'{type(exc).__name__}: {exc!r}'}, 500)

    def _delete_waypoint(self):
        store = self.state.waypoint_store
        if store is None:
            self._send_json({'ok': False, 'reason': 'Waypoint store not ready'}, 503)
            return
        data = self._read_json_body()
        waypoint_id = str(data.get('id') or '')
        if not waypoint_id:
            self._send_json({'ok': False, 'reason': 'Missing waypoint id'}, 400)
            return
        deleted = store.delete(waypoint_id)
        waypoints = store.list()
        should_stop = False
        with self.state.lock:
            active_target = self.state.navigation_target or {}
            if active_target.get('id') == waypoint_id:
                self.state.navigation_active = False
                self.state.navigation_target = None
                self.state.motor_enabled = False
                self.state.stats['motor_enabled'] = False
                self.state.stats['motor_reason'] = 'Waypoint deleted'
                self.state.stats['navigation_active'] = False
                self.state.stats['navigation_target'] = None
                self.state.stats['navigation_status'] = 'CANCELLED'
                self.state.stats['navigation_phase'] = 'IDLE'
                self.state.stats['navigation_reason'] = 'Waypoint deleted'
                should_stop = True
            self.state.stats['waypoints'] = waypoints
        if should_stop and self.state.serial_mgr is not None:
            self.state.serial_mgr.send_stop()
        self._send_json({'ok': deleted, 'deleted': deleted, 'waypoints': waypoints})

    def _start_navigation(self):
        store = self.state.waypoint_store
        data = self._read_json_body()
        waypoint = None
        waypoint_id = data.get('id')
        if waypoint_id and store is not None:
            waypoint = store.get(waypoint_id)
        if waypoint is None and 'x' in data and 'y' in data:
            try:
                waypoint = {
                    'id': str(data.get('id') or 'direct'),
                    'label': WaypointStore._clean_label(data.get('label'), 1),
                    'x': float(data.get('x')),
                    'y': float(data.get('y')),
                }
            except Exception:
                waypoint = None
        if waypoint is None:
            self._send_json({'ok': False, 'reason': 'Waypoint not found'}, 404)
            return

        now = time.monotonic()
        with self.state.lock:
            serial_ready = bool(self.state.stats.get('serial_enabled', False))
            lidar_status = self.state.stats.get('lidar_status', 'DISABLED')
            lidar_age = now - float(self.state.stats.get('lidar_last_update', 0.0) or 0.0)
            if not serial_ready:
                self.state.stats['navigation_status'] = 'WAIT_MOTOR'
                self.state.stats['navigation_reason'] = 'Serial not connected'
                payload = {'ok': False, 'reason': 'Serial not connected'}
            elif lidar_status != 'TRACKING' or lidar_age > NAV_LIDAR_STALE_S:
                self.state.stats['navigation_status'] = 'WAIT_LIDAR'
                self.state.stats['navigation_reason'] = 'LiDAR pose is not ready'
                payload = {'ok': False, 'reason': 'LiDAR pose is not ready'}
            else:
                # theta la TUY CHON: tu waypoint (load tu file) hoac tu request body
                wp_theta = waypoint.get('theta', None)
                # Cho phep override theta tu request body
                if 'theta' in data and data.get('theta') is not None:
                    try:
                        wp_theta = float(data.get('theta'))
                        if not math.isfinite(wp_theta):
                            wp_theta = None
                    except (TypeError, ValueError):
                        wp_theta = None
                elif 'theta_deg' in data and data.get('theta_deg') is not None:
                    try:
                        td = float(data.get('theta_deg'))
                        if math.isfinite(td):
                            wp_theta = math.radians(td)
                    except (TypeError, ValueError):
                        pass
                target = {
                    'id': str(waypoint.get('id', 'direct')),
                    'label': str(waypoint.get('label') or 'Point'),
                    'x': float(waypoint.get('x', 0.0)),
                    'y': float(waypoint.get('y', 0.0)),
                    'theta': float(wp_theta) if wp_theta is not None else None,
                }
                self.state.navigation_target = target
                self.state.navigation_active = True
                self.state.motor_enabled = True
                self.state.stats['motor_enabled'] = True
                self.state.stats['motor_reason'] = f'Navigating to {target["label"]}'
                self.state.stats['navigation_active'] = True
                self.state.stats['navigation_target'] = target
                self.state.stats['navigation_status'] = 'TURN_TO_PATH'
                self.state.stats['navigation_phase'] = 'TURN_TO_PATH'
                self.state.stats['navigation_reason'] = ''
                payload = {'ok': True, 'target': target, 'reason': 'Navigation started'}
        self._send_json(payload, 200 if payload.get('ok') else 409)

    def _cancel_navigation(self):
        with self.state.lock:
            self.state.navigation_active = False
            self.state.navigation_target = None
            self.state.motor_enabled = False
            self.state.stats['motor_enabled'] = False
            self.state.stats['motor_reason'] = 'Navigation cancelled'
            self.state.stats['navigation_active'] = False
            self.state.stats['navigation_target'] = None
            self.state.stats['navigation_status'] = 'CANCELLED'
            self.state.stats['navigation_phase'] = 'IDLE'
            self.state.stats['navigation_reason'] = 'Cancelled from web UI'
            payload = {'ok': True, 'reason': 'Navigation cancelled'}

        if self.state.serial_mgr is not None:
            self.state.serial_mgr.send_stop()

        self._send_json(payload)


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
    parser = argparse.ArgumentParser(description='Camera based Web Test V2')
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
    parser.add_argument('--no-lidar', action='store_true')
    parser.add_argument('--lidar-port', type=str, default=LIDAR_PORT)
    parser.add_argument('--lidar-baudrate', type=int, default=LIDAR_BAUDRATE)
    parser.add_argument('--lidar-baudrates', type=str, default=None)
    parser.add_argument('--map-file', type=str, default=MAP_FILE,
                        help='File path de luu/tai persistent map (.npz)')
    parser.add_argument('--no-load-map', action='store_true',
                        help='Khong load map cu, bat dau ban do moi tu goc')
    parser.add_argument('--map-autosave', type=float, default=MAP_AUTOSAVE_PERIOD_S,
                        help='Chu ky autosave (giay). 0 = tat autosave')
    parser.add_argument('--waypoints-file', type=str, default=None,
                        help='File JSON de luu cac vi tri da gan nhan')
    parser.add_argument('--no-auto-detect', action='store_true',
                        help='Tat tu dong xac dinh port (su dung port mac dinh tu config)')
    args = parser.parse_args()

    app_state = SharedAppState()
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
        lidar_baudrates = tuple(int(value.strip()) for value in args.lidar_baudrates.split(',') if value.strip())
    else:
        lidar_baudrates = (args.lidar_baudrate,) + tuple(
            baudrate for baudrate in LIDAR_BAUDRATES if baudrate != args.lidar_baudrate
        )

    # ===== AUTO-DETECT PORTS =====
    # Tren Linux, ttyUSB0/ttyUSB1 co the doi nhau giua cac lan reboot.
    # Chay probe de xac dinh dau la LiDAR, dau la Arduino.
    # UU TIEN: tim LiDAR truoc, sau do Arduino lay port con lai.
    detected_lidar_port = None
    detected_arduino_port = None
    if not args.no_auto_detect and sys.platform.startswith('linux'):
        print('\n=== AUTO-DETECT USB PORTS ===')
        # Neu user ky lenh --serial-port hoac --lidar-port khac mac dinh -> dung lam preferred
        pref_lidar = args.lidar_port if args.lidar_port != LIDAR_PORT else None
        pref_arduino = args.serial_port
        detected_lidar_port, detected_arduino_port = auto_detect_ports(
            preferred_lidar=pref_lidar,
            preferred_arduino=pref_arduino,
            lidar_baudrates=lidar_baudrates,
        )
        # Apply detected ports
        if detected_lidar_port:
            args.lidar_port = detected_lidar_port
        if detected_arduino_port and args.serial_port is None:
            args.serial_port = detected_arduino_port
        print()

    serial_mgr = None
    use_serial = not args.no_serial
    if args.send_serial:
        use_serial = True

    if use_serial:
        # Neu da auto-detect ra Arduino port -> dung truc tiep
        if detected_arduino_port:
            port = detected_arduino_port
        else:
            preferred_serial_ports = [args.serial_port]
            if args.serial_port is None and ARDUINO_PORT is not None:
                preferred_serial_ports.append(ARDUINO_PORT)
            port = find_serial_port(
                preferred_ports=preferred_serial_ports,
                exclude_ports=[args.lidar_port])
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
    ctrl_omega_turn = AxisPD(TURN_KP, TURN_KD, -OMEGA_ALIGN_MAX, OMEGA_ALIGN_MAX)
    ctrl_vy_strafe = AxisPD(STRAFE_KP, STRAFE_KD, -VY_ALIGN_MAX, VY_ALIGN_MAX)
    ctrl_omega_hold = AxisPD(HOLD_KP, HOLD_KD, -HOLD_OMEGA_MAX, HOLD_OMEGA_MAX)
    pid_vx = PID(VX_KP, VX_KI, VX_KD, -VX_MAX, VX_MAX, derivative_tau=0.10)
    vx_limiter = SlewRateLimiter(VX_SLEW_MPS_S)
    vy_limiter = SlewRateLimiter(VY_SLEW_MPS_S)
    omega_limiter = SlewRateLimiter(OMEGA_SLEW_RAD_S2)
    align_mode = 'TURN'
    align_tuner = AlignmentAutoTuner()

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

    lidar_mgr = None
    if args.no_lidar:
        app_state.stats['lidar_status'] = 'DISABLED'
    else:
        # Neu --no-load-map, doi ten file de bo qua viec load (giu file cu nguyen)
        map_file_arg = args.map_file
        if args.no_load_map:
            # Dung mot file rac de skip load nhung van save sang file moi
            map_file_arg = args.map_file
            # Don gian: tam thoi xoa truoc khi load - hoac chi can flag
        lidar_mgr = LidarPoseManager(
            app_state, args.lidar_port, lidar_baudrates,
            map_file=map_file_arg,
            autosave_period=args.map_autosave,
        )
        if args.no_load_map:
            # Chuyen flag de _process_scans biet skip load
            lidar_mgr._skip_load = True
        app_state.lidar_mgr = lidar_mgr
        lidar_mgr.start()

    print('\n=== CAMERA BASED WEB TEST V2 ===')
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
    # ===== Nav phase machine state =====
    # 'TURN_TO_PATH'  : robot quay tai cho de huong ve goal
    # 'DRIVE'         : tien thang ve goal, giu heading
    # 'TURN_TO_GOAL'  : (chi neu co goal_theta) tai goal -> quay den huong cuoi
    # 'ARRIVED'       : da xong
    nav_phase = 'TURN_TO_PATH'
    last_vx_cmd = 0.0
    last_omega_cmd = 0.0
    # Counter de xac nhan da on dinh tai goal/heading (chong drift gay nhay phase)
    nav_arrived_counter = 0
    nav_heading_arrived_counter = 0
    # PID state cho heading control (D-term de dap dao dong)
    nav_heading_err_prev = 0.0
    nav_heading_integral = 0.0
    nav_pid_first_tick = True
    heading_ekf = HeadingEKF()

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
                navigation_active = app_state.navigation_active
                navigation_target = dict(app_state.navigation_target) if app_state.navigation_target else None
                measured_wz_feedback = float(app_state.stats.get('measured_wz', 0.0) or 0.0)
                robot_theta_deg_feedback = float(app_state.stats.get('robot_theta_deg', 0.0) or 0.0)
                lidar_x = float(app_state.stats.get('lidar_x', 0.0) or 0.0)
                lidar_y = float(app_state.stats.get('lidar_y', 0.0) or 0.0)
                lidar_theta_deg = float(app_state.stats.get('lidar_theta_deg', 0.0) or 0.0)
                lidar_status = app_state.stats.get('lidar_status', 'DISABLED')
                lidar_last_update = float(app_state.stats.get('lidar_last_update', 0.0) or 0.0)
                motor_enabled_snapshot = app_state.motor_enabled
                app_state.new_measurement = False

            base_telm_age = float('inf')
            if serial_mgr and serial_mgr.last_telm_time > 0.0:
                base_telm_age = max(0.0, time.monotonic() - serial_mgr.last_telm_time)
            has_base_heading = (
                serial_mgr is not None
                and base_telm_age <= NAV_BASE_TELEM_STALE_S
                and math.isfinite(robot_theta_deg_feedback)
            )
            has_base_gyro = has_base_heading and math.isfinite(measured_wz_feedback)

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
            bearing_rate = 0.0
            omega_p = 0.0
            omega_i = 0.0
            omega_d = 0.0
            vy_p = 0.0
            vy_d = 0.0
            align_gain_scale = align_tuner.gain_scale
            navigation_status = 'IDLE'
            navigation_reason = ''
            navigation_distance = 0.0
            navigation_bearing_error_deg = 0.0
            effective_theta_deg = lidar_theta_deg
            heading_source = 'lidar'
            send_stop_now = False
            nav_control_active = bool(navigation_active and navigation_target)

            if nav_control_active != last_navigation_active:
                ctrl_omega_turn.reset()
                ctrl_vy_strafe.reset()
                ctrl_omega_hold.reset()
                pid_vx.reset()
                align_tuner.reset()
                vx_limiter.reset(0.0)
                vy_limiter.reset(0.0)
                omega_limiter.reset(0.0)
                align_mode = 'TURN'
                lost_since = None
                # Reset phase machine khi bat dau navigation moi
                nav_phase = 'TURN_TO_PATH'
                last_vx_cmd = 0.0
                last_omega_cmd = 0.0
                nav_arrived_counter = 0
                nav_heading_arrived_counter = 0
                # Reset PID state
                nav_heading_err_prev = 0.0
                nav_heading_integral = 0.0
                nav_pid_first_tick = True
                if nav_control_active:
                    heading_ekf.reset(
                        math.radians(lidar_theta_deg),
                        math.radians(robot_theta_deg_feedback) if has_base_heading else None,
                    )
                else:
                    heading_ekf.clear()
                last_navigation_active = nav_control_active

            if nav_control_active:
                now_nav = time.monotonic()
                lidar_age = now_nav - lidar_last_update
                navigation_status = 'MOVING'
                if lidar_status != 'TRACKING' or lidar_age > NAV_LIDAR_STALE_S:
                    navigation_status = 'WAIT_LIDAR'
                    navigation_reason = 'LiDAR pose is stale'
                    vx_target = 0.0
                    vy_target = 0.0
                    omega_target = 0.0
                elif not motor_enabled_snapshot:
                    navigation_status = 'WAIT_MOTOR'
                    navigation_reason = 'Motor is stopped'
                    vx_target = 0.0
                    vy_target = 0.0
                    omega_target = 0.0
                else:
                    # ===== PHASE MACHINE: TURN_TO_PATH -> DRIVE -> TURN_TO_GOAL =====
                    target_x = float(navigation_target.get('x', 0.0))
                    target_y = float(navigation_target.get('y', 0.0))
                    # goal_theta CO THE = None (giu huong cuoi tu DRIVE) hoac = float (rad)
                    goal_theta_raw = navigation_target.get('theta', None)
                    has_goal_heading = goal_theta_raw is not None
                    goal_theta = float(goal_theta_raw) if has_goal_heading else 0.0

                    dx_world = target_x - lidar_x
                    dy_world = target_y - lidar_y
                    navigation_distance = math.hypot(dx_world, dy_world)

                    # ===== HEADING FUSION: EKF IMU + LiDAR =====
                    lidar_theta_rad = math.radians(lidar_theta_deg)
                    heading_source = 'lidar'
                    if NAV_USE_IMU_HEADING:
                        if not heading_ekf.initialized:
                            heading_ekf.reset(
                                lidar_theta_rad,
                                math.radians(robot_theta_deg_feedback) if has_base_heading else None,
                            )

                        heading_ekf.predict(
                            measured_wz_feedback if has_base_gyro else 0.0,
                            dt,
                        )

                        if has_base_heading:
                            heading_ekf.update_imu_yaw(math.radians(robot_theta_deg_feedback))

                        lidar_r_deg = (
                            NAV_GOAL_HEADING_EKF_LIDAR_R_DEG
                            if nav_phase == 'TURN_TO_GOAL'
                            else NAV_HEADING_EKF_LIDAR_R_DEG
                        )
                        heading_ekf.update_lidar(lidar_theta_rad, lidar_r_deg)

                        theta = heading_ekf.theta_world_rad
                        effective_theta_deg = math.degrees(theta)
                        heading_source = 'ekf_imu_lidar' if has_base_heading else 'ekf_lidar_only'
                    else:
                        theta = lidar_theta_rad
                        effective_theta_deg = lidar_theta_deg

                    # Heading dat duoc tu pose hien tai den goal (rad, [-pi, pi])
                    desired_heading = math.atan2(dy_world, dx_world)
                    heading_error = wrap_angle_rad(desired_heading - theta)
                    navigation_bearing_error_deg = math.degrees(heading_error)
                    abs_heading_err_deg = abs(navigation_bearing_error_deg)

                    # ----- Phase transitions (CO HYSTERESIS + STABLE COUNTER) -----
                    if nav_phase == 'TURN_TO_PATH':
                        # Quay tai cho cho den khi heading khop
                        if navigation_distance <= NAV_GOAL_TOLERANCE_M:
                            # Da o tai goal -> chuyen TURN_TO_GOAL hoac ARRIVED
                            nav_phase = 'TURN_TO_GOAL' if has_goal_heading else 'ARRIVED'
                            nav_arrived_counter = 0
                            nav_heading_arrived_counter = 0
                            nav_heading_integral = 0.0
                            nav_pid_first_tick = True
                            print(f'[Nav] phase: TURN_TO_PATH -> '
                                  f'{"TURN_TO_GOAL" if has_goal_heading else "ARRIVED"} '
                                  f'(dist={navigation_distance:.2f}m)')
                        elif abs_heading_err_deg <= NAV_PATH_HEADING_TOL_DEG:
                            nav_phase = 'DRIVE'
                            print(f'[Nav] phase: TURN_TO_PATH -> DRIVE '
                                  f'(heading_err={navigation_bearing_error_deg:+.1f}deg, '
                                  f'dist={navigation_distance:.2f}m)')

                    elif nav_phase == 'DRIVE':
                        # Neu lech heading qua nhieu (vd robot bi xo) -> ve TURN_TO_PATH
                        if navigation_distance <= NAV_GOAL_TOLERANCE_M:
                            nav_phase = 'TURN_TO_GOAL' if has_goal_heading else 'ARRIVED'
                            nav_arrived_counter = 0
                            nav_heading_arrived_counter = 0
                            nav_heading_integral = 0.0
                            nav_pid_first_tick = True
                            print(f'[Nav] phase: DRIVE -> '
                                  f'{"TURN_TO_GOAL" if has_goal_heading else "ARRIVED"} '
                                  f'(dist={navigation_distance:.2f}m)')
                        elif abs_heading_err_deg > NAV_PATH_HEADING_HYST_DEG:
                            nav_phase = 'TURN_TO_PATH'
                            print(f'[Nav] phase: DRIVE -> TURN_TO_PATH '
                                  f'(heading_err={navigation_bearing_error_deg:+.1f}deg)')

                    elif nav_phase == 'TURN_TO_GOAL':
                        # ===== CHI QUAN TAM GOC, KHONG QUAN TAM VI TRI DRIFT =====
                        # Robot chi quay tai cho (vx=0, vy=0) -> vi tri thuc te khong di chuyen.
                        # Neu pose drift do ICP jitter, dung KE QUAY VE TURN_TO_PATH (gay loop)
                        # ma chap nhan vi tri hien tai. Da den day -> hoan thanh nhiem vu.
                        heading_error = wrap_angle_rad(goal_theta - theta)
                        navigation_bearing_error_deg = math.degrees(heading_error)
                        abs_heading_err_deg = abs(navigation_bearing_error_deg)
                        if abs_heading_err_deg <= NAV_HEADING_TOLERANCE_DEG:
                            # Stable counter: phai on dinh trong N tick
                            nav_heading_arrived_counter += 1
                        elif (nav_heading_arrived_counter > 0 and
                              abs_heading_err_deg <= NAV_HEADING_EXIT_DEG):
                            # Da vao vung goal-heading roi thi giu hysteresis rong hon
                            # de ICP jitter khong kich lai lenh quay tai cho.
                            nav_heading_arrived_counter += 1
                        else:
                            # Lech ra ngoai tolerance -> reset counter
                            nav_heading_arrived_counter = 0

                        if nav_heading_arrived_counter >= NAV_ARRIVED_HOLD_TICKS:
                            nav_phase = 'ARRIVED'
                            print(f'[Nav] phase: TURN_TO_GOAL -> ARRIVED '
                                  f'(stable for {nav_heading_arrived_counter} ticks, '
                                  f'final_heading_err={navigation_bearing_error_deg:+.1f}deg)')

                    # ----- Compute control commands theo phase -----
                    if nav_phase == 'ARRIVED':
                        navigation_status = 'ARRIVED'
                        navigation_reason = f'Reached {navigation_target.get("label", "target")}'
                        vx_target = 0.0
                        vy_target = 0.0
                        omega_target = 0.0
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

                    elif nav_phase == 'TURN_TO_PATH' or nav_phase == 'TURN_TO_GOAL':
                        # KHONG di chuyen tinh tien khi quay -> chong canh tranh
                        navigation_status = nav_phase
                        vx_target = 0.0
                        vy_target = 0.0

                        # Chon tolerance theo phase:
                        # - TURN_TO_PATH: muc dich la vao DRIVE -> dung NAV_PATH_HEADING_TOL_DEG (12 deg)
                        # - TURN_TO_GOAL: muc dich la dung yen tai goal -> dung NAV_HEADING_TOLERANCE_DEG (8 deg)
                        if nav_phase == 'TURN_TO_PATH':
                            phase_tol_deg = NAV_PATH_HEADING_TOL_DEG  # 12 deg
                            phase_kp = NAV_HEADING_KP
                            phase_ki = NAV_HEADING_KI
                            phase_kd = NAV_HEADING_KD
                            phase_omega_min = NAV_OMEGA_MIN_MOVE
                            phase_omega_max = min(NAV_OMEGA_MAX, OMEGA_MAX)
                            use_integral = True
                        else:
                            # Sau khi da vao vung heading dung, giu hysteresis rong hon
                            # de khong quay qua quay lai vi heading rung nhe tai goal.
                            if nav_heading_arrived_counter > 0:
                                phase_tol_deg = NAV_HEADING_EXIT_DEG
                            else:
                                phase_tol_deg = NAV_HEADING_TOLERANCE_DEG
                            phase_kp = NAV_HEADING_KP * NAV_GOAL_TURN_KP_SCALE
                            phase_ki = 0.0
                            phase_kd = NAV_HEADING_KD * NAV_GOAL_TURN_KD_SCALE
                            phase_omega_min = NAV_GOAL_TURN_MIN_MOVE
                            phase_omega_max = min(
                                NAV_GOAL_TURN_OMEGA_MAX,
                                NAV_OMEGA_MAX,
                                OMEGA_MAX,
                            )
                            use_integral = False

                        # ===== PID DAY DU CHO HEADING (P + I + D) =====
                        # D-term la KEY de DAP DAO DONG: khi err giam nhanh, D pull omega ve 0
                        # giup robot phanh truoc khi over-shoot.
                        dt_pid = max(dt, 1e-3)
                        if nav_pid_first_tick:
                            # Lan dau: D = 0 (khong co prev)
                            d_err = 0.0
                            nav_pid_first_tick = False
                        else:
                            # Sai so heading la bien goc, can wrap delta de tranh
                            # derivative kick khi error nhay qua bien -pi/pi.
                            d_err = wrap_angle_rad(
                                heading_error - nav_heading_err_prev
                            ) / dt_pid
                        nav_heading_err_prev = heading_error

                        # I-term: tich luy sai so (chi khi ngoai tolerance)
                        if use_integral and abs_heading_err_deg > phase_tol_deg:
                            nav_heading_integral += heading_error * dt_pid
                            # Anti-windup: clamp integral
                            i_max = NAV_HEADING_I_LIMIT / max(NAV_HEADING_KI, 1e-6)
                            nav_heading_integral = float(np.clip(
                                nav_heading_integral, -i_max, i_max))
                        else:
                            # Trong tolerance hoac khi dang TURN_TO_GOAL:
                            # xa het I-term cu de tranh no day robot quay lo.
                            nav_heading_integral *= 0.6

                        # PID output = Kp*e + Ki*integral + Kd*de/dt.
                        # Khi err giam nhanh, d_err < 0 nen D-term se keo omega ve 0.
                        omega_p = phase_kp * heading_error
                        omega_i = phase_ki * nav_heading_integral
                        omega_d = phase_kd * d_err
                        omega_raw = omega_p + omega_i + omega_d

                        # Trong tolerance: CHU DONG GIM omega = 0 de robot dung han
                        # (khong cho I-term hoac jitter day omega lo nho)
                        if abs_heading_err_deg < phase_tol_deg:
                            omega_raw = 0.0
                        else:
                            # Min speed de thang ma sat (chi khi ngoai tolerance)
                            if abs(omega_raw) < phase_omega_min:
                                omega_raw = phase_omega_min * (
                                    1.0 if heading_error > 0 else -1.0)

                        omega_target = float(np.clip(
                            omega_raw,
                            -phase_omega_max,
                            phase_omega_max,
                        ))

                    elif nav_phase == 'DRIVE':
                        # Tien thang. CHI dung vx + omega (KHONG dung vy de tranh canh tranh).
                        navigation_status = 'DRIVE'

                        # Vx co decel zone va min speed
                        # P control * Kp, gioi han toi NAV_VX_MAX
                        vx_raw = NAV_TRANSLATION_KP * navigation_distance

                        # Decel khi gan goal: ramp tu NAV_VX_MAX xuong 0 trong vung NAV_DECEL_DIST_M
                        if navigation_distance < NAV_DECEL_DIST_M:
                            decel_factor = navigation_distance / NAV_DECEL_DIST_M
                            vx_raw = min(vx_raw, NAV_VX_MAX * decel_factor)
                            # Trong vung approach cuoi, ep cham hon
                            if navigation_distance < NAV_FINAL_APPROACH_M:
                                vx_raw = min(vx_raw, NAV_VX_MAX * 0.4)

                        # Min speed: tranh tinh do ma sat
                        if vx_raw < NAV_VX_MIN_MOVE and navigation_distance > NAV_GOAL_TOLERANCE_M:
                            vx_raw = NAV_VX_MIN_MOVE

                        vx_target = float(np.clip(vx_raw, 0.0, min(NAV_VX_MAX, VX_MAX)))
                        vy_target = 0.0

                        # Omega trong DRIVE: PID nhe de giu line + dap dao dong
                        dt_pid = max(dt, 1e-3)
                        if nav_pid_first_tick:
                            d_err = 0.0
                            nav_pid_first_tick = False
                        else:
                            d_err = wrap_angle_rad(
                                heading_error - nav_heading_err_prev
                            ) / dt_pid
                        nav_heading_err_prev = heading_error
                        # Drive khong dung I-term (dung leak) vi robot dang chay
                        nav_heading_integral *= 0.85

                        omega_drive = (NAV_HEADING_KP * heading_error * 0.7 +
                                       NAV_HEADING_KD * d_err * 0.6)
                        omega_target = float(np.clip(
                            omega_drive,
                            -min(NAV_OMEGA_MAX * 0.7, OMEGA_MAX),
                            min(NAV_OMEGA_MAX * 0.7, OMEGA_MAX),
                        ))

                    else:
                        # Unknown phase - safety stop
                        vx_target = 0.0
                        vy_target = 0.0
                        omega_target = 0.0

                    # ----- Rate limiting -----
                    # Han che thay doi vx, omega qua nhanh -> mượt hon
                    dt_safe = max(dt, 1e-3)
                    max_vx_change = NAV_VX_RATE_LIMIT * dt_safe
                    if vx_target > last_vx_cmd + max_vx_change:
                        vx_target = last_vx_cmd + max_vx_change
                    elif vx_target < last_vx_cmd - max_vx_change:
                        vx_target = last_vx_cmd - max_vx_change
                    last_vx_cmd = vx_target

                    max_omega_change = NAV_OMEGA_RATE_LIMIT * dt_safe
                    if omega_target > last_omega_cmd + max_omega_change:
                        omega_target = last_omega_cmd + max_omega_change
                    elif omega_target < last_omega_cmd - max_omega_change:
                        omega_target = last_omega_cmd - max_omega_change
                    last_omega_cmd = omega_target

                    # Update phase trong stats (cho UI)
                    with app_state.lock:
                        app_state.stats['navigation_phase'] = nav_phase
                err_dist = navigation_distance

            if not nav_control_active and ball_detected:
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

                if align_mode == 'TURN':
                    vy_target = 0.0
                    omega_target = ctrl_omega_turn.compute(err_x_db, bearing_rate, scale=align_gain_scale)
                    omega_p = ctrl_omega_turn.last_p
                    omega_d = ctrl_omega_turn.last_d
                elif align_mode == 'STRAFE':
                    vy_target = ctrl_vy_strafe.compute(err_x_db, bearing_rate, scale=align_gain_scale)
                    omega_target = ctrl_omega_hold.compute(err_x_db, bearing_rate, scale=1.0)
                    vy_p = ctrl_vy_strafe.last_p
                    vy_d = ctrl_vy_strafe.last_d
                    omega_p = ctrl_omega_hold.last_p
                    omega_d = ctrl_omega_hold.last_d
                else:
                    vy_target = 0.0
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
                    ctrl_omega_turn.reset()
                    ctrl_vy_strafe.reset()
                    ctrl_omega_hold.reset()
                    pid_vx.reset()
                    align_tuner.reset()
                    align_mode = 'TURN'
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

            if send_stop_now and serial_mgr:
                serial_mgr.send_stop()
                serial_mgr.read_feedback()
            elif serial_mgr and app_state.motor_enabled:
                serial_mgr.send_velocity(vx, vy, omega)
                serial_mgr.read_feedback()
            elif serial_mgr and not app_state.motor_enabled:
                ctrl_omega_turn.reset()
                ctrl_vy_strafe.reset()
                ctrl_omega_hold.reset()
                pid_vx.reset()
                align_tuner.reset()
                align_mode = 'TURN'
                vx_limiter.reset(0.0)
                vy_limiter.reset(0.0)
                omega_limiter.reset(0.0)
                vx = 0.0
                vy = 0.0
                omega = 0.0
                serial_mgr.read_feedback()

            with app_state.lock:
                app_state.stats['navigation_active'] = nav_control_active and navigation_status != 'ARRIVED'
                app_state.stats['navigation_target'] = navigation_target
                app_state.stats['navigation_status'] = navigation_status
                app_state.stats['navigation_reason'] = navigation_reason
                app_state.stats['navigation_distance'] = navigation_distance
                app_state.stats['navigation_bearing_error_deg'] = navigation_bearing_error_deg
                app_state.stats['heading_effective_deg'] = effective_theta_deg
                app_state.stats['heading_source'] = heading_source
                app_state.stats['heading_bias_deg'] = (
                    math.degrees(heading_ekf.bias_rad) if heading_ekf.initialized else 0.0
                )
                app_state.stats['heading_std_deg'] = (
                    heading_ekf.heading_std_deg if heading_ekf.initialized else 0.0
                )

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
                display_status = navigation_status if nav_control_active else ('TRACKING' if ball_detected else 'SEARCHING')
                info = {
                    'status': display_status,
                    'ball_detected': ball_detected,
                    'vx': vx,
                    'vy': vy,
                    'omega': omega,
                    'vx_target': vx_target,
                    'vy_target': vy_target,
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
                    'heading_source': heading_source,
                    'heading_effective_deg': float(app_state.stats.get('heading_effective_deg', 0.0) or 0.0),
                    'heading_bias_deg': float(app_state.stats.get('heading_bias_deg', 0.0) or 0.0),
                    'heading_std_deg': float(app_state.stats.get('heading_std_deg', 0.0) or 0.0),
                    'align_mode': align_mode,
                    'align_gain_scale': align_gain_scale,
                    'align_overshoots': align_tuner.overshoots,
                    'bearing_rate': bearing_rate,
                    'omega_p': omega_p,
                    'omega_i': omega_i,
                    'omega_d': omega_d,
                    'vy_p': vy_p,
                    'vy_d': vy_d,
                    'vx_p': pid_vx.last_p,
                    'vx_i': pid_vx.last_i,
                    'vx_d': pid_vx.last_d,
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
        if lidar_mgr:
            lidar_mgr.stop()
        if serial_mgr:
            serial_mgr.send_stop()
            serial_mgr.close()
        cap.release()
        print('[Exit] Stopped.')


if __name__ == '__main__':
    main()