"""
calibration.py - Robot self-calibration module

Calibrate 3 nhom tham so theo thu tu:
    1. Motor Kv/Ks    (feedforward)
    2. Wheel diameter (encoder -> meters)
    3. Rotation radius (xoay tai cho)

Su dung:
    cal = Calibrator(serial_mgr, app_state, config_path='robot_config.json')
    await cal.run_full()                    # Chay tat ca
    await cal.calibrate_motors()            # Chi calibrate motor
    await cal.calibrate_wheel_diameter()    # Chi calibrate wheel
    await cal.calibrate_rotation_radius()   # Chi calibrate rotation
    cal.save_config()                       # Luu vao file
    cal.load_config()                       # Load tu file

Yeu cau:
    - serial_mgr phai co cac method:
        send_velocity(vx, vy, omega)
        send_raw_pwm(m0, m1, m2, m3)        # MOI - bypass PID
        query_raw_rpm() -> [r0, r1, r2, r3] # MOI - lay RPM tuc thoi
        measured_vx, measured_vy, measured_wz   # van toc do tu encoder
    - app_state.stats co:
        'lidar_x', 'lidar_y', 'lidar_theta_deg'    # tu SLAM
        'imu_yaw_deg' hoac tuong duong
"""

import json
import math
import os
import time
import asyncio
import logging
from dataclasses import dataclass, asdict, field
from typing import Optional, Callable

log = logging.getLogger('calibration')


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class WheelMotorParams:
    """Tham so feedforward cho 1 banh xe, 1 chieu quay."""
    kv: float = 0.0   # PWM per RPM
    ks: float = 0.0   # PWM offset (static friction)
    rmse: float = 0.0 # do khop cua hoi quy (RPM)


@dataclass
class RobotConfig:
    """Toan bo tham so calibration cua robot."""
    # Wheel/chassis (đơn vị met)
    wheel_diameter_m: float = 0.097
    rotation_radius_m: float = 0.165
    rpm_max: float = 729.0

    # Motor feedforward - 4 banh × 2 chieu (forward, reverse)
    # Index: motor_kv[wheel_idx][direction] voi direction in {'fwd', 'rev'}
    motor_params: dict = field(default_factory=lambda: {
        'wheel_0': {'fwd': asdict(WheelMotorParams()),
                    'rev': asdict(WheelMotorParams())},
        'wheel_1': {'fwd': asdict(WheelMotorParams()),
                    'rev': asdict(WheelMotorParams())},
        'wheel_2': {'fwd': asdict(WheelMotorParams()),
                    'rev': asdict(WheelMotorParams())},
        'wheel_3': {'fwd': asdict(WheelMotorParams()),
                    'rev': asdict(WheelMotorParams())},
    })

    # Metadata
    calibration_time: str = ''
    notes: str = ''

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'RobotConfig':
        cfg = cls()
        cfg.wheel_diameter_m = float(d.get('wheel_diameter_m', cfg.wheel_diameter_m))
        cfg.rotation_radius_m = float(d.get('rotation_radius_m', cfg.rotation_radius_m))
        cfg.rpm_max = float(d.get('rpm_max', cfg.rpm_max))
        if 'motor_params' in d:
            cfg.motor_params = d['motor_params']
        cfg.calibration_time = d.get('calibration_time', '')
        cfg.notes = d.get('notes', '')
        return cfg

    def average_kv_ks(self) -> tuple:
        """Tinh Kv/Ks trung binh tu 4 banh × 2 chieu (de fallback vao Arduino)."""
        kvs, kss = [], []
        for wheel in self.motor_params.values():
            for direction in wheel.values():
                if direction.get('kv', 0) > 0:
                    kvs.append(direction['kv'])
                    kss.append(direction['ks'])
        if not kvs:
            return (0.0, 0.0)
        return (sum(kvs) / len(kvs), sum(kss) / len(kss))


# ============================================================
# LINEAR REGRESSION (no scipy/numpy dependency for portability)
# ============================================================

def linear_regression(xs: list, ys: list) -> tuple:
    """
    Fit y = a + b*x bang OLS.
    Return: (intercept, slope, rmse)
    """
    n = len(xs)
    if n < 2:
        return (0.0, 0.0, float('inf'))
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return (0.0, 0.0, float('inf'))
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    # RMSE
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    rmse = math.sqrt(sse / n)
    return (a, b, rmse)


# ============================================================
# CALIBRATOR
# ============================================================

class Calibrator:
    """
    Quan ly toan bo qua trinh calibration.
    Co the chay tu CLI hoac tu web UI.
    """

    DEFAULT_CONFIG_PATH = 'robot_config.json'
    LIDAR_STALE_S = 1.0

    def __init__(self,
                 serial_mgr,
                 app_state,
                 config_path: Optional[str] = None,
                 progress_callback: Optional[Callable] = None,
                 log_callback: Optional[Callable] = None):
        self.serial_mgr = serial_mgr
        self.app_state = app_state
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = RobotConfig()
        self._progress_cb = progress_callback or (lambda p, m: None)
        self._log_cb = log_callback or (lambda msg: log.info(msg))
        self._abort = False

    # ------------- LOGGING & PROGRESS -------------

    def _emit(self, msg: str):
        self._log_cb(msg)
        log.info(msg)

    def _progress(self, percent: float, msg: str):
        self._progress_cb(percent, msg)

    def abort(self):
        """User huy calibration giua chung."""
        self._abort = True
        try:
            self.serial_mgr.send_velocity(0.0, 0.0, 0.0)
        except Exception:
            pass

    def _check_abort(self):
        if self._abort:
            raise CalibrationAborted('User aborted')

    def _read_serial_feedback(self):
        """Doc telemetry moi nhat neu calibrator dang la reader chinh."""
        reader = getattr(self.serial_mgr, 'read_feedback', None)
        if callable(reader):
            try:
                reader()
            except Exception:
                pass

    def _get_encoder_pose(self) -> Optional[tuple]:
        """Lay odometry encoder tu Arduino sau reset, neu telemetry con moi."""
        if self.serial_mgr is None:
            return None
        last_telm = float(getattr(self.serial_mgr, 'last_telm_time', 0.0) or 0.0)
        if last_telm > 0.0 and time.monotonic() - last_telm > 1.0:
            return None
        try:
            x = float(getattr(self.serial_mgr, 'robot_x'))
            y = float(getattr(self.serial_mgr, 'robot_y'))
            theta_deg = float(getattr(self.serial_mgr, 'robot_theta_deg', 0.0) or 0.0)
        except (TypeError, ValueError, AttributeError):
            return None
        return (x, y, math.radians(theta_deg))

    async def _read_feedback_for(self, seconds: float, interval_s: float = 0.03):
        end = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < end:
            self._check_abort()
            self._read_serial_feedback()
            await asyncio.sleep(interval_s)

    # ------------- SAFETY -------------

    async def _safety_stop(self, duration_s: float = 1.5):
        """Dung robot va doi on dinh."""
        try:
            self.serial_mgr.send_velocity(0.0, 0.0, 0.0)
        except Exception as e:
            self._emit(f'[WARN] send_velocity stop failed: {e}')
        await self._read_feedback_for(duration_s)

    async def _wait_settle(self, seconds: float):
        """Cho robot on dinh, kiem tra abort."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._check_abort()
            self._read_serial_feedback()
            await asyncio.sleep(0.05)

    # ------------- GET POSE FROM SLAM -------------

    def _get_pose(self, prefer_icp: bool = False) -> tuple:
        """Lay (x, y, theta_rad) tu SLAM.

        Khi wheel calibration chay, uu tien pose ICP thuan neu web app publish
        duoc. Pose fused co encoder trong vong lap se lam phep do bi tu tham
        chieu va co the day wheel_diameter di sai qua moi lan chay.
        """
        now = time.monotonic()
        with self.app_state.lock:
            if prefer_icp:
                icp_last = float(self.app_state.stats.get('lidar_icp_last_update', 0.0) or 0.0)
                if icp_last > 0.0 and now - icp_last <= self.LIDAR_STALE_S:
                    x = float(self.app_state.stats.get('lidar_icp_x', 0.0) or 0.0)
                    y = float(self.app_state.stats.get('lidar_icp_y', 0.0) or 0.0)
                    theta_deg = float(self.app_state.stats.get('lidar_icp_theta_deg', 0.0) or 0.0)
                    return (x, y, math.radians(theta_deg))
            x = float(self.app_state.stats.get('lidar_x', 0.0) or 0.0)
            y = float(self.app_state.stats.get('lidar_y', 0.0) or 0.0)
            theta_deg = float(self.app_state.stats.get('lidar_theta_deg', 0.0) or 0.0)
        return (x, y, math.radians(theta_deg))

    def _get_lidar_tracking(self) -> tuple:
        """Lay trang thai TRACKING va do tuoi cua pose LiDAR."""
        if self.app_state is None:
            return ('DISABLED', float('inf'))
        with self.app_state.lock:
            status = str(self.app_state.stats.get('lidar_status', 'DISABLED') or 'DISABLED')
            last_update = float(self.app_state.stats.get('lidar_last_update', 0.0) or 0.0)
        age = time.monotonic() - last_update if last_update > 0.0 else float('inf')
        return (status, age)

    def _require_fresh_lidar_tracking(self):
        status, age = self._get_lidar_tracking()
        if status != 'TRACKING' or age > self.LIDAR_STALE_S:
            raise RuntimeError(
                f'Wheel calibration requires LiDAR TRACKING (status={status}, age={age:.2f}s)')

    def _get_imu_yaw(self) -> Optional[float]:
        """
        Lay IMU yaw absolute (rad).
        Tra ve None neu IMU khong san sang.
        Code can adapt theo cach app_state luu IMU yaw.
        """
        with self.app_state.lock:
            # Tu trong code goc, IMU yaw duoc lay tu telemetry Arduino
            # va luu vao app_state. Field name co the can chinh.
            theta_deg = self.app_state.stats.get('imu_yaw_deg', None)
            if theta_deg is None:
                # Fallback: dung lidar_theta_deg
                theta_deg = self.app_state.stats.get('lidar_theta_deg', None)
            if theta_deg is None:
                theta_deg = getattr(self.serial_mgr, 'robot_theta_deg', None)
            if theta_deg is None:
                return None
            return math.radians(float(theta_deg))

    # ============================================================
    # 1. MOTOR Kv/Ks CALIBRATION
    # ============================================================

    async def calibrate_motors(self,
                               pwm_levels: list = None,
                               settle_s: float = 1.5,
                               sample_s: float = 1.0,
                               num_samples: int = 5) -> dict:
        """
        Calibrate Kv/Ks cho 4 banh × 2 chieu.

        QUAN TRONG: Robot phai duoc nhac len, banh xe khong cham dat,
                    de tranh ma sat san anh huong ket qua.

        Phuong phap:
          - Cap PWM tu thap den cao (60, 80, ..., 220)
          - Doi steady-state, do RPM
          - Fit PWM = Ks + Kv * RPM
        """
        self._emit('========== MOTOR Kv/Ks CALIBRATION ==========')
        self._emit('CHU Y: Robot phai duoc NHAC LEN, banh xe KHONG cham dat!')
        self._emit('Hieu chinh nay duoc thuc hien voi PID tat (raw PWM).')

        if pwm_levels is None:
            pwm_levels = [60, 80, 100, 120, 140, 160, 180, 200, 220]

        wheel_count = 4
        directions = [('fwd', +1), ('rev', -1)]
        total_steps = wheel_count * len(directions) * len(pwm_levels)
        step = 0
        max_observed_rpm = 0.0

        # Reset robot truoc khi test
        await self._safety_stop(1.0)

        for wheel_idx in range(wheel_count):
            for dir_name, sign in directions:
                self._check_abort()
                self._emit(f'\n--- Wheel {wheel_idx}, {dir_name.upper()} ---')

                pwms_recorded = []
                rpms_recorded = []

                for pwm in pwm_levels:
                    self._check_abort()
                    target_pwm = sign * pwm

                    # Apply raw PWM chi cho banh nay, cac banh khac = 0
                    pwm_array = [0, 0, 0, 0]
                    pwm_array[wheel_idx] = target_pwm
                    self.serial_mgr.send_raw_pwm(*pwm_array)

                    # Doi steady-state
                    await self._wait_settle(settle_s)

                    # Lay nhieu mau RPM
                    rpm_samples = []
                    for _ in range(num_samples):
                        self._check_abort()
                        rpms = self.serial_mgr.query_raw_rpm()
                        if rpms is not None and len(rpms) >= 4:
                            rpm_samples.append(rpms[wheel_idx])
                        await asyncio.sleep(sample_s / num_samples)

                    if not rpm_samples:
                        self._emit(f'  [WARN] PWM={pwm}: khong nhan duoc RPM')
                        step += 1
                        self._progress(step / total_steps,
                                       f'wheel{wheel_idx} {dir_name} PWM={pwm}: NO DATA')
                        continue

                    avg_rpm = sum(rpm_samples) / len(rpm_samples)
                    abs_rpm = abs(avg_rpm)

                    # Chi nhan diem co RPM > nguong nho (banh thuc su quay)
                    if abs_rpm < 5.0:
                        self._emit(f'  PWM={pwm}: RPM={avg_rpm:.1f} (qua nho, bo qua)')
                    else:
                        pwms_recorded.append(pwm)  # luu PWM duong
                        rpms_recorded.append(abs_rpm)
                        max_observed_rpm = max(max_observed_rpm, abs_rpm)
                        self._emit(f'  PWM={pwm}: RPM={avg_rpm:.1f}')

                    step += 1
                    self._progress(step / total_steps,
                                   f'wheel{wheel_idx} {dir_name} PWM={pwm} RPM={avg_rpm:.0f}')

                # Stop wheel
                self.serial_mgr.send_raw_pwm(0, 0, 0, 0)
                await asyncio.sleep(0.5)

                # Fit PWM = Ks + Kv * RPM
                if len(pwms_recorded) >= 3:
                    ks, kv, rmse = linear_regression(rpms_recorded, pwms_recorded)
                    self._emit(f'  -> Kv={kv:.4f} PWM/RPM, Ks={ks:.2f} PWM, RMSE={rmse:.2f}')
                    self.config.motor_params[f'wheel_{wheel_idx}'][dir_name] = {
                        'kv': float(kv),
                        'ks': float(ks),
                        'rmse': float(rmse),
                    }
                else:
                    self._emit(f'  [ERR] Khong du diem de fit (chi co {len(pwms_recorded)})')

        await self._safety_stop(0.5)
        if max_observed_rpm > 10.0:
            estimated_rpm_max = max_observed_rpm * (255.0 / max(pwm_levels))
            self.config.rpm_max = float(estimated_rpm_max)
            self._emit(f'Estimated RPM_MAX from motor calibration: {estimated_rpm_max:.1f}')
        self._emit('\n========== MOTOR CALIBRATION DONE ==========')
        return self.config.motor_params

    # ============================================================
    # 2. WHEEL DIAMETER CALIBRATION
    # ============================================================

    async def calibrate_wheel_diameter(self,
                                       target_distance_m: float = 1.0,
                                       speed_mps: float = 0.15,
                                       manual_measured_m: Optional[float] = None,
                                       current_diameter_m: Optional[float] = None) -> dict:
        """
        Cho robot di thang trong khoang `target_distance_m`.
        So sanh:
          - Khoang cach robot tinh tu encoder (theo wheel_diameter cu)
          - Khoang cach do tu LiDAR/SLAM (auto)
          - Khoang cach do bang thuoc (manual, neu nhap)

        Wheel diameter moi = wheel_diameter_cu × (real_distance / encoder_distance)
        """
        self._emit('========== WHEEL DIAMETER CALIBRATION ==========')
        self._emit(f'Robot se di thang ~{target_distance_m}m voi v={speed_mps} m/s')
        self._emit('Robot phai dat tren MAT SAN PHANG, khong vuong vat can.')

        if current_diameter_m is None:
            current_diameter_m = self.config.wheel_diameter_m
        send_wheel = getattr(self.serial_mgr, 'send_wheel_diameter', None)
        if callable(send_wheel):
            send_wheel(float(current_diameter_m))
            await self._read_feedback_for(0.15)

        # Lay pose ban dau tu LiDAR
        await self._safety_stop(1.5)  # cho LiDAR/SLAM on dinh
        self._require_fresh_lidar_tracking()
        x0, y0, _ = self._get_pose(prefer_icp=True)
        self._emit(f'Pose ban dau (LiDAR): ({x0:.3f}, {y0:.3f})')

        # Reset encoder odometry (gui lenh R)
        try:
            self.serial_mgr.reset_odometry()
        except AttributeError:
            self._emit('[WARN] serial_mgr khong co reset_odometry, bo qua')

        await self._read_feedback_for(0.35)
        enc0 = self._get_encoder_pose()
        if enc0 is None:
            self._emit('[WARN] Khong doc duoc odometry Arduino sau reset; '
                       'fallback sang tich phan measured_vx')
        else:
            self._emit(f'Encoder odom reset: ({enc0[0]:+.3f}, {enc0[1]:+.3f})')

        # Dung theo quang duong do duoc, timeout chi la hang rao an toan.
        # Cach cu chay theo thoi gian uoc tinh * 1.3 se lam robot di qua xa,
        # nhat la khi wheel_diameter hien tai dang sai.
        timeout_s = max((target_distance_m / max(abs(speed_mps), 1e-3)) * 3.0, 8.0)
        timeout_s = min(timeout_s, 20.0)

        self._emit(
            f'Bat dau di toi khi dat ~{target_distance_m:.2f}m '
            f'(timeout {timeout_s:.1f}s)...')

        start_time = time.monotonic()
        encoder_distance = 0.0
        encoder_distance_integrated = 0.0
        encoder_method = 'velocity_integral'
        last_measure_time = start_time
        last_measured_vx = 0.0
        lidar_lost = False
        last_debug_log_time = start_time  # DEBUG: in log moi 1 giay

        # Vong di thang: tich phan vx tu encoder de tinh quang duong.
        # Neu co LiDAR/SLAM, dung theo quang duong thuc do duoc tu pose.
        while True:
            self._check_abort()
            self.serial_mgr.send_velocity(speed_mps, 0.0, 0.0)
            self._read_serial_feedback()

            now = time.monotonic()
            elapsed = now - start_time
            if elapsed >= timeout_s:
                self._emit(f'[WARN] Wheel calibration timeout sau {elapsed:.1f}s')
                break

            dt = now - last_measure_time
            # Tich phan vx: encoder_distance += vx * dt
            vx = float(getattr(self.serial_mgr, 'measured_vx', 0.0) or 0.0)
            # Trung binh giua sample truoc va sau (trapezoidal)
            avg_vx = (vx + last_measured_vx) * 0.5
            encoder_distance_integrated += abs(avg_vx) * dt
            last_measure_time = now
            last_measured_vx = vx

            enc_now = self._get_encoder_pose()
            if enc0 is not None and enc_now is not None:
                encoder_distance = math.hypot(enc_now[0] - enc0[0],
                                              enc_now[1] - enc0[1])
                encoder_method = 'arduino_odometry'
            else:
                encoder_distance = encoder_distance_integrated
                encoder_method = 'velocity_integral'

            lidar_status, lidar_age = self._get_lidar_tracking()
            if lidar_status != 'TRACKING' or lidar_age > self.LIDAR_STALE_S:
                self._emit(
                    f'[ERR] LiDAR tracking lost during wheel calibration '
                    f'(status={lidar_status}, age={lidar_age:.2f}s)')
                lidar_lost = True
                break

            x_now, y_now, _ = self._get_pose(prefer_icp=True)
            lidar_distance_now = math.hypot(x_now - x0, y_now - y0)
            stop_distance = lidar_distance_now

            # Progress
            self._progress(min(stop_distance / max(target_distance_m, 1e-6), 1.0),
                           f'Driving... lidar={lidar_distance_now:.3f}m '
                           f'encoder={encoder_distance:.3f}m vx={vx:.3f}m/s')

            # DEBUG: in log chi tiet moi 1 giay de phat hien van de
            if now - last_debug_log_time >= 1.0:
                self._emit(
                    f'[DBG t={elapsed:5.2f}s] '
                    f'pose=({x_now:+.3f},{y_now:+.3f}) '
                    f'd_lidar={lidar_distance_now:.3f}m '
                    f'd_enc={encoder_distance:.3f}m '
                    f'vx={vx:+.3f}m/s '
                    f'lidar_age={lidar_age:.2f}s')
                last_debug_log_time = now

            if stop_distance >= target_distance_m:
                self._emit(f'[OK] Dat muc tieu {target_distance_m}m sau {elapsed:.2f}s '
                           f'(lidar={lidar_distance_now:.3f}m)')
                break

            await asyncio.sleep(0.05)

        # Stop
        await self._safety_stop(2.0)  # cho robot dung han va SLAM on dinh

        if lidar_lost:
            raise RuntimeError('LiDAR tracking lost during wheel calibration')

        # Lay pose cuoi
        x1, y1, _ = self._get_pose(prefer_icp=True)
        lidar_distance = math.hypot(x1 - x0, y1 - y0)
        enc1 = self._get_encoder_pose()
        if enc0 is not None and enc1 is not None:
            encoder_distance = math.hypot(enc1[0] - enc0[0],
                                          enc1[1] - enc0[1])
            encoder_method = 'arduino_odometry'
        self._emit(f'Pose cuoi (LiDAR): ({x1:.3f}, {y1:.3f})')
        self._emit(f'Khoang cach LiDAR  do duoc: {lidar_distance:.4f} m')
        self._emit(f'Khoang cach encoder bao   : {encoder_distance:.4f} m '
                   f'({encoder_method})')
        if manual_measured_m is not None:
            self._emit(f'Khoang cach thuoc do (manual): {manual_measured_m:.4f} m')

        # Quyet dinh dung khoang cach nao lam reference
        if manual_measured_m is not None and manual_measured_m > 0.05:
            real_distance = manual_measured_m
            method = 'manual_tape'
        elif lidar_distance > 0.05:
            real_distance = lidar_distance
            method = 'lidar_icp' if encoder_method == 'arduino_odometry' else 'lidar_slam'
        else:
            self._emit('[ERR] Khong do duoc khoang cach hop ly!')
            return {'error': 'no_valid_distance',
                    'encoder_distance': encoder_distance,
                    'lidar_distance': lidar_distance}

        if encoder_distance < 0.05:
            self._emit('[ERR] Encoder bao distance qua nho - co the encoder loi')
            return {'error': 'encoder_too_small',
                    'encoder_distance': encoder_distance,
                    'real_distance': real_distance}

        # Tinh wheel diameter moi
        ratio = real_distance / encoder_distance
        new_diameter = current_diameter_m * ratio
        self._emit(f'Ratio (real/encoder) = {ratio:.4f}')
        self._emit(f'Wheel diameter cu  : {current_diameter_m * 1000:.2f} mm')
        self._emit(f'Wheel diameter moi : {new_diameter * 1000:.2f} mm')
        self._emit(f'Reference method   : {method}')

        # Sanity check: diameter khong duoc thay doi qua 30%
        if abs(ratio - 1.0) > 0.30:
            self._emit('[WARN] Ratio lech > 30% - co the do sai, hay kiem tra lai')

        self.config.wheel_diameter_m = float(new_diameter)
        self._emit('========== WHEEL DIAMETER DONE ==========')

        return {
            'old_diameter_m': current_diameter_m,
            'new_diameter_m': new_diameter,
            'encoder_distance_m': encoder_distance,
            'lidar_distance_m': lidar_distance,
            'manual_distance_m': manual_measured_m,
            'reference_method': method,
            'encoder_method': encoder_method,
            'ratio': ratio,
        }

    # ============================================================
    # 3. ROTATION RADIUS CALIBRATION
    # ============================================================

    async def calibrate_rotation_radius(self,
                                        target_rotations: float = 1.0,
                                        omega_rad_s: float = 0.6,
                                        manual_measured_rad: Optional[float] = None,
                                        current_radius_m: Optional[float] = None) -> dict:
        """
        Cho robot xoay tai cho `target_rotations` vong (mac dinh 1 vong = 2pi rad).
        So sanh:
          - Goc encoder bao (theo rotation_radius cu)
          - Goc IMU do (auto, chinh xac vi BNO085)
          - Goc do tay (manual, neu nhap)

        Rotation radius moi = radius_cu × (encoder_omega / real_omega)
        """
        self._emit('========== ROTATION RADIUS CALIBRATION ==========')
        self._emit(f'Robot se xoay tai cho ~{target_rotations} vong voi omega={omega_rad_s} rad/s')
        self._emit('Hay dam bao co du khong gian xung quanh (>= 1m).')

        if current_radius_m is None:
            current_radius_m = self.config.rotation_radius_m
        send_radius = getattr(self.serial_mgr, 'send_radius', None)
        if callable(send_radius):
            send_radius(float(current_radius_m))
            await self._read_feedback_for(0.15)

        # Doi LiDAR/IMU on dinh
        await self._safety_stop(1.5)

        # Lay yaw ban dau tu IMU (uu tien) hoac LiDAR theta
        yaw0_imu = self._get_imu_yaw()
        _, _, yaw0_lidar = self._get_pose()

        if yaw0_imu is None:
            self._emit('[WARN] Khong doc duoc IMU yaw, dung LiDAR theta')
            yaw0 = yaw0_lidar
            yaw_source = 'lidar'
        else:
            yaw0 = yaw0_imu
            yaw_source = 'imu'

        self._emit(f'Yaw ban dau ({yaw_source}): {math.degrees(yaw0):.2f} deg')

        # Reset encoder odometry
        try:
            self.serial_mgr.reset_odometry()
        except AttributeError:
            pass
        await self._read_feedback_for(0.35)

        # Dung theo goc do duoc, timeout chi la hang rao an toan.
        # Cach cu nhan them 30% thoi gian de "du phong" se lam robot quay hon 1 vong
        # neu he thong bam toc do quay tot.
        target_rad = target_rotations * 2 * math.pi
        timeout_s = max((target_rad / max(abs(omega_rad_s), 1e-3)) * 2.0, 8.0)
        timeout_s = min(timeout_s, 30.0)

        self._emit(
            f'Bat dau xoay toi khi dat ~{math.degrees(target_rad):.1f} deg '
            f'(timeout {timeout_s:.1f}s)...')

        start_time = time.monotonic()
        encoder_yaw_integrated = 0.0
        last_measure_time = start_time
        last_wz = 0.0

        # Theo doi yaw IMU lien tuc de xu ly wrap-around (vuot 360°)
        prev_yaw_unwrapped = yaw0
        total_yaw_change = 0.0  # tich luy goc that, khong wrap

        use_imu_for_stop = yaw0_imu is not None

        while True:
            self._check_abort()
            self.serial_mgr.send_velocity(0.0, 0.0, omega_rad_s)
            self._read_serial_feedback()

            now = time.monotonic()
            elapsed = now - start_time
            if elapsed >= timeout_s:
                self._emit(f'[WARN] Rotation calibration timeout sau {elapsed:.1f}s')
                break

            dt = now - last_measure_time

            # Use wheel-based wz (measured_wz_enc) so the integration is independent
            # of IMU. This is essential: if measured_wz = imu_wz, both integrals
            # would be equal and the calibration ratio would always be ~1.0.
            wz = float(getattr(self.serial_mgr, 'measured_wz_enc', None)
                       or getattr(self.serial_mgr, 'measured_wz', 0.0) or 0.0)
            avg_wz = (wz + last_wz) * 0.5
            encoder_yaw_integrated += avg_wz * dt
            last_measure_time = now
            last_wz = wz

            # IMU integration co xu ly wrap
            cur_yaw_imu = self._get_imu_yaw()
            if cur_yaw_imu is not None:
                d_yaw = cur_yaw_imu - prev_yaw_unwrapped
                # Wrap vao [-pi, pi]
                while d_yaw > math.pi:
                    d_yaw -= 2 * math.pi
                while d_yaw < -math.pi:
                    d_yaw += 2 * math.pi
                total_yaw_change += d_yaw
                prev_yaw_unwrapped = cur_yaw_imu

            stop_rad = abs(total_yaw_change) if use_imu_for_stop else abs(encoder_yaw_integrated)
            self._progress(min(stop_rad / max(target_rad, 1e-6), 1.0),
                           f'Rotating... encoder={math.degrees(encoder_yaw_integrated):.0f}deg '
                           f'IMU={math.degrees(total_yaw_change):.0f}deg')

            if stop_rad >= target_rad:
                break

            await asyncio.sleep(0.05)

        await self._safety_stop(2.0)

        # Lay yaw cuoi (de cross-check)
        yaw1_imu = self._get_imu_yaw()
        _, _, yaw1_lidar = self._get_pose()

        # Encoder bao bao nhieu rad
        encoder_total_rad = abs(encoder_yaw_integrated)
        # IMU bao bao nhieu rad (da tich luy co xu ly wrap)
        imu_total_rad = abs(total_yaw_change)

        self._emit(f'Encoder bao: {math.degrees(encoder_total_rad):.2f} deg '
                   f'({encoder_total_rad / (2*math.pi):.3f} vong)')
        self._emit(f'IMU do    : {math.degrees(imu_total_rad):.2f} deg '
                   f'({imu_total_rad / (2*math.pi):.3f} vong)')
        if manual_measured_rad is not None:
            self._emit(f'Manual    : {math.degrees(manual_measured_rad):.2f} deg')

        # Quyet dinh ground truth
        if manual_measured_rad is not None and abs(manual_measured_rad) > 0.1:
            real_rad = abs(manual_measured_rad)
            method = 'manual'
        elif imu_total_rad > 0.1:
            real_rad = imu_total_rad
            method = 'imu'
        else:
            self._emit('[ERR] Khong do duoc goc xoay hop ly!')
            return {'error': 'no_valid_rotation'}

        if encoder_total_rad < 0.1:
            self._emit('[ERR] Encoder bao goc xoay qua nho!')
            return {'error': 'encoder_rotation_too_small'}

        # Cong thuc:
        #   omega_real = sum(wheel_speeds) / rotation_radius
        # => omega_encoder_with_old_R = omega_real * (R_real / R_old)  (sai khi R_old != R_real)
        # => R_real = R_old * (omega_encoder / omega_real)
        ratio = encoder_total_rad / real_rad
        new_radius = current_radius_m * ratio
        self._emit(f'Ratio (encoder/real) = {ratio:.4f}')
        self._emit(f'Rotation radius cu : {current_radius_m * 1000:.2f} mm')
        self._emit(f'Rotation radius moi: {new_radius * 1000:.2f} mm')
        self._emit(f'Reference method   : {method}')

        if abs(ratio - 1.0) > 0.30:
            self._emit('[WARN] Ratio lech > 30% - kiem tra lai banh xe va san')

        self.config.rotation_radius_m = float(new_radius)
        self._emit('========== ROTATION RADIUS DONE ==========')

        return {
            'old_radius_m': current_radius_m,
            'new_radius_m': new_radius,
            'encoder_total_rad': encoder_total_rad,
            'imu_total_rad': imu_total_rad,
            'manual_total_rad': manual_measured_rad,
            'reference_method': method,
            'ratio': ratio,
        }

    # ============================================================
    # FULL CALIBRATION
    # ============================================================

    async def run_full(self,
                       skip_motors: bool = False,
                       skip_wheel: bool = False,
                       skip_rotation: bool = False,
                       wheel_distance_m: float = 1.0,
                       rotation_count: float = 1.0,
                       manual_distance_m: Optional[float] = None,
                       manual_rotation_rad: Optional[float] = None) -> dict:
        """
        Chay full calibration theo thu tu dung:
            1. Motor Kv/Ks  (robot phai duoc nhac len!)
            2. Wheel diameter
            3. Rotation radius

        Sau khi xong, ghi vao file config.
        """
        self._emit('################################################')
        self._emit('#   FULL ROBOT CALIBRATION                     #')
        self._emit('################################################')

        results = {}

        if not skip_motors:
            self._emit('\n>>> Step 1/3: Motor Kv/Ks')
            self._emit('YEU CAU: Nhac robot len, banh xe khong cham dat!')
            self._emit('Co 10 giay de chuan bi...')
            await asyncio.sleep(10)
            results['motors'] = await self.calibrate_motors()
        else:
            self._emit('Skip Motor calibration.')

        if not skip_wheel:
            self._emit('\n>>> Step 2/3: Wheel Diameter')
            self._emit('YEU CAU: Dat robot tren san phang, khong gian thoang.')
            self._emit('Co 10 giay de chuan bi...')
            await asyncio.sleep(10)
            results['wheel'] = await self.calibrate_wheel_diameter(
                target_distance_m=wheel_distance_m,
                manual_measured_m=manual_distance_m,
            )
        else:
            self._emit('Skip Wheel Diameter calibration.')

        if not skip_rotation:
            self._emit('\n>>> Step 3/3: Rotation Radius')
            self._emit('YEU CAU: Khong gian xung quanh >= 1m.')
            self._emit('Co 10 giay de chuan bi...')
            await asyncio.sleep(10)
            results['rotation'] = await self.calibrate_rotation_radius(
                target_rotations=rotation_count,
                manual_measured_rad=manual_rotation_rad,
            )
        else:
            self._emit('Skip Rotation Radius calibration.')

        # Save
        self.config.calibration_time = time.strftime('%Y-%m-%d %H:%M:%S')
        self.save_config()
        self._emit(f'\nDA LUU vao {self.config_path}')
        self._emit('Hay restart Python program de load tham so moi.')

        return results

    # ============================================================
    # FILE I/O
    # ============================================================

    def save_config(self, path: Optional[str] = None) -> bool:
        """Luu config ra file JSON (atomic)."""
        target = path or self.config_path
        try:
            tmp = target + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
            os.replace(tmp, target)
            self._emit(f'Saved config to {target}')
            return True
        except Exception as e:
            self._emit(f'[ERR] Save config failed: {e}')
            return False

    def load_config(self, path: Optional[str] = None) -> bool:
        """Load config tu file JSON. Return True neu load thanh cong."""
        target = path or self.config_path
        if not os.path.exists(target):
            self._emit(f'Config file khong ton tai: {target} (dung gia tri mac dinh)')
            return False
        try:
            with open(target, 'r', encoding='utf-8') as f:
                d = json.load(f)
            self.config = RobotConfig.from_dict(d)
            self._emit(f'Loaded config from {target}')
            self._emit(f'  wheel_diameter_m  = {self.config.wheel_diameter_m}')
            self._emit(f'  rotation_radius_m = {self.config.rotation_radius_m}')
            kv_avg, ks_avg = self.config.average_kv_ks()
            self._emit(f'  avg_kv = {kv_avg:.4f}, avg_ks = {ks_avg:.2f}')
            return True
        except Exception as e:
            self._emit(f'[ERR] Load config failed: {e}')
            return False


class CalibrationAborted(Exception):
    pass
