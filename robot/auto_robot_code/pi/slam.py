"""
slam.py
=======
SLAM va pose estimation:
  - PoseEKF: EKF 3-DOF fusion encoder + IMU + ICP
  - LidarPoseManager: quan ly LiDAR, SLAM, publish cloud/map
"""
import asyncio
import glob
import math
import sys
import threading
import time
import traceback
from collections import deque

import numpy as np
import serial

try:
    import kissicp
    LIDAR_IMPORT_ERROR = None
except Exception as exc:
    kissicp = None
    LIDAR_IMPORT_ERROR = exc

from config import (
    LIDAR_PORT, LIDAR_BAUDRATES, LIDAR_PATH_LEN,
    MAP_FILE, MAP_AUTOSAVE_PERIOD_S, MAP_SEND_EVERY_N_SCANS, MAP_SEND_MAX_POINTS,
    PC_SEND_EVERY_N_SCANS, PC_RANGE_MIN_M, PC_RANGE_MAX_M, MAP_VOXEL_SIZE_M,
    RELOC_MAX_TRIES, RELOC_MAX_MATCH_DIST_M,
)
from controllers import wrap_angle_rad
from point_cloud import PointCloudFilter, PersistentMap


class PoseEKF:
    """
    EKF 3-DOF cho pose robot tren mat phang.
    State: [px, py, theta]

    Predict : encoder vx + IMU gyro wz
    Update A: ICP full pose (chat luong-weighted)
    Update B: IMU absolute yaw (BNO085, khong drift)
    """

    Q_XY       = 0.02
    Q_THETA    = 0.008
    R_ICP_POS  = 0.03
    R_ICP_ANG  = 0.015
    R_IMU_YAW  = 0.0003

    def __init__(self):
        self.x = np.zeros(3, dtype=np.float64)
        self.P = np.eye(3, dtype=np.float64) * 0.1
        self.initialized = False

    def reset(self, pose=(0.0, 0.0, 0.0)):
        self.x = np.array([float(pose[0]), float(pose[1]), float(pose[2])], dtype=np.float64)
        self.P = np.diag([0.01, 0.01, 0.005])
        self.initialized = True

    def predict(self, enc_vx, enc_wz, dt):
        if not self.initialized:
            return
        px, py, th = self.x
        dx   = enc_vx * dt
        dth  = enc_wz  * dt
        cos_th = math.cos(th)
        sin_th = math.sin(th)
        self.x[0] = px + dx * cos_th
        self.x[1] = py + dx * sin_th
        self.x[2] = wrap_angle_rad(th + dth)
        F = np.array([
            [1.0,  0.0, -dx * sin_th],
            [0.0,  1.0,  dx * cos_th],
            [0.0,  0.0,  1.0        ],
        ], dtype=np.float64)
        q_xy = self.Q_XY * dt
        q_th = self.Q_THETA * dt
        Q = np.diag([q_xy * q_xy, q_xy * q_xy, q_th * q_th])
        self.P = F @ self.P @ F.T + Q

    def update_icp(self, icp_x, icp_y, icp_theta, icp_error=0.05, icp_overlap=0.8):
        if not self.initialized:
            return
        r_pos = max(self.R_ICP_POS, icp_error * 0.5) ** 2
        r_ang = max(self.R_ICP_ANG, icp_error * 0.3) ** 2
        if icp_overlap < 0.6:
            penalty = 1.0 + (0.6 - icp_overlap) * 8.0
            r_pos *= penalty
            r_ang *= penalty
        R = np.diag([r_pos, r_pos, r_ang])
        H = np.eye(3, dtype=np.float64)
        innov = np.array([
            icp_x     - self.x[0],
            icp_y     - self.x[1],
            wrap_angle_rad(icp_theta - self.x[2]),
        ], dtype=np.float64)
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        self.x = self.x + K @ innov
        self.x[2] = wrap_angle_rad(self.x[2])
        self.P = (np.eye(3, dtype=np.float64) - K @ H) @ self.P

    def update_imu_yaw(self, imu_theta_rad):
        if not self.initialized:
            return
        innov = wrap_angle_rad(imu_theta_rad - self.x[2])
        h = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        S_val = float(h @ self.P @ h.T) + self.R_IMU_YAW
        if abs(S_val) < 1e-12:
            return
        K = (self.P @ h.T) / S_val
        self.x += K[:, 0] * innov
        self.x[2] = wrap_angle_rad(self.x[2])
        self.P = (np.eye(3, dtype=np.float64) - K @ h) @ self.P

    @property
    def pose(self):
        return (float(self.x[0]), float(self.x[1]), float(self.x[2]))

    @property
    def position_std(self):
        return float(math.sqrt(max(0.0, (self.P[0, 0] + self.P[1, 1]) / 2.0)))

    @property
    def heading_std_deg(self):
        return float(math.degrees(math.sqrt(max(0.0, self.P[2, 2]))))


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
        self.pc_filter = PointCloudFilter()
        self._pc_scan_counter = 0
        self.map = PersistentMap()
        self.map_file = map_file
        self.autosave_period = float(autosave_period)
        self._map_send_counter = 0
        self._save_request = False

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
        try:
            if self.map.size > 50:
                self.map.save(self.map_file, pose=self.map.last_pose, note='shutdown')
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
                test_ser.write(b'\xA5\x25')
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
                     rejected, hard_resets, point_count, map_points,
                     heading_source='pose_ekf', heading_std_deg=0.0,
                     icp_pose=None):
        now = time.monotonic()
        self.path.append((float(pose[0]), float(pose[1])))
        icp_payload = {}
        if icp_pose is not None:
            icp_payload = {
                'lidar_icp_x': float(icp_pose[0]),
                'lidar_icp_y': float(icp_pose[1]),
                'lidar_icp_theta_deg': float(math.degrees(icp_pose[2])),
                'lidar_icp_last_update': now,
            }
        with self.app_state.lock:
            self.app_state.stats.update({
                'lidar_enabled': True,
                'lidar_status': 'TRACKING',
                'lidar_x': float(pose[0]),
                'lidar_y': float(pose[1]),
                'lidar_theta_deg': float(math.degrees(pose[2])),
                'heading_effective_deg': float(math.degrees(pose[2])),
                'heading_source': str(heading_source or 'pose_ekf'),
                'heading_bias_deg': 0.0,
                'heading_std_deg': float(heading_std_deg),
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
            if icp_payload:
                self.app_state.stats.update(icp_payload)
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
        self._map_send_counter += 1
        if not force and self._map_send_counter % MAP_SEND_EVERY_N_SCANS != 0:
            return
        sub = self.map.downsample_for_send(MAP_SEND_MAX_POINTS)
        if sub.shape[0] == 0:
            payload = []
        else:
            payload = [
                [round(float(p[0]), 3), round(float(p[1]), 3)]
                for p in sub
            ]
        with self.app_state.lock:
            self.app_state.stats['lidar_map'] = payload
            self.app_state.stats['lidar_map_total'] = self.map.size
            self.app_state.stats['lidar_map_loaded'] = self.map.loaded_from_file

    def request_save(self):
        self._save_request = True

    def _try_save_if_due(self, pose):
        now = time.monotonic()
        due = self.autosave_period > 0 and (now - self.map.last_save_t) >= self.autosave_period
        if not (due or self._save_request):
            return
        if self.map.size < 50 and not self._save_request:
            self.map.last_save_t = now
            return
        ok = self.map.save(self.map_file, pose=pose,
                           note='auto' if due else 'manual')
        with self.app_state.lock:
            self.app_state.stats['lidar_map_last_save'] = time.time() if ok else 0.0
            self.app_state.stats['lidar_map_save_ok'] = bool(ok)
        self._save_request = False

    def _try_relocalize(self, first_scan_points):
        if not self.map.loaded_from_file or self.map.size < 50:
            return None
        if first_scan_points is None or first_scan_points.shape[0] < kissicp.ICP_MIN_MATCHES:
            return None

        init_pose = (float(self.map.last_pose[0]),
                     float(self.map.last_pose[1]),
                     float(self.map.last_pose[2]))

        target = self.map.points_xy
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
        skip_load = getattr(self, '_skip_load', False)
        loaded = False if skip_load else self.map.load(self.map_file)
        if skip_load:
            print('[Map] --no-load-map: skipping load, starting fresh')
        if loaded:
            with self.app_state.lock:
                self.app_state.stats['lidar_map_loaded'] = True
                self.app_state.stats['lidar_map_total'] = self.map.size
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
        consecutive_successes = 0
        scan_count = 0
        accepted = 0
        rejected = 0
        hard_resets = 0
        relocalized = not loaded

        ekf = PoseEKF()
        ekf.reset(pose)

        self.path.clear()
        self.path.append((pose[0], pose[1]))
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

            dt_scan = 1.0 / kissicp.ASSUMED_SCAN_RATE_HZ
            with self.app_state.lock:
                enc_vx_raw      = float(self.app_state.stats.get('measured_vx', 0.0) or 0.0)
                enc_wz          = float(self.app_state.stats.get('measured_wz', 0.0) or 0.0)
                imu_theta_raw   = float(self.app_state.stats.get('robot_theta_deg', 0.0) or 0.0)
                has_imu_theta   = bool(self.app_state.stats.get('imu_ok', False))
                calibration_active = bool(getattr(self.app_state, 'calibration_active', False)
                                          or self.app_state.stats.get('calibration_active', False))
            enc_vx = 0.0 if calibration_active else -enc_vx_raw
            if calibration_active:
                enc_wz = 0.0
            imu_theta_rad = math.radians(imu_theta_raw) if has_imu_theta else None

            ekf.predict(enc_vx, enc_wz, dt_scan)

            if imu_theta_rad is not None:
                ekf.update_imu_yaw(imu_theta_rad)

            ekf_pose = ekf.pose
            velocity_per_scan = (
                ekf_pose[0] - pose[0],
                ekf_pose[1] - pose[1],
                wrap_angle_rad(ekf_pose[2] - pose[2]),
            )
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

            if not relocalized:
                reloc_pose = self._try_relocalize(points)
                if reloc_pose is not None:
                    pose = reloc_pose
                    ekf.reset(pose)
                    self.path.clear()
                    self.path.append((pose[0], pose[1]))
                    serial_mgr = self.app_state.serial_mgr
                    if serial_mgr is not None:
                        serial_mgr.send_imu_offset(pose[2])
                        print(f'[IMU] Sent IMU_OFFSET {math.degrees(pose[2]):+.2f} deg '
                              f'to align with reloc pose')
                    with self.app_state.lock:
                        self.app_state.stats['lidar_map_relocalized'] = True
                else:
                    print('[Map] Relocalization failed, starting fresh map at origin')
                    self.map.reset()
                    pose = (0.0, 0.0, 0.0)
                    ekf.reset(pose)
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
                    heading_source='pose_ekf', heading_std_deg=ekf.heading_std_deg,
                )
                continue

            match_thresh = kissicp.adaptive_threshold(velocity_per_scan)
            target_map = local_map.query_near(pose)
            if target_map.shape[0] < kissicp.ICP_MIN_MATCHES:
                local_map.add_scan(points, pose)
                continue

            result, fail_reason = kissicp.icp_2d(
                points,
                target_map,
                init_delta=ekf.pose,
                max_match_dist=match_thresh,
            )

            if result is None:
                consecutive_failures += 1
                consecutive_successes = 0
                rejected += 1
                if consecutive_failures >= kissicp.MAX_CONSECUTIVE_FAILURES:
                    local_map.scans_world.clear()
                    local_map.add_scan(points, pose)
                    consecutive_failures = 0
                    hard_resets += 1
                    velocity_per_scan = (0.0, 0.0, 0.0)
                    pose = ekf.pose
                    status = 'RESET'
                else:
                    status = 'ICP_FAIL'
                    pose = ekf.pose
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
            consecutive_successes += 1
            new_pose, error, matches, overlap = result
            step = kissicp.relative_delta(pose, new_pose)

            if not kissicp.velocity_is_plausible(step):
                rejected += 1
                consecutive_successes = 0
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

            if calibration_active:
                pose = new_pose
                ekf.reset(pose)
            else:
                ekf.update_icp(new_pose[0], new_pose[1], new_pose[2],
                               icp_error=float(error), icp_overlap=float(overlap))
                if imu_theta_rad is not None:
                    ekf.update_imu_yaw(imu_theta_rad)
                pose = ekf.pose
            velocity_per_scan = step
            local_map.add_scan(points, pose)
            accepted += 1
            map_points = sum(scan.shape[0] for scan in local_map.scans_world)

            try:
                self.pc_filter.update(points, pose)
            except Exception as filter_exc:
                print(f'[LiDAR] PC filter error: {type(filter_exc).__name__}: {filter_exc!r}')

            self._pc_scan_counter += 1
            send_cloud_now = (self._pc_scan_counter % max(1, PC_SEND_EVERY_N_SCANS) == 0)
            if send_cloud_now:
                self._publish_cloud()

            try:
                pts_local = points[:, :2] if points.ndim == 2 else points
                dists = np.linalg.norm(pts_local, axis=1)
                pts_local = pts_local[(dists >= PC_RANGE_MIN_M) & (dists <= PC_RANGE_MAX_M)]
                if pts_local.shape[0] > 0:
                    keys = np.floor(pts_local / MAP_VOXEL_SIZE_M).astype(np.int64)
                    key1d = keys[:, 0] * 1000003 + keys[:, 1]
                    _, idx_unique = np.unique(key1d, return_index=True)
                    pts_local = pts_local[idx_unique]
                    c, s = math.cos(pose[2]), math.sin(pose[2])
                    world_pts = np.empty_like(pts_local)
                    world_pts[:, 0] = c * pts_local[:, 0] - s * pts_local[:, 1] + pose[0]
                    world_pts[:, 1] = s * pts_local[:, 0] + c * pts_local[:, 1] + pose[1]
                    self.map.add_scan_with_raycast(
                        world_pts.astype(np.float32),
                        sensor_xy=(pose[0], pose[1]),
                    )
                self.map.last_pose = np.array(
                    [float(pose[0]), float(pose[1]), float(pose[2])],
                    dtype=np.float32)
            except Exception as map_exc:
                print(f'[Map] update error: {type(map_exc).__name__}: {map_exc!r}')

            self._publish_map()
            self._try_save_if_due(pose)

            self._update_pose(
                pose, step, error, matches, overlap, scan_count, accepted,
                rejected, hard_resets, points.shape[0], map_points,
                heading_source='icp_cal' if calibration_active else 'pose_ekf',
                heading_std_deg=ekf.heading_std_deg,
                icp_pose=new_pose,
            )
