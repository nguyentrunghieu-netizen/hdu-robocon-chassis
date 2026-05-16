"""
serial_manager.py
=================
Quan ly giao tiep serial voi Arduino:
  - SerialManager: gui lenh, doc telemetry, calibration commands
  - find_serial_port, list_detectable_serial_ports
  - probe_lidar_port, probe_arduino_port
  - auto_detect_ports
"""
import glob
import json
import os
import re
import sys
import threading
import time

import serial


class SerialManager:
    _TELM_RE = re.compile(
        r'^T\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)'
        r'\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)'
        r'(?:\s+(0|1))?'
        r'(?:\s+(-?\d+\.?\d*))?'  # field 8: wheel-based wz_enc
    )

    def __init__(self, port, baudrate=115200, config_path='robot_config.json'):
        self.port = port
        self.baudrate = baudrate
        self.config_path = config_path
        self.ser = None
        self.measured_vx = 0.0
        self.measured_vy = 0.0
        self.measured_wz = 0.0
        self.measured_wz_enc = 0.0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta_deg = 0.0
        self.imu_ok = False
        self.last_telm_time = 0.0
        self._qr_response = None
        self._qr_event = threading.Event()
        self._write_lock = threading.Lock()

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.05)
            time.sleep(2)
            self.ser.reset_input_buffer()
            print(f"[Serial] Connected: {self.port}")
            self._apply_config_to_arduino(self.config_path)
            return True
        except serial.SerialException as exc:
            print(f"[Serial] Connect error {self.port}: {exc}")
            self.ser = None
            return False

    def _apply_config_to_arduino(self, config_path=None):
        """Load robot config and push runtime parameters to Arduino."""
        try:
            target_path = config_path or self.config_path
            if not os.path.exists(target_path):
                return
            with open(target_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            motor_params = cfg.get('motor_params', {})
            kvs, kss = [], []
            for wheel in motor_params.values():
                for direction in wheel.values():
                    kv = float(direction.get('kv', 0.0))
                    ks = float(direction.get('ks', 0.0))
                    if kv > 0.0:
                        kvs.append(kv)
                        kss.append(ks)
            if kvs:
                avg_kv = sum(kvs) / len(kvs)
                avg_ks = sum(kss) / len(kss)
                self.send_kv_ks(avg_kv, avg_ks)
                print(f'[Serial] Applied KV: kv={avg_kv:.5f} ks={avg_ks:.3f}')
            rpm_max = cfg.get('rpm_max', 0.0)
            if rpm_max and float(rpm_max) > 10.0:
                self.send_rpm_max(float(rpm_max))
            radius = cfg.get('rotation_radius_m', 0.0)
            if radius and float(radius) > 0.01:
                self.send_radius(float(radius))
            diameter = cfg.get('wheel_diameter_m', 0.0)
            if diameter and float(diameter) > 0.01:
                self.send_wheel_diameter(float(diameter))
        except Exception as exc:
            print(f'[Serial] _apply_config_to_arduino error: {exc}')

    def is_open(self):
        return self.ser is not None and self.ser.is_open

    def send_velocity(self, vx, vy, omega):
        if self.ser and self.ser.is_open:
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

    def send_kv_ks(self, kv: float, ks: float):
        if self.ser and self.ser.is_open:
            cmd = f'KV {kv:.5f} {ks:.3f}\n'
            with self._write_lock:
                try:
                    self.ser.write(cmd.encode('ascii'))
                    print(f'[Serial] Sent: {cmd.strip()}')
                except serial.SerialException:
                    self.ser = None

    def send_rpm_max(self, rpm_max: float):
        if self.ser and self.ser.is_open:
            cmd = f'RPMMAX {rpm_max:.2f}\n'
            with self._write_lock:
                try:
                    self.ser.write(cmd.encode('ascii'))
                    print(f'[Serial] Sent: {cmd.strip()}')
                except serial.SerialException:
                    self.ser = None

    def send_radius(self, radius_m: float):
        if self.ser and self.ser.is_open:
            cmd = f'RADIUS {radius_m:.5f}\n'
            with self._write_lock:
                try:
                    self.ser.write(cmd.encode('ascii'))
                    print(f'[Serial] Sent: {cmd.strip()}')
                except serial.SerialException:
                    self.ser = None

    def send_wheel_diameter(self, diameter_m: float):
        if self.ser and self.ser.is_open:
            cmd = f'WHEEL {diameter_m:.5f}\n'
            with self._write_lock:
                try:
                    self.ser.write(cmd.encode('ascii'))
                    print(f'[Serial] Sent: {cmd.strip()}')
                except serial.SerialException:
                    self.ser = None

    def send_raw_pwm(self, m0, m1, m2, m3):
        if self.ser and self.ser.is_open:
            cmd = f'P {m0:.1f} {m1:.1f} {m2:.1f} {m3:.1f}\n'
            with self._write_lock:
                try:
                    self.ser.write(cmd.encode('ascii'))
                except serial.SerialException:
                    self.ser = None

    def reset_odometry(self):
        if self.ser and self.ser.is_open:
            with self._write_lock:
                try:
                    self.ser.write(b'R\n')
                except serial.SerialException:
                    self.ser = None

    def send_imu_offset(self, target_yaw_rad):
        if self.ser and self.ser.is_open:
            cmd = f'IMU_OFFSET {target_yaw_rad:.6f}\n'
            with self._write_lock:
                try:
                    self.ser.write(cmd.encode('ascii'))
                except serial.SerialException:
                    self.ser = None

    def send_heading_control_mode(self, external_control: bool):
        if self.ser and self.ser.is_open:
            cmd = f'HEADCTRL {1 if external_control else 0}\n'
            with self._write_lock:
                try:
                    self.ser.write(cmd.encode('ascii'))
                except serial.SerialException:
                    self.ser = None

    def query_raw_rpm(self, timeout=0.5):
        if not (self.ser and self.ser.is_open):
            return None
        self._qr_event.clear()
        self._qr_response = None
        with self._write_lock:
            try:
                self.ser.write(b'Q\n')
            except serial.SerialException:
                self.ser = None
                return None
        if self._qr_event.wait(timeout):
            return self._qr_response
        return None

    def read_feedback(self):
        if self.ser and self.ser.is_open:
            try:
                last_line = None
                while self.ser.in_waiting:
                    line = self.ser.readline().decode('ascii', errors='ignore').strip()
                    if line:
                        if line.startswith('QR '):
                            _m = re.match(
                                r'^QR\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)'
                                r'\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)', line)
                            if _m:
                                self._qr_response = [float(_m.group(i)) for i in range(1, 5)]
                                self._qr_event.set()
                            continue
                        match = self._TELM_RE.match(line)
                        if match:
                            self.measured_vx = -float(match.group(1))
                            self.measured_vy = float(match.group(2))
                            self.measured_wz = float(match.group(3))
                            self.robot_x = float(match.group(4))
                            self.robot_y = float(match.group(5))
                            self.robot_theta_deg = float(match.group(6))
                            self.imu_ok = match.group(7) == '1' if match.group(7) is not None else False
                            if match.group(8) is not None:
                                self.measured_wz_enc = float(match.group(8))
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
    preferred_ports = [port for port in (preferred_ports or []) if port]
    exclude_ports = set(exclude_ports or [])
    candidates = list_detectable_serial_ports(preferred_ports=preferred_ports)

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


def list_detectable_serial_ports(preferred_ports=None):
    preferred_ports = [port for port in (preferred_ports or []) if port]
    candidates = []

    for port in preferred_ports:
        if port not in candidates:
            candidates.append(port)

    if sys.platform.startswith('linux'):
        platform_ports = sorted(glob.glob('/dev/ttyACM*')) + sorted(glob.glob('/dev/ttyUSB*'))
        for port in platform_ports:
            if port not in candidates:
                candidates.append(port)
        return candidates

    if sys.platform == 'darwin':
        platform_ports = sorted(glob.glob('/dev/tty.usbmodem*') + glob.glob('/dev/tty.usbserial*'))
        for port in platform_ports:
            if port not in candidates:
                candidates.append(port)
        return candidates

    for port in [f'COM{i}' for i in range(1, 30)]:
        try:
            test_ser = serial.Serial(port, 115200, timeout=0.05)
            test_ser.close()
            if port not in candidates:
                candidates.append(port)
        except (serial.SerialException, OSError):
            continue
    return candidates


def probe_lidar_port(port, baudrates=(460800, 256000, 115200), timeout=0.5):
    for baud in baudrates:
        try:
            with serial.Serial(port, baud, timeout=timeout) as ser:
                try:
                    ser.dtr = False
                except (OSError, serial.SerialException):
                    pass
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                ser.write(b'\xA5\x25')
                ser.flush()
                time.sleep(0.1)
                ser.reset_input_buffer()
                ser.write(b'\xA5\x50')
                ser.flush()
                time.sleep(0.15)
                resp = ser.read(7)
                bytes_in_buf = ser.in_waiting
                if len(resp) >= 2 and resp[0] == 0xA5 and resp[1] == 0x5A:
                    return True
                if len(resp) >= 2 and resp[0] == 0x54:
                    return True
                if bytes_in_buf > 50 or (len(resp) >= 4 and resp.count(0x54) >= 1):
                    return True
        except (serial.SerialException, OSError):
            continue
        except Exception:
            continue
    return False


def probe_arduino_port(port, baudrate=115200, timeout=2.5):
    try:
        ser = serial.Serial(port, baudrate, timeout=0.3)
    except (serial.SerialException, OSError):
        return False

    try:
        time.sleep(0.5)
        ser.reset_input_buffer()
        try:
            ser.write(b'S\n')
        except Exception:
            pass

        deadline = time.monotonic() + timeout
        buf = b''
        while time.monotonic() < deadline:
            try:
                chunk = ser.read(64)
            except Exception:
                break

            if chunk:
                buf += chunk
                if b'MECANUM_READY' in buf or b'[IMU]' in buf or b'ODOM_RESET' in buf:
                    return True
                for line in buf.split(b'\n'):
                    line = line.strip()
                    if line.startswith(b'T ') and line.count(b' ') >= 5:
                        return True
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
    all_ports = list_detectable_serial_ports(preferred_ports=[preferred_lidar, preferred_arduino])
    if not all_ports:
        print('[Auto-detect] Khong tim thay port serial nao de probe')
        return None, None

    print(f'[Auto-detect] Cac port se probe: {all_ports}')

    lidar_port = None
    arduino_port = None
    tried = set()
    probe_order = []

    if preferred_lidar:
        probe_order.append(preferred_lidar)
    for port in all_ports:
        if port not in probe_order:
            probe_order.append(port)

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
        print('[Auto-detect]   -> Khong phai LiDAR')

    print('[Auto-detect] Tim Arduino...')
    arduino_candidates = [port for port in all_ports if port != lidar_port]
    if preferred_arduino and preferred_arduino in arduino_candidates:
        arduino_candidates.remove(preferred_arduino)
        arduino_candidates.insert(0, preferred_arduino)

    for port in arduino_candidates:
        print(f'[Auto-detect]   Probe Arduino tren {port}...')
        if probe_arduino_port(port):
            arduino_port = port
            print(f'[Auto-detect]   -> Arduino: {port} ✓')
            break
        print('[Auto-detect]   -> Khong phai Arduino')

    print(f'[Auto-detect] Ket qua: LiDAR={lidar_port}, Arduino={arduino_port}')
    return lidar_port, arduino_port
