"""
web_server.py
=============
HTTP server, API endpoints, calibration runner:
  - WebHandler: GET/POST request handlers
  - ThreadedHTTPServer
  - update_web_frame()
  - _calibration_run_async()
  - run_calibration_cli()
"""
import asyncio
import json
import math
import os
import sys
import threading
import time
import traceback
from http import server
from socketserver import ThreadingMixIn

import cv2

from config import (
    LIDAR_BAUDRATES, LIDAR_PORT, MAP_AUTOSAVE_PERIOD_S,
    NAV_LIDAR_STALE_S,
)
from planner import WaypointStore
from serial_manager import SerialManager, find_serial_port, auto_detect_ports
from slam import LidarPoseManager
from web_ui import draw_dashboard, HTML_PAGE, CALIBRATION_PAGE

# ===================== CALIBRATION MODULE =====================
_CAL_SUBDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Calibrate')
if _CAL_SUBDIR not in sys.path:
    sys.path.insert(0, _CAL_SUBDIR)
try:
    from calibration import Calibrator, CalibrationAborted
    _CAL_IMPORT_ERROR = None
except Exception as _exc:
    Calibrator = None
    CalibrationAborted = None
    _CAL_IMPORT_ERROR = _exc


# ======================== CALIBRATION RUNNER STATE ========================
_CAL_LOG_MAXLINES = 200
_cal_runner = {
    'task': None,
    'status': 'idle',    # idle/running/done/error/aborted
    'progress': 0.0,
    'message': '',
    'log': [],
    'cal': None,
}


class ThreadedHTTPServer(ThreadingMixIn, server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def update_web_frame(state, frame):
    success, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not success:
        return
    with state.lock:
        state.latest_jpeg = encoded.tobytes()


class WebHandler(server.BaseHTTPRequestHandler):
    state = None

    def do_GET(self):
        if self.path == '/':
            self._send_html()
            return
        if self.path == '/calibration':
            self._send_calibration()
            return
        if self.path == '/state':
            self._send_state()
            return
        if self.path == '/stream.mjpg':
            self._send_stream()
            return
        if self.path == '/api/calibration/status':
            self._cal_status()
            return
        if self.path == '/api/calibration/config':
            self._cal_config()
            return

        self.send_error(404)

    def do_POST(self):
        if self.path == '/api/motor/start':
            self._set_motor_enabled(True)
            return
        if self.path == '/api/motor/stop':
            self._set_motor_enabled(False)
            return
        if self.path == '/api/camera_tracking/start':
            self._set_camera_tracking_enabled(True)
            return
        if self.path == '/api/camera_tracking/stop':
            self._set_camera_tracking_enabled(False)
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
        if self.path == '/api/calibration/start':
            self._cal_start()
            return
        if self.path == '/api/calibration/abort':
            self._cal_abort()
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

    def _send_calibration(self):
        body = CALIBRATION_PAGE.encode('utf-8')
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

    def _set_camera_tracking_enabled(self, enabled):
        should_stop = False
        with self.state.lock:
            self.state.camera_tracking_enabled = bool(enabled)
            self.state.ball_detected = False
            self.state.new_measurement = False
            self.state.stats['camera_tracking_enabled'] = self.state.camera_tracking_enabled
            self.state.stats['ball_detected'] = False
            self.state.stats['status'] = 'SEARCHING' if enabled else 'TRACKING_OFF'
            if not enabled and not self.state.navigation_active:
                should_stop = True
            payload = {
                'ok': True,
                'camera_tracking_enabled': self.state.camera_tracking_enabled,
            }

        if should_stop and self.state.serial_mgr is not None:
            self.state.serial_mgr.send_stop()

        self._send_json(payload)

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
                wp_theta = waypoint.get('theta', None)
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

    # ---- Calibration endpoints ----

    def _cal_start(self):
        if Calibrator is None:
            self._send_json({'error': f'calibration module unavailable: {_CAL_IMPORT_ERROR}'}, 503)
            return
        data = self._read_json_body()
        mode = str(data.get('mode', 'all'))
        if mode not in ('all', 'motors', 'wheel', 'rotation'):
            self._send_json({'error': 'invalid mode'}, 400)
            return

        if _cal_runner['status'] == 'running':
            self._send_json({'error': 'calibration already running'}, 409)
            return

        serial_mgr = self.state.serial_mgr
        if serial_mgr is None or not serial_mgr.is_open():
            self._send_json({'error': 'serial not connected'}, 409)
            return

        manual_distance = data.get('manual_distance', None)
        manual_rotation = data.get('manual_rotation', None)
        if manual_distance is not None:
            try:
                manual_distance = float(manual_distance)
            except (TypeError, ValueError):
                manual_distance = None
        if manual_rotation is not None:
            try:
                manual_rotation = float(manual_rotation)
            except (TypeError, ValueError):
                manual_rotation = None

        if mode in ('wheel', 'all'):
            now = time.monotonic()
            with self.state.lock:
                lidar_status = str(self.state.stats.get('lidar_status', 'DISABLED') or 'DISABLED')
                lidar_last_update = float(self.state.stats.get('lidar_last_update', 0.0) or 0.0)
            lidar_age = now - lidar_last_update if lidar_last_update > 0.0 else float('inf')
            if lidar_status != 'TRACKING' or lidar_age > 1.0:
                self._send_json({
                    'error': f'wheel calibration requires LiDAR TRACKING '
                             f'(status={lidar_status}, age={lidar_age:.2f}s)'
                }, 409)
                return

        with self.state.lock:
            self.state.motor_enabled = False
            self.state.navigation_active = False
            self.state.calibration_active = True
            self.state.calibration_mode = mode
            self.state.stats['motor_enabled'] = False
            self.state.stats['motor_reason'] = 'Calibration running'
            self.state.stats['navigation_active'] = False
            self.state.stats['calibration_active'] = True
            self.state.stats['calibration_mode'] = mode

        config_path = self.state.config_path
        app_state = self.state

        _cal_runner['status'] = 'running'
        _cal_runner['progress'] = 0.0
        _cal_runner['message'] = 'Starting...'
        _cal_runner['log'] = [f'[CAL] Start mode={mode}']

        def on_progress(p, msg):
            _cal_runner['progress'] = float(p)
            _cal_runner['message'] = str(msg)

        def on_log(msg):
            _cal_runner['log'].append(str(msg))
            if len(_cal_runner['log']) > _CAL_LOG_MAXLINES:
                _cal_runner['log'] = _cal_runner['log'][-_CAL_LOG_MAXLINES:]

        cal = Calibrator(serial_mgr, app_state, config_path=config_path,
                         progress_callback=on_progress, log_callback=on_log)
        cal.load_config()
        _cal_runner['cal'] = cal

        def _thread_fn():
            try:
                asyncio.run(_calibration_run_async(
                    cal, mode, manual_distance, manual_rotation))
                _cal_runner['status'] = 'done'
                _cal_runner['progress'] = 1.0
                _cal_runner['message'] = 'Complete'
                on_log('[CAL] Done. Robot config saved.')
            except Exception as exc:
                if CalibrationAborted and isinstance(exc, CalibrationAborted):
                    _cal_runner['status'] = 'aborted'
                    _cal_runner['message'] = 'Aborted'
                    on_log('[CAL] Aborted by user.')
                else:
                    _cal_runner['status'] = 'error'
                    _cal_runner['message'] = f'{type(exc).__name__}: {exc}'
                    on_log(f'[CAL] ERROR: {type(exc).__name__}: {exc}')
            finally:
                _cal_runner['cal'] = None
                with app_state.lock:
                    app_state.calibration_active = False
                    app_state.calibration_mode = ''
                    app_state.stats['calibration_active'] = False
                    app_state.stats['calibration_mode'] = ''
                try:
                    serial_mgr.send_stop()
                except Exception:
                    pass

        t = threading.Thread(target=_thread_fn, name='calibration', daemon=True)
        _cal_runner['task'] = t
        t.start()

        self._send_json({'ok': True, 'mode': mode})

    def _cal_status(self):
        log_tail = _cal_runner['log'][-30:]
        payload = {
            'status': _cal_runner['status'],
            'progress': _cal_runner['progress'],
            'message': _cal_runner['message'],
            'log_tail': log_tail,
        }
        self._send_json(payload)

    def _cal_abort(self):
        cal = _cal_runner.get('cal')
        if cal is not None:
            cal.abort()
        _cal_runner['status'] = 'aborted'
        _cal_runner['message'] = 'Aborted by user'
        self._send_json({'ok': True})

    def _cal_config(self):
        if Calibrator is None:
            self._send_json({'error': 'calibration module unavailable'}, 503)
            return
        config_path = self.state.config_path
        try:
            cal = Calibrator(None, None, config_path=config_path)
            cal.load_config()
            self._send_json(cal.config.to_dict())
        except Exception as exc:
            self._send_json({'error': str(exc)}, 500)


async def _calibration_run_async(cal, mode, manual_distance=None, manual_rotation=None):
    """Run the requested calibration mode. Called inside asyncio.run() in a thread."""
    app_state = getattr(cal, 'app_state', None)
    if app_state is not None:
        with app_state.lock:
            app_state.calibration_active = True
            app_state.calibration_mode = mode
            app_state.navigation_active = False
            app_state.stats['calibration_active'] = True
            app_state.stats['calibration_mode'] = mode
            app_state.stats['navigation_active'] = False
    try:
        apply_config = getattr(cal.serial_mgr, '_apply_config_to_arduino', None)
        if callable(apply_config):
            apply_config(cal.config_path)

        if mode == 'all':
            await cal.run_full(
                manual_distance_m=manual_distance,
                manual_rotation_rad=manual_rotation,
            )
        elif mode == 'motors':
            await cal.calibrate_motors()
            cal.save_config()
        elif mode == 'wheel':
            await cal.calibrate_wheel_diameter(manual_measured_m=manual_distance)
            cal.save_config()
        elif mode == 'rotation':
            await cal.calibrate_rotation_radius(manual_measured_rad=manual_rotation)
            cal.save_config()

        apply_config = getattr(cal.serial_mgr, '_apply_config_to_arduino', None)
        if callable(apply_config):
            apply_config(cal.config_path)
    finally:
        if app_state is not None:
            with app_state.lock:
                app_state.calibration_active = False
                app_state.calibration_mode = ''
                app_state.stats['calibration_active'] = False
                app_state.stats['calibration_mode'] = ''


def run_calibration_cli(args, app_state_cls):
    """Run calibration via --calibrate CLI flag, then exit.

    app_state_cls: SharedAppState class (passed in to avoid circular import).
    """
    if Calibrator is None:
        print(f'[CAL] calibration module not available: {_CAL_IMPORT_ERROR}')
        sys.exit(1)

    print('=' * 60)
    print(f'ROBOT CALIBRATION CLI  mode={args.calibrate}  config={args.config}')
    print('=' * 60)

    use_lidar = args.calibrate in ('all', 'wheel', 'rotation') and not args.no_lidar

    detected_lidar_port, detected_arduino_port = None, None
    if not args.no_auto_detect:
        lidar_baudrates = (args.lidar_baudrate,) + tuple(
            b for b in LIDAR_BAUDRATES if b != args.lidar_baudrate)
        detected_lidar_port, detected_arduino_port = auto_detect_ports(
            preferred_lidar=args.lidar_port if not args.no_lidar else None,
            preferred_arduino=args.serial_port,
            lidar_baudrates=lidar_baudrates,
        )

    arduino_port = detected_arduino_port or args.serial_port or find_serial_port()
    if arduino_port is None:
        print('[CAL] Cannot find Arduino. Use --serial-port <port>')
        sys.exit(1)

    serial_mgr = SerialManager(arduino_port, config_path=args.config)
    if not serial_mgr.connect():
        print(f'[CAL] Cannot connect serial on {arduino_port}')
        sys.exit(1)

    app_state = app_state_cls()
    app_state.config_path = args.config

    lidar_mgr = None
    if use_lidar:
        lidar_port = detected_lidar_port or args.lidar_port
        lidar_baudrates = (args.lidar_baudrate,) + tuple(
            b for b in LIDAR_BAUDRATES if b != args.lidar_baudrate)
        lidar_mgr = LidarPoseManager(app_state, lidar_port, lidar_baudrates)
        app_state.lidar_mgr = lidar_mgr
        app_state.serial_mgr = serial_mgr
        lidar_mgr.start()
        print('[CAL] Waiting 5s for LiDAR/SLAM to stabilize...')
        for i in range(5, 0, -1):
            print(f'  {i}...')
            time.sleep(1)

    def on_log(msg):
        print(f'  {msg}')

    def on_progress(p, msg):
        bar_len = 30
        filled = int(p * bar_len)
        bar = '#' * filled + '-' * (bar_len - filled)
        sys.stdout.write(f'\r[{bar}] {p*100:5.1f}% {str(msg)[:50]:<50}')
        sys.stdout.flush()

    cal = Calibrator(serial_mgr, app_state, config_path=args.config,
                     progress_callback=on_progress, log_callback=on_log)
    cal.load_config()

    try:
        asyncio.run(_calibration_run_async(
            cal, args.calibrate,
            manual_distance=getattr(args, 'manual_distance', None),
            manual_rotation=getattr(args, 'manual_rotation', None),
        ))
        print('\n[CAL] Calibration complete. Config saved to:', args.config)
    except KeyboardInterrupt:
        print('\n[CAL] Interrupted.')
        cal.abort()
    except Exception as exc:
        print(f'\n[CAL] Error: {type(exc).__name__}: {exc}')
        traceback.print_exc(limit=6)
    finally:
        serial_mgr.send_velocity(0.0, 0.0, 0.0)
        time.sleep(0.3)
        app_state.running = False
        if lidar_mgr:
            lidar_mgr.stop()
        serial_mgr.close()
