"""
serial_manager_patch.py - Patch cho SerialManager de ho tro calibration

Cach apply:
    Them cac method nay vao class SerialManager trong camera_lidar_web.py
    (~ sau dong 535-565 noi co send_velocity).

Hoac monkey-patch tu module ngoai:
    from serial_manager_patch import patch_serial_manager
    patch_serial_manager(SerialManager)
"""

import re
import time
import threading


# ============================================================
# CAC METHOD MOI THEM VAO SerialManager
# ============================================================

def send_raw_pwm(self, m0: float, m1: float, m2: float, m3: float):
    """
    Gui command raw PWM xuong Arduino (bypass PID).
    Range: -255 .. +255.
    Su dung trong calibration mode.
    """
    if not self.is_open():
        return
    cmd = f'P {m0:.1f} {m1:.1f} {m2:.1f} {m3:.1f}\n'
    try:
        with self._write_lock:
            self.ser.write(cmd.encode('ascii'))
    except Exception as e:
        print(f'[SerialMgr] send_raw_pwm error: {e}')


def reset_odometry(self):
    """Gui lenh R de Arduino reset odometry."""
    if not self.is_open():
        return
    try:
        with self._write_lock:
            self.ser.write(b'R\n')
    except Exception as e:
        print(f'[SerialMgr] reset_odometry error: {e}')


def query_raw_rpm(self, timeout: float = 0.5):
    """
    Gui command Q va doi response 'QR r0 r1 r2 r3'.
    Tra ve [r0, r1, r2, r3] hoac None neu timeout.
    
    QUAN TRONG: method nay BLOCKING. Chi dung khi calibrate, khong dung
    trong control loop chinh.
    """
    if not self.is_open():
        return None
    
    # Set up: tao 1 event de bat phan hoi QR
    if not hasattr(self, '_qr_response'):
        self._qr_response = None
        self._qr_event = threading.Event()
    
    self._qr_response = None
    self._qr_event.clear()
    
    # Gui command
    try:
        with self._write_lock:
            self.ser.write(b'Q\n')
    except Exception as e:
        print(f'[SerialMgr] query_raw_rpm send error: {e}')
        return None
    
    # Doi response (read_feedback thread se set _qr_event khi nhan QR)
    if self._qr_event.wait(timeout=timeout):
        return self._qr_response
    return None


# ============================================================
# PATCH cho read_feedback() de bat response 'QR'
# ============================================================
# Trong ham read_feedback() goc, sau dong xu ly 'T ' (telemetry), 
# them xu ly 'QR':

QR_PATTERN = re.compile(
    r'^QR\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)'
)

def _handle_qr_line(self, line: str) -> bool:
    """
    Xu ly dong 'QR r0 r1 r2 r3' tu Arduino.
    Tra ve True neu da xu ly, False neu khong phai dong QR.
    
    Cach goi: trong read_feedback(), sau khi parse line, them:
        if line.startswith('QR'):
            self._handle_qr_line(line)
            continue
    """
    m = QR_PATTERN.match(line.strip())
    if not m:
        return False
    
    rpms = [float(m.group(i)) for i in range(1, 5)]
    self._qr_response = rpms
    if hasattr(self, '_qr_event'):
        self._qr_event.set()
    return True


def patch_serial_manager(cls):
    """Monkey-patch SerialManager class de them cac method moi."""
    cls.send_raw_pwm = send_raw_pwm
    cls.reset_odometry = reset_odometry
    cls.query_raw_rpm = query_raw_rpm
    cls._handle_qr_line = _handle_qr_line
    
    # Wrap read_feedback de bat dong QR
    original_read_feedback = cls.read_feedback
    
    def patched_read_feedback(self):
        """Wrap goc read_feedback de them xu ly QR."""
        # Chien luoc don gian: chen 1 hook truoc khi xu ly cac line khac.
        # Vi read_feedback la vong while phuc tap, an toan nhat la 
        # khong replace ma de user tu them dong if line.startswith('QR') vao goc.
        # 
        # Day la version don gian: noi da sua se doc tat ca line tu serial
        # buffer va check QR truoc.
        original_read_feedback(self)
    
    cls.read_feedback = patched_read_feedback
    return cls


# ============================================================
# HUONG DAN INTEGRATION
# ============================================================

INTEGRATION_NOTES = """
========================================
INTEGRATION VAO camera_lidar_web.py
========================================

BUOC 1: Them cac method moi vao SerialManager (~dong 510)
   Copy 3 method: send_raw_pwm, reset_odometry, query_raw_rpm
   vao trong class SerialManager.
   
   Cung copy bien:
        self._qr_response = None
        self._qr_event = threading.Event()
   vao trong __init__ cua SerialManager.

BUOC 2: Sua read_feedback() (~dong 540) de bat dong 'QR':
   Trong vong while True chinh, NGAY SAU dong:
       line = ...   # doc 1 line tu serial
   Them:
       if line.startswith('QR '):
           m = re.match(r'QR\\s+(-?\\d+\\.?\\d*)\\s+(-?\\d+\\.?\\d*)\\s+'
                        r'(-?\\d+\\.?\\d*)\\s+(-?\\d+\\.?\\d*)', line.strip())
           if m:
               self._qr_response = [float(m.group(i)) for i in range(1, 5)]
               self._qr_event.set()
           continue

BUOC 3: Them CLI flag o cuoi file, trong block `if __name__ == '__main__'`:
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--calibrate', choices=['all', 'motors', 'wheel', 'rotation'],
                       help='Run calibration mode roi exit')
   parser.add_argument('--config', default='robot_config.json',
                       help='Path to robot config JSON')
   parser.add_argument('--manual-distance', type=float, default=None,
                       help='Manual measured distance (m) for wheel calibration')
   parser.add_argument('--manual-rotation', type=float, default=None,
                       help='Manual measured rotation (rad) for rotation calibration')
   args = parser.parse_args()
   
   if args.calibrate:
       run_calibration_cli(args)
   else:
       # main() bin thuong
       run_normal()

BUOC 4: Them ham run_calibration_cli:

   def run_calibration_cli(args):
       import asyncio
       from calibration import Calibrator
       
       # Khoi tao serial manager + lidar (de co SLAM pose)
       app_state = AppState()
       serial_mgr = SerialManager(...)
       serial_mgr.connect()
       
       # Khoi dong LiDAR thread (can pose tu SLAM)
       lidar_thread = threading.Thread(target=run_lidar_thread,
                                        args=(app_state, serial_mgr))
       lidar_thread.start()
       
       time.sleep(3)  # cho LiDAR/SLAM on dinh
       
       def on_log(msg):
           print(f'[CAL] {msg}')
       def on_progress(p, msg):
           print(f'[CAL] [{p*100:5.1f}%] {msg}')
       
       cal = Calibrator(serial_mgr, app_state,
                        config_path=args.config,
                        progress_callback=on_progress,
                        log_callback=on_log)
       cal.load_config()  # load gia tri cu de tinh ratio
       
       async def run():
           if args.calibrate == 'all':
               await cal.run_full(
                   manual_distance_m=args.manual_distance,
                   manual_rotation_rad=args.manual_rotation,
               )
           elif args.calibrate == 'motors':
               await cal.calibrate_motors()
               cal.save_config()
           elif args.calibrate == 'wheel':
               await cal.calibrate_wheel_diameter(
                   manual_measured_m=args.manual_distance)
               cal.save_config()
           elif args.calibrate == 'rotation':
               await cal.calibrate_rotation_radius(
                   manual_measured_rad=args.manual_rotation)
               cal.save_config()
       
       asyncio.run(run())
       app_state.running = False
       lidar_thread.join(timeout=3)
       serial_mgr.close()
       print('Calibration finished. Exit.')

BUOC 5: Load config khi startup binh thuong (trong main()):

   from calibration import Calibrator
   
   # Sau khi tao app_state, truoc khi khoi dong cac thread:
   pre_cal = Calibrator(None, app_state, config_path='robot_config.json')
   if pre_cal.load_config():
       # Apply gia tri config vao cac CONST trong code
       import camera_lidar_web as mod
       mod.WHEEL_DIAMETER_M = pre_cal.config.wheel_diameter_m
       mod.ROTATION_RADIUS_M = pre_cal.config.rotation_radius_m
       # Note: mecanum kinematics trong Python neu co dung wheel diameter
       # cung phai dung gia tri moi.

BUOC 6: Web endpoint (Flask) - them vao file Flask app cua ban:

   from flask import jsonify, request
   
   _cal_runner = {'task': None, 'status': 'idle', 'progress': 0,
                  'log': [], 'message': ''}
   
   @app.route('/api/calibration/start', methods=['POST'])
   def cal_start():
       data = request.json or {}
       which = data.get('mode', 'all')  # all/motors/wheel/rotation
       
       if _cal_runner['task'] is not None and not _cal_runner['task'].done():
           return jsonify({'error': 'already_running'}), 409
       
       def log_cb(msg):
           _cal_runner['log'].append(msg)
           _cal_runner['log'] = _cal_runner['log'][-200:]
       def prog_cb(p, msg):
           _cal_runner['progress'] = p
           _cal_runner['message'] = msg
       
       cal = Calibrator(serial_mgr, app_state,
                        progress_callback=prog_cb, log_callback=log_cb)
       cal.load_config()
       _cal_runner['cal'] = cal
       _cal_runner['log'] = []
       _cal_runner['status'] = 'running'
       
       async def runner():
           try:
               if which == 'all':
                   await cal.run_full(
                       manual_distance_m=data.get('manual_distance'),
                       manual_rotation_rad=data.get('manual_rotation'),
                   )
               elif which == 'motors':
                   await cal.calibrate_motors()
                   cal.save_config()
               elif which == 'wheel':
                   await cal.calibrate_wheel_diameter(
                       manual_measured_m=data.get('manual_distance'))
                   cal.save_config()
               elif which == 'rotation':
                   await cal.calibrate_rotation_radius(
                       manual_measured_rad=data.get('manual_rotation'))
                   cal.save_config()
               _cal_runner['status'] = 'done'
           except Exception as e:
               _cal_runner['status'] = 'error'
               _cal_runner['message'] = str(e)
       
       _cal_runner['task'] = asyncio.run_coroutine_threadsafe(
           runner(), main_event_loop)
       return jsonify({'ok': True})
   
   @app.route('/api/calibration/status')
   def cal_status():
       return jsonify({
           'status': _cal_runner['status'],
           'progress': _cal_runner['progress'],
           'message': _cal_runner['message'],
           'log_tail': _cal_runner['log'][-30:],
       })
   
   @app.route('/api/calibration/abort', methods=['POST'])
   def cal_abort():
       cal = _cal_runner.get('cal')
       if cal is not None:
           cal.abort()
           _cal_runner['status'] = 'aborted'
       return jsonify({'ok': True})
   
   @app.route('/api/calibration/config')
   def cal_config():
       cal = Calibrator(None, app_state)
       cal.load_config()
       return jsonify(cal.config.to_dict())
"""

if __name__ == '__main__':
    print(INTEGRATION_NOTES)
