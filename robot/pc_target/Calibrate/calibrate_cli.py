#!/usr/bin/env python3
"""
calibrate_cli.py - Chay calibration tu terminal

Cach dung:
    # Calibrate tat ca
    python3 calibrate_cli.py --mode all

    # Chi calibrate motor (robot phai duoc nhac len)
    python3 calibrate_cli.py --mode motors

    # Calibrate wheel diameter, nhap tay khoang cach do bang thuoc:
    python3 calibrate_cli.py --mode wheel --manual-distance 1.02

    # Calibrate rotation, nhap tay goc do duoc:
    python3 calibrate_cli.py --mode rotation --manual-rotation 6.31

    # Chi load + xem config hien tai
    python3 calibrate_cli.py --show-config

    # Chi save config voi gia tri default
    python3 calibrate_cli.py --reset-config

LUU Y: File nay can chay TRONG MOI TRUONG cua robot, voi serial_mgr 
       va lidar thread da khoi dong. Day la skeleton, ban can ket noi 
       voi cac module san co cua minh (SerialManager, LiDAR, AppState).
"""

import argparse
import asyncio
import sys
import time
import threading

from calibration import Calibrator, CalibrationAborted


def main():
    parser = argparse.ArgumentParser(description='Robot calibration CLI')
    parser.add_argument('--mode',
                        choices=['all', 'motors', 'wheel', 'rotation'],
                        default='all',
                        help='Calibration mode')
    parser.add_argument('--config', default='robot_config.json',
                        help='Config file path')
    parser.add_argument('--manual-distance', type=float, default=None,
                        help='Manual measured distance for wheel cal (meters)')
    parser.add_argument('--manual-rotation', type=float, default=None,
                        help='Manual measured rotation for rotation cal (radians, '
                             'use 6.2832 for 1 turn)')
    parser.add_argument('--manual-rotation-deg', type=float, default=None,
                        help='Same as --manual-rotation but in degrees')
    parser.add_argument('--wheel-distance', type=float, default=1.0,
                        help='Target distance for wheel cal (meters)')
    parser.add_argument('--rotation-count', type=float, default=1.0,
                        help='Number of full turns for rotation cal')
    parser.add_argument('--show-config', action='store_true',
                        help='Just show current config and exit')
    parser.add_argument('--reset-config', action='store_true',
                        help='Save default config and exit')
    parser.add_argument('--serial-port', default=None,
                        help='Serial port (auto-detect if None)')
    args = parser.parse_args()

    # Convert deg sang rad neu can
    manual_rot = args.manual_rotation
    if args.manual_rotation_deg is not None:
        import math
        manual_rot = math.radians(args.manual_rotation_deg)

    # ===== SHOW / RESET CONFIG (khong can robot) =====
    if args.show_config:
        cal = Calibrator(None, None, config_path=args.config)
        if cal.load_config():
            import json
            print(json.dumps(cal.config.to_dict(), indent=2, ensure_ascii=False))
        return 0

    if args.reset_config:
        cal = Calibrator(None, None, config_path=args.config)
        cal.save_config()
        print(f'Saved default config to {args.config}')
        return 0

    # ===== CALIBRATION MODE: can robot online =====
    print('=' * 60)
    print('ROBOT CALIBRATION MODE')
    print('=' * 60)
    print(f'Mode: {args.mode}')
    print(f'Config file: {args.config}')

    # Khoi tao serial manager va LiDAR
    # NOTE: Cac import nay phu thuoc vao file goc cua ban
    # Thay duong dan import phu hop:
    try:
        # Vi du:
        # from camera_lidar_web import SerialManager, AppState, LidarSlamThread
        # from camera_lidar_web import auto_detect_arduino_port
        print('LUU Y: ban can sua import o dau ham main() de tro vao module')
        print('       cua minh (SerialManager, AppState, LidarSlamThread).')
        print()
        print('Skeleton da san sang. Hay sua imports trong file nay.')
        return 1
    except ImportError as e:
        print(f'Import error: {e}')
        return 1


def run_calibration_with_robot(args, manual_rot,
                                serial_mgr_cls,
                                app_state_cls,
                                lidar_thread_starter):
    """
    Logic chinh sau khi imports da OK.
    
    Args:
        serial_mgr_cls: class SerialManager
        app_state_cls: class AppState
        lidar_thread_starter: callable de khoi dong LiDAR thread, 
                              nhan (app_state, serial_mgr) -> Thread
    """
    app_state = app_state_cls()
    
    # Connect serial
    serial_mgr = serial_mgr_cls(port=args.serial_port)
    if not serial_mgr.connect():
        print('[ERR] Khong ket noi duoc serial. Kiem tra cap USB va quyen.')
        return 1
    
    print('[OK] Serial connected.')
    
    # Khoi dong LiDAR thread (de co SLAM pose)
    if args.mode in ('all', 'wheel', 'rotation'):
        print('[INFO] Khoi dong LiDAR thread...')
        lidar_thread = lidar_thread_starter(app_state, serial_mgr)
        # Cho LiDAR/SLAM on dinh
        for i in range(5, 0, -1):
            print(f'  Doi LiDAR san sang... {i}s')
            time.sleep(1)
    else:
        lidar_thread = None
    
    # Tao calibrator
    cal_logs = []
    def on_log(msg):
        cal_logs.append(msg)
        print(f'[CAL] {msg}')
    def on_progress(p, msg):
        # In dong duy nhat, tu xoa
        bar_len = 30
        filled = int(p * bar_len)
        bar = '#' * filled + '-' * (bar_len - filled)
        sys.stdout.write(f'\r[{bar}] {p*100:5.1f}% {msg[:50]:<50}')
        sys.stdout.flush()
    
    cal = Calibrator(serial_mgr, app_state,
                     config_path=args.config,
                     progress_callback=on_progress,
                     log_callback=on_log)
    cal.load_config()
    
    # Run
    try:
        asyncio.run(_run_async(cal, args, manual_rot))
        print('\n' + '=' * 60)
        print('CALIBRATION HOAN TAT')
        print(f'Config da luu vao: {args.config}')
        print('Hay khoi dong lai chuong trinh chinh de load tham so moi.')
        print('=' * 60)
        return 0
    except CalibrationAborted:
        print('\n[ABORTED] User huy calibration.')
        return 2
    except KeyboardInterrupt:
        print('\n[CTRL+C] Dung khan.')
        cal.abort()
        return 2
    except Exception as e:
        print(f'\n[ERR] {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
        return 3
    finally:
        serial_mgr.send_velocity(0.0, 0.0, 0.0)
        time.sleep(0.5)
        app_state.running = False
        if lidar_thread:
            lidar_thread.join(timeout=3)
        serial_mgr.close()


async def _run_async(cal, args, manual_rot):
    if args.mode == 'all':
        await cal.run_full(
            wheel_distance_m=args.wheel_distance,
            rotation_count=args.rotation_count,
            manual_distance_m=args.manual_distance,
            manual_rotation_rad=manual_rot,
        )
    elif args.mode == 'motors':
        await cal.calibrate_motors()
        cal.config.calibration_time = time.strftime('%Y-%m-%d %H:%M:%S')
        cal.save_config()
    elif args.mode == 'wheel':
        await cal.calibrate_wheel_diameter(
            target_distance_m=args.wheel_distance,
            manual_measured_m=args.manual_distance,
        )
        cal.config.calibration_time = time.strftime('%Y-%m-%d %H:%M:%S')
        cal.save_config()
    elif args.mode == 'rotation':
        await cal.calibrate_rotation_radius(
            target_rotations=args.rotation_count,
            manual_measured_rad=manual_rot,
        )
        cal.config.calibration_time = time.strftime('%Y-%m-%d %H:%M:%S')
        cal.save_config()


if __name__ == '__main__':
    sys.exit(main())
