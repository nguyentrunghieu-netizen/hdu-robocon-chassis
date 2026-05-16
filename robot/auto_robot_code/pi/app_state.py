"""
app_state.py
============
Trang thai dung chung cho toan bo ung dung (thread-safe).
"""
import threading

from config import WAYPOINTS_FILE, TARGET_BBOX_H


class SharedAppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.motor_enabled = False
        self.camera_tracking_enabled = True
        self.serial_mgr = None
        self.lidar_mgr = None
        self.config_path = 'robot_config.json'
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
        self.calibration_active = False
        self.calibration_mode = ''
        # A* path state
        self.nav_path_waypoints = []
        self.nav_path_idx = 0
        self.nav_path_target_id = None
        self.nav_last_plan_t = 0.0
        self.stats = {
            'status': 'SEARCHING',
            'ball_detected': False,
            'camera_tracking_enabled': True,
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
            'serial_port': '',
            'robot_x': 0.0,
            'robot_y': 0.0,
            'robot_theta_deg': 0.0,
            'imu_ok': False,
            'lidar_enabled': False,
            'lidar_status': 'DISABLED',
            'lidar_error_text': '',
            'lidar_port': '',
            'lidar_baudrate': 0,
            'lidar_x': 0.0,
            'lidar_y': 0.0,
            'lidar_theta_deg': 0.0,
            'lidar_icp_x': 0.0,
            'lidar_icp_y': 0.0,
            'lidar_icp_theta_deg': 0.0,
            'lidar_icp_last_update': 0.0,
            'heading_source': 'pose_ekf',
            'heading_bias_deg': 0.0,
            'heading_std_deg': 0.0,
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
            'calibration_active': False,
            'calibration_mode': '',
            'hist_err_x': [],
            'hist_err_dist': [],
            'hist_omega': [],
            'hist_vx': [],
        }
