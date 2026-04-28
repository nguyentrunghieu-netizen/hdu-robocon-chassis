"""
LIDAR + ICP 2D - bản nâng cấp theo phong cách KISS-ICP.

3 cải tiến quan trọng nhất so với bản trước:

1. MOTION DESKEWING (KISS-ICP step 1):
   - Bù chuyển động của LIDAR trong lúc quét 1 vòng (~100ms).
   - Dùng vận tốc ước lượng + giả thiết vận tốc không đổi.
   - Mỗi điểm được "kéo" về thời điểm tham chiếu (giữa scan).
   - Đây là cải tiến lớn nhất khi LIDAR di chuyển/xoay trong lúc scan.

2. ADAPTIVE CORRESPONDENCE THRESHOLD (KISS-ICP step 3):
   - Ngưỡng matching tự nới rộng khi di chuyển nhanh.
   - Tự thắt chặt khi đứng yên.
   - Tránh tình trạng "1 ngưỡng cứng không phù hợp với mọi tốc độ".

3. SLIDING WINDOW LOCAL MAP (thay thế keyframe đơn lẻ):
   - Tích lũy điểm từ N scan gần nhất vào 1 voxel map.
   - ICP so với map này thay vì so với 1 scan duy nhất.
   - Map có nhiều thông tin hơn => matching ổn định hơn.
   - Tự nhiên xử lý vấn đề "mất track khi đi xa keyframe".

Yêu cầu:
    pip install numpy scipy
"""

import asyncio
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


LOCAL_RPLIDARC1 = Path(__file__).resolve().parent / "rplidarc1"
if LOCAL_RPLIDARC1.exists():
    sys.path.insert(0, str(LOCAL_RPLIDARC1))

from rplidarc1 import RPLidar


# ============================================================
# Cấu hình
# ============================================================
PORT = os.environ.get("LIDAR_PORT", "COM14")
BAUDRATE = int(os.environ.get("LIDAR_BAUDRATE", "460800"))
RUN_SECONDS = float(os.environ.get("ICP_RUN_SECONDS", "30"))

MIN_QUALITY = int(os.environ.get("ICP_MIN_QUALITY", "20"))
MIN_RANGE_M = float(os.environ.get("ICP_MIN_RANGE_M", "0.20"))
MAX_RANGE_M = float(os.environ.get("ICP_MAX_RANGE_M", "8.0"))
VOXEL_SIZE_M = float(os.environ.get("ICP_VOXEL_SIZE_M", "0.05"))
MAX_POINTS = int(os.environ.get("ICP_MAX_POINTS", "800"))

# ICP cơ bản
ICP_MAX_ITER = int(os.environ.get("ICP_MAX_ITER", "30"))
ICP_TRIM_RATIO = float(os.environ.get("ICP_TRIM_RATIO", "0.80"))
ICP_MIN_MATCHES = int(os.environ.get("ICP_MIN_MATCHES", "30"))
ICP_CONVERGED_STEP_M = float(os.environ.get("ICP_CONVERGED_STEP_M", "0.0005"))
ICP_CONVERGED_STEP_DEG = float(os.environ.get("ICP_CONVERGED_STEP_DEG", "0.05"))

# ADAPTIVE THRESHOLD - thay cho ICP_MAX_MATCH_DIST_M cứng
# Công thức: threshold = base + scale * |velocity|
ADAPTIVE_THRESH_BASE_M = float(os.environ.get("ICP_THRESH_BASE_M", "0.10"))
ADAPTIVE_THRESH_SCALE = float(os.environ.get("ICP_THRESH_SCALE", "2.5"))
ADAPTIVE_THRESH_MAX_M = float(os.environ.get("ICP_THRESH_MAX_M", "1.0"))

# Quality gates (lỏng hơn vì có local map giúp accuracy)
ICP_MAX_ERROR_M = float(os.environ.get("ICP_MAX_ERROR_M", "0.15"))
ICP_MIN_OVERLAP = float(os.environ.get("ICP_MIN_OVERLAP", "0.20"))

# Plausibility - dùng giới hạn vận tốc thực tế (m/s, deg/s) thay vì /scan
MAX_VELOCITY_MS = float(os.environ.get("ICP_MAX_VELOCITY_MS", "3.0"))
MAX_ANGULAR_DPS = float(os.environ.get("ICP_MAX_ANGULAR_DPS", "180"))
ASSUMED_SCAN_RATE_HZ = float(os.environ.get("ICP_SCAN_RATE_HZ", "10"))

# Local map
LOCAL_MAP_NUM_SCANS = int(os.environ.get("ICP_MAP_NUM_SCANS", "8"))
LOCAL_MAP_VOXEL_M = float(os.environ.get("ICP_MAP_VOXEL_M", "0.05"))
LOCAL_MAP_RANGE_M = float(os.environ.get("ICP_MAP_RANGE_M", "12.0"))

# Recovery
MAX_CONSECUTIVE_FAILURES = int(os.environ.get("ICP_MAX_FAIL_STREAK", "5"))

LIDAR_YAW_OFFSET_DEG = float(os.environ.get("LIDAR_YAW_OFFSET_DEG", "0"))


# ============================================================
# Hàm hình học
# ============================================================
def wrap_angle_rad(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rotation_matrix(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def transform_points(points, pose_delta):
    if points.size == 0:
        return points
    dx, dy, theta = pose_delta
    return points @ rotation_matrix(theta).T + np.array([dx, dy], dtype=np.float64)


def compose_delta(a, b):
    """Áp b trước, rồi áp a. Kết quả tương đương a o b."""
    ax, ay, ath = a
    bx, by, bth = b
    R = rotation_matrix(ath)
    t = R @ np.array([bx, by], dtype=np.float64)
    return ax + float(t[0]), ay + float(t[1]), wrap_angle_rad(ath + bth)


def apply_robot_delta(pose, delta_local):
    x, y, theta = pose
    dx, dy, dth = delta_local
    c, s = math.cos(theta), math.sin(theta)
    return x + c*dx - s*dy, y + s*dx + c*dy, wrap_angle_rad(theta + dth)


def relative_delta(from_pose, to_pose):
    fx, fy, fth = from_pose
    tx, ty, tth = to_pose
    dxw, dyw = tx - fx, ty - fy
    c, s = math.cos(fth), math.sin(fth)
    return c*dxw + s*dyw, -s*dxw + c*dyw, wrap_angle_rad(tth - fth)


def estimate_rigid_transform(source, target):
    sc = source.mean(axis=0)
    tc = target.mean(axis=0)
    H = (source - sc).T @ (target - tc)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1.0
        R = Vt.T @ U.T
    theta = math.atan2(R[1, 0], R[0, 0])
    t = tc - R @ sc
    return float(t[0]), float(t[1]), theta


def voxel_downsample(points, voxel_size):
    if points.shape[0] == 0 or voxel_size <= 0:
        return points
    grid = np.floor(points / voxel_size).astype(np.int32)
    _, idx = np.unique(grid, axis=0, return_index=True)
    return points[np.sort(idx)]


def limit_points(points, max_points):
    if points.shape[0] <= max_points:
        return points
    step = max(1, points.shape[0] // max_points)
    return points[::step][:max_points]


# ============================================================
# CẢI TIẾN #1: MOTION DESKEWING
# ============================================================
def deskew_scan(points_xyt, velocity_per_scan):
    """Bù motion distortion trong 1 scan.

    points_xyt: Nx3 array (x, y, t) với t in [0, 1] là thời gian chuẩn hoá trong scan.
    velocity_per_scan: (vx, vy, w) - chuyển động ước lượng của LIDAR trong 1 scan đầy đủ.
                       Đơn vị: m và rad PER SCAN.

    Trả về Nx2 - tất cả điểm như thể được đo cùng tại t_ref = 0.5 (giữa scan).
    """
    if points_xyt.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    vx, vy, w = velocity_per_scan
    if vx == 0.0 and vy == 0.0 and w == 0.0:
        return points_xyt[:, :2].copy()

    t_ref = 0.5  # tham chiếu giữa scan -> giảm sai số tối đa
    dt = points_xyt[:, 2] - t_ref     # mỗi điểm cách t_ref bao nhiêu

    # Điểm đo tại thời điểm t cần đưa về frame tại t_ref bằng cách áp transform
    # tương ứng với chuyển động (t - t_ref) * velocity (xấp xỉ tuyến tính cho motion nhỏ)
    angles = dt * w
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    x = points_xyt[:, 0]
    y = points_xyt[:, 1]

    x_new = cos_a * x - sin_a * y + dt * vx
    y_new = sin_a * x + cos_a * y + dt * vy
    return np.column_stack([x_new, y_new])


def scan_to_points(scan, velocity_per_scan=(0., 0., 0.)):
    """Convert raw scan -> Nx2 deskewed points trong frame LIDAR (mid-scan)."""
    raw = []
    n_total = max(1, len(scan) - 1)
    yaw_offset = math.radians(LIDAR_YAW_OFFSET_DEG)

    for i, item in enumerate(scan):
        q = item.get("q", 0)
        d_mm = item.get("d_mm")
        if q < MIN_QUALITY or d_mm is None:
            continue
        d = float(d_mm) / 1000.0
        if not (MIN_RANGE_M <= d <= MAX_RANGE_M):
            continue
        a = math.radians(float(item["a_deg"]) % 360.0) + yaw_offset
        t = i / n_total
        raw.append((d * math.cos(a), d * math.sin(a), t))

    if not raw:
        return np.empty((0, 2), dtype=np.float64)

    pts_xyt = np.array(raw, dtype=np.float64)
    pts = deskew_scan(pts_xyt, velocity_per_scan)
    pts = voxel_downsample(pts, VOXEL_SIZE_M)
    return limit_points(pts, MAX_POINTS)


# ============================================================
# CẢI TIẾN #2: ADAPTIVE CORRESPONDENCE THRESHOLD
# ============================================================
def adaptive_threshold(velocity_per_scan):
    """Ngưỡng matching tự điều chỉnh theo chuyển động ước lượng.

    Khi đứng yên: ngưỡng = base (sát nhau, ít noise)
    Khi di chuyển nhanh: ngưỡng nới rộng (tha thứ cho ước lượng vận tốc sai)
    """
    vx, vy, w = velocity_per_scan
    motion = math.hypot(vx, vy) + 0.15 * abs(w)  # 0.15m/rad - quy đổi rough rotation -> linear
    th = ADAPTIVE_THRESH_BASE_M + ADAPTIVE_THRESH_SCALE * motion
    return min(th, ADAPTIVE_THRESH_MAX_M)


# ============================================================
# CẢI TIẾN #3: SLIDING WINDOW LOCAL MAP
# ============================================================
class LocalMap:
    """Sliding window: giữ N scan gần nhất (đã transform về world frame).

    Map này được dùng làm target cho ICP - nhiều thông tin hơn 1 scan đơn,
    giảm noise, không bị "mất track khi đi xa keyframe" như single-keyframe.
    """
    def __init__(self, max_scans, voxel_size, max_range):
        self.max_scans = max_scans
        self.voxel_size = voxel_size
        self.max_range = max_range
        self.scans_world = deque(maxlen=max_scans)  # tự động bỏ scan cũ

    def add_scan(self, points_local, pose):
        """Thêm 1 scan vào map. points_local trong frame LIDAR, pose là pose toàn cục."""
        pts_world = transform_points(points_local, pose)
        self.scans_world.append(pts_world)

    def query_near(self, center_pose):
        """Lấy điểm trong map nằm gần center_pose (để ICP nhanh hơn)."""
        if not self.scans_world:
            return np.empty((0, 2), dtype=np.float64)

        all_pts = np.vstack(self.scans_world)
        cx, cy, _ = center_pose
        dists = np.linalg.norm(all_pts - np.array([cx, cy]), axis=1)
        nearby = all_pts[dists < self.max_range]
        # Voxel downsample để loại trùng lặp giữa các scan
        return voxel_downsample(nearby, self.voxel_size)


# ============================================================
# ICP với KDTree, init guess, adaptive threshold
# ============================================================
def trim_correspondences(distances, keep_mask):
    keep_idx = np.flatnonzero(keep_mask)
    if keep_idx.size == 0:
        return keep_mask
    n_keep = max(ICP_MIN_MATCHES, int(keep_idx.size * ICP_TRIM_RATIO))
    n_keep = min(n_keep, keep_idx.size)
    best = np.argpartition(distances[keep_idx], n_keep - 1)[:n_keep]
    out = np.zeros_like(keep_mask, dtype=bool)
    out[keep_idx[best]] = True
    return out


def icp_2d(source, target, init_delta=None, max_match_dist=0.20):
    if source.shape[0] < ICP_MIN_MATCHES or target.shape[0] < ICP_MIN_MATCHES:
        return None, "too_few_points"

    tree = cKDTree(target)
    total = init_delta if init_delta is not None else (0., 0., 0.)
    transformed = transform_points(source, total)

    for _ in range(ICP_MAX_ITER):
        dists, idx = tree.query(transformed, k=1)
        keep = dists < max_match_dist
        keep = trim_correspondences(dists, keep)
        if int(np.count_nonzero(keep)) < ICP_MIN_MATCHES:
            return None, f"matches={int(np.count_nonzero(keep))}"

        s = transformed[keep]
        t = target[idx[keep]]
        step = estimate_rigid_transform(s, t)

        transformed = transform_points(transformed, step)
        total = compose_delta(step, total)

        if (math.hypot(step[0], step[1]) < ICP_CONVERGED_STEP_M
                and abs(math.degrees(step[2])) < ICP_CONVERGED_STEP_DEG):
            break

    final_d, _ = tree.query(transform_points(source, total), k=1)
    final_keep = final_d < max_match_dist
    final_keep = trim_correspondences(final_d, final_keep)
    matches = int(np.count_nonzero(final_keep))
    err = float(final_d[final_keep].mean()) if matches > 0 else float("inf")
    overlap = matches / max(1, min(source.shape[0], target.shape[0]))

    if matches < ICP_MIN_MATCHES:
        return None, f"matches={matches}"
    if err > ICP_MAX_ERROR_M:
        return None, f"err={err:.3f}"
    if overlap < ICP_MIN_OVERLAP:
        return None, f"overlap={overlap:.2f}"
    return (total, err, matches, overlap), None


# ============================================================
# Plausibility theo tốc độ thực tế
# ============================================================
def velocity_is_plausible(delta_per_scan):
    dx, dy, dth = delta_per_scan
    speed_ms = math.hypot(dx, dy) * ASSUMED_SCAN_RATE_HZ
    angular_dps = abs(math.degrees(dth)) * ASSUMED_SCAN_RATE_HZ
    return speed_ms <= MAX_VELOCITY_MS and angular_dps <= MAX_ANGULAR_DPS


# ============================================================
# Vòng lặp xử lý chính
# ============================================================
async def process_scans(lidar):
    local_map = LocalMap(LOCAL_MAP_NUM_SCANS, LOCAL_MAP_VOXEL_M, LOCAL_MAP_RANGE_M)
    pose = (0., 0., 0.)
    prev_pose = pose

    # Vận tốc ước lượng (m, rad PER SCAN) - dùng cho cả deskewing và adaptive threshold
    velocity_per_scan = (0., 0., 0.)

    current_scan = []
    prev_angle = None
    consecutive_failures = 0

    scan_count = 0
    accepted = rejected = hard_resets = 0
    start_time = time.time()
    deadline = start_time + RUN_SECONDS if RUN_SECONDS > 0 else None

    while deadline is None or time.time() < deadline:
        try:
            point = await asyncio.wait_for(lidar.output_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue

        a = float(point["a_deg"])
        wrapped = prev_angle is not None and prev_angle > 300.0 and a < 60.0
        prev_angle = a

        if not (wrapped and len(current_scan) > 60):
            current_scan.append(point)
            continue

        scan_count += 1
        scan_buf = current_scan
        current_scan = [point]

        # === DESKEWING TRONG scan_to_points ===
        points = scan_to_points(scan_buf, velocity_per_scan)

        if points.shape[0] < ICP_MIN_MATCHES:
            print(f"scan={scan_count:04d} skipped: {points.shape[0]} pts")
            continue

        # === KHỞI TẠO LOCAL MAP ===
        if len(local_map.scans_world) == 0:
            local_map.add_scan(points, pose)
            print(f"scan={scan_count:04d} init local map ({points.shape[0]} pts)")
            continue

        # === ADAPTIVE THRESHOLD ===
        match_thresh = adaptive_threshold(velocity_per_scan)

        # === ICP scan-to-MAP (không phải scan-to-scan) ===
        target_map = local_map.query_near(pose)
        if target_map.shape[0] < ICP_MIN_MATCHES:
            local_map.add_scan(points, pose)
            print(f"scan={scan_count:04d} map too small, just add")
            continue

        # Init guess: dùng vận tốc trước (constant velocity)
        # Initial pose của scan = pose trước + velocity
        predicted_pose = apply_robot_delta(pose, velocity_per_scan)

        # ICP cần biết transform từ source (current scan local frame)
        # về target (map - world frame). Init guess = predicted_pose.
        result, fail_reason = icp_2d(
            points, target_map,
            init_delta=predicted_pose,
            max_match_dist=match_thresh,
        )

        if result is None:
            consecutive_failures += 1
            rejected += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                # Hard reset: clear map, dùng scan hiện tại làm map mới
                local_map.scans_world.clear()
                local_map.add_scan(points, pose)
                consecutive_failures = 0
                hard_resets += 1
                velocity_per_scan = (0., 0., 0.)
                print(f"scan={scan_count:04d} >>> HARD RESET ({fail_reason}, "
                      f"thresh={match_thresh:.2f}m)")
            else:
                print(f"scan={scan_count:04d} ICP failed #{consecutive_failures}: "
                      f"{fail_reason}, thresh={match_thresh:.2f}m, pts={points.shape[0]}")
            continue

        consecutive_failures = 0
        new_pose, error, matches, overlap = result

        # === Plausibility theo vận tốc thực tế ===
        step = relative_delta(pose, new_pose)
        if not velocity_is_plausible(step):
            speed = math.hypot(step[0], step[1]) * ASSUMED_SCAN_RATE_HZ
            ang = abs(math.degrees(step[2])) * ASSUMED_SCAN_RATE_HZ
            rejected += 1
            print(f"scan={scan_count:04d} rejected: implausible "
                  f"v={speed:.2f}m/s w={ang:.1f}°/s")
            continue

        # === Cập nhật pose, vận tốc, map ===
        pose = new_pose
        velocity_per_scan = step  # vận tốc per-scan = step delta
        prev_pose = pose
        local_map.add_scan(points, pose)
        accepted += 1

        speed_ms = math.hypot(step[0], step[1]) * ASSUMED_SCAN_RATE_HZ
        ang_dps = math.degrees(step[2]) * ASSUMED_SCAN_RATE_HZ

        print(
            f"scan={scan_count:04d} "
            f"x={pose[0]:+.3f} y={pose[1]:+.3f} θ={math.degrees(pose[2]):+6.1f}° | "
            f"v={speed_ms:+.2f}m/s w={ang_dps:+5.1f}°/s | "
            f"err={error:.3f} ovl={overlap:.2f} thresh={match_thresh:.2f} "
            f"map={sum(s.shape[0] for s in local_map.scans_world)}pts"
        )

    lidar.stop_event.set()
    elapsed = max(0.001, time.time() - start_time)
    print(
        f"\nDONE: scans={scan_count} accepted={accepted} rejected={rejected} "
        f"hard_resets={hard_resets} ({elapsed:.1f}s)\n"
        f"Final pose: x={pose[0]:+.3f}m y={pose[1]:+.3f}m θ={math.degrees(pose[2]):+.1f}°"
    )


async def main():
    print(f"Opening RPLidar on {PORT} @ {BAUDRATE}")
    print(f"Pipeline: KISS-ICP-style 2D (deskew + adaptive thresh + local map)")
    print(
        f"Local map: {LOCAL_MAP_NUM_SCANS} scans, voxel={LOCAL_MAP_VOXEL_M:.2f}m, "
        f"range={LOCAL_MAP_RANGE_M:.1f}m"
    )
    print(
        f"Adaptive thresh: base={ADAPTIVE_THRESH_BASE_M:.2f}m, "
        f"scale={ADAPTIVE_THRESH_SCALE:.1f}, max={ADAPTIVE_THRESH_MAX_M:.2f}m"
    )
    print(
        f"Plausibility: v_max={MAX_VELOCITY_MS:.1f}m/s, "
        f"w_max={MAX_ANGULAR_DPS:.0f}°/s @ {ASSUMED_SCAN_RATE_HZ:.0f}Hz"
    )

    lidar = RPLidar(PORT, BAUDRATE)
    try:
        await asyncio.gather(
            lidar.simple_scan(make_return_dict=False),
            process_scans(lidar),
        )
    finally:
        time.sleep(0.2)
        lidar.reset()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("stopped by user")