import asyncio
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


LOCAL_RPLIDARC1 = Path(__file__).resolve().parent / "rplidarc1"
if LOCAL_RPLIDARC1.exists():
    sys.path.insert(0, str(LOCAL_RPLIDARC1))

from rplidarc1 import RPLidar


PORT = os.environ.get("LIDAR_PORT", "COM14")
BAUDRATE = int(os.environ.get("LIDAR_BAUDRATE", "460800"))
RUN_SECONDS = float(os.environ.get("ICP_RUN_SECONDS", "30"))

MIN_QUALITY = int(os.environ.get("ICP_MIN_QUALITY", "20"))
MIN_RANGE_M = float(os.environ.get("ICP_MIN_RANGE_M", "0.50"))
MAX_RANGE_M = float(os.environ.get("ICP_MAX_RANGE_M", "6.0"))
ANGLE_BIN_DEG = float(os.environ.get("ICP_ANGLE_BIN_DEG", "1.0"))
ANGLE_MIN_DEG = float(os.environ.get("ICP_ANGLE_MIN_DEG", "0"))
ANGLE_MAX_DEG = float(os.environ.get("ICP_ANGLE_MAX_DEG", "360"))
VOXEL_SIZE_M = float(os.environ.get("ICP_VOXEL_SIZE_M", "0.04"))
MAX_POINTS = int(os.environ.get("ICP_MAX_POINTS", "650"))

ICP_MAX_ITER = int(os.environ.get("ICP_MAX_ITER", "24"))
ICP_MAX_MATCH_DIST_M = float(os.environ.get("ICP_MAX_MATCH_DIST_M", "0.15"))
ICP_TRIM_RATIO = float(os.environ.get("ICP_TRIM_RATIO", "0.55"))
ICP_MIN_MATCHES = int(os.environ.get("ICP_MIN_MATCHES", "45"))
ICP_CONVERGED_STEP_M = float(os.environ.get("ICP_CONVERGED_STEP_M", "0.001"))
ICP_CONVERGED_STEP_DEG = float(os.environ.get("ICP_CONVERGED_STEP_DEG", "0.08"))
ICP_MAX_ERROR_M = float(os.environ.get("ICP_MAX_ERROR_M", "0.07"))
ICP_MIN_OVERLAP = float(os.environ.get("ICP_MIN_OVERLAP", "0.30"))
ICP_DEADBAND_TRANS_M = float(os.environ.get("ICP_DEADBAND_TRANS_M", "0.025"))
ICP_DEADBAND_ROT_DEG = float(os.environ.get("ICP_DEADBAND_ROT_DEG", "0.70"))
ICP_MAX_DELTA_TRANS_M = float(os.environ.get("ICP_MAX_DELTA_TRANS_M", "0.05"))
ICP_MAX_DELTA_ROT_DEG = float(os.environ.get("ICP_MAX_DELTA_ROT_DEG", "2.0"))
ICP_MODE = os.environ.get("ICP_MODE", "previous").lower()
if ICP_MODE not in {"keyframe", "previous"}:
    ICP_MODE = "keyframe"
ICP_KEYFRAME_TRANS_M = float(os.environ.get("ICP_KEYFRAME_TRANS_M", "0.70"))
ICP_KEYFRAME_ROT_DEG = float(os.environ.get("ICP_KEYFRAME_ROT_DEG", "18"))

# Set this if the LiDAR 0-degree direction is not robot-forward.
LIDAR_YAW_OFFSET_DEG = float(os.environ.get("LIDAR_YAW_OFFSET_DEG", "0"))


def wrap_angle_rad(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def rotation_matrix(theta):
    ct = math.cos(theta)
    st = math.sin(theta)
    return np.array([[ct, -st], [st, ct]], dtype=np.float64)


def transform_points(points, pose_delta):
    if points.size == 0:
        return points
    dx, dy, theta = pose_delta
    return points @ rotation_matrix(theta).T + np.array([dx, dy], dtype=np.float64)


def compose_delta(delta_a, delta_b):
    ax, ay, ath = delta_a
    bx, by, bth = delta_b
    translated = rotation_matrix(ath) @ np.array([bx, by], dtype=np.float64)
    return (
        ax + float(translated[0]),
        ay + float(translated[1]),
        wrap_angle_rad(ath + bth),
    )


def apply_robot_delta(pose, delta_prev_frame):
    x, y, theta = pose
    dx, dy, dtheta = delta_prev_frame
    c = math.cos(theta)
    s = math.sin(theta)
    return (
        x + c * dx - s * dy,
        y + s * dx + c * dy,
        wrap_angle_rad(theta + dtheta),
    )


def relative_delta(from_pose, to_pose):
    fx, fy, ftheta = from_pose
    tx, ty, ttheta = to_pose
    dx_world = tx - fx
    dy_world = ty - fy
    c = math.cos(ftheta)
    s = math.sin(ftheta)
    return (
        c * dx_world + s * dy_world,
        -s * dx_world + c * dy_world,
        wrap_angle_rad(ttheta - ftheta),
    )


def estimate_rigid_transform(source, target):
    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid

    h_mat = src_centered.T @ tgt_centered
    u_mat, _, vt_mat = np.linalg.svd(h_mat)
    rot = vt_mat.T @ u_mat.T
    if np.linalg.det(rot) < 0:
        vt_mat[1, :] *= -1.0
        rot = vt_mat.T @ u_mat.T

    theta = math.atan2(rot[1, 0], rot[0, 0])
    trans = tgt_centroid - rot @ src_centroid
    return float(trans[0]), float(trans[1]), theta


def voxel_downsample(points, voxel_size):
    if points.shape[0] == 0 or voxel_size <= 0:
        return points
    grid = np.floor(points / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(grid, axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def limit_points(points, max_points):
    if points.shape[0] <= max_points:
        return points
    step = max(1, points.shape[0] // max_points)
    return points[::step][:max_points]


def nearest_neighbors(source, target):
    best_idx = np.empty(source.shape[0], dtype=np.int32)
    best_dist = np.empty(source.shape[0], dtype=np.float64)
    chunk_size = 256

    for start in range(0, source.shape[0], chunk_size):
        chunk = source[start : start + chunk_size]
        diff = chunk[:, None, :] - target[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        idx = np.argmin(dist2, axis=1)
        best_idx[start : start + chunk.shape[0]] = idx
        best_dist[start : start + chunk.shape[0]] = np.sqrt(dist2[np.arange(chunk.shape[0]), idx])

    return best_idx, best_dist


def trim_correspondences(distances, keep_mask):
    keep_indices = np.flatnonzero(keep_mask)
    if keep_indices.size == 0:
        return keep_mask

    trim_count = max(ICP_MIN_MATCHES, int(keep_indices.size * ICP_TRIM_RATIO))
    trim_count = min(trim_count, keep_indices.size)
    best_local = np.argpartition(distances[keep_indices], trim_count - 1)[:trim_count]

    trimmed = np.zeros_like(keep_mask, dtype=bool)
    trimmed[keep_indices[best_local]] = True
    return trimmed


def apply_deadband(delta):
    dx, dy, dtheta = delta
    if (
        math.hypot(dx, dy) < ICP_DEADBAND_TRANS_M
        and abs(math.degrees(dtheta)) < ICP_DEADBAND_ROT_DEG
    ):
        return 0.0, 0.0, 0.0
    return delta


def delta_is_plausible(delta):
    dx, dy, dtheta = delta
    return (
        math.hypot(dx, dy) <= ICP_MAX_DELTA_TRANS_M
        and abs(math.degrees(dtheta)) <= ICP_MAX_DELTA_ROT_DEG
    )


def icp_2d(source_points, target_points):
    if source_points.shape[0] < ICP_MIN_MATCHES or target_points.shape[0] < ICP_MIN_MATCHES:
        return None

    total_delta = (0.0, 0.0, 0.0)
    transformed = source_points.copy()

    for _ in range(ICP_MAX_ITER):
        indices, distances = nearest_neighbors(transformed, target_points)
        keep = distances < ICP_MAX_MATCH_DIST_M
        keep = trim_correspondences(distances, keep)
        if int(np.count_nonzero(keep)) < ICP_MIN_MATCHES:
            return None

        src_matched = transformed[keep]
        tgt_matched = target_points[indices[keep]]
        step_delta = estimate_rigid_transform(src_matched, tgt_matched)

        transformed = transform_points(transformed, step_delta)
        total_delta = compose_delta(step_delta, total_delta)

        step_dist = math.hypot(step_delta[0], step_delta[1])
        step_rot_deg = abs(math.degrees(step_delta[2]))
        if step_dist < ICP_CONVERGED_STEP_M and step_rot_deg < ICP_CONVERGED_STEP_DEG:
            break

    _, final_distances = nearest_neighbors(transform_points(source_points, total_delta), target_points)
    final_keep = final_distances < ICP_MAX_MATCH_DIST_M
    final_keep = trim_correspondences(final_distances, final_keep)
    mean_error = float(final_distances[final_keep].mean()) if np.any(final_keep) else float("inf")
    matches = int(np.count_nonzero(final_keep))
    overlap = matches / max(1, min(source_points.shape[0], target_points.shape[0]))
    if matches < ICP_MIN_MATCHES or mean_error > ICP_MAX_ERROR_M or overlap < ICP_MIN_OVERLAP:
        return None
    return total_delta, mean_error, matches, overlap


def angle_allowed(angle_deg):
    angle = angle_deg % 360.0
    amin = ANGLE_MIN_DEG % 360.0
    amax = ANGLE_MAX_DEG % 360.0
    if abs((ANGLE_MAX_DEG - ANGLE_MIN_DEG) % 360.0) < 1e-9:
        return True
    if amin <= amax:
        return amin <= angle <= amax
    return angle >= amin or angle <= amax


def scan_to_points(scan):
    binned = {}

    for item in scan:
        quality = item.get("q", 0)
        distance_mm = item.get("d_mm")
        if quality < MIN_QUALITY or distance_mm is None:
            continue

        raw_angle = float(item["a_deg"]) % 360.0
        if not angle_allowed(raw_angle):
            continue

        distance_m = float(distance_mm) / 1000.0
        if distance_m < MIN_RANGE_M or distance_m > MAX_RANGE_M:
            continue

        bin_key = round(raw_angle / ANGLE_BIN_DEG) * ANGLE_BIN_DEG if ANGLE_BIN_DEG > 0 else raw_angle
        bin_key %= 360.0
        binned.setdefault(bin_key, []).append(distance_m)

    if not binned:
        return np.empty((0, 2), dtype=np.float64)

    points = []
    yaw_offset = math.radians(LIDAR_YAW_OFFSET_DEG)
    for angle_deg, distances in binned.items():
        distance_m = float(np.median(distances))
        angle = math.radians(angle_deg) + yaw_offset
        points.append((distance_m * math.cos(angle), distance_m * math.sin(angle)))

    arr = np.array(points, dtype=np.float64)
    arr = voxel_downsample(arr, VOXEL_SIZE_M)
    return limit_points(arr, MAX_POINTS)


async def process_scans(lidar):
    reference_points = None
    keyframe_pose = (0.0, 0.0, 0.0)
    current_scan = []
    prev_angle = None
    pose = (0.0, 0.0, 0.0)
    prev_pose = pose
    scan_count = 0
    accepted_count = 0
    rejected_count = 0
    zeroed_count = 0
    start_time = time.time()
    deadline = start_time + RUN_SECONDS if RUN_SECONDS > 0 else None

    while deadline is None or time.time() < deadline:
        try:
            point = await asyncio.wait_for(lidar.output_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue

        angle = float(point["a_deg"])
        wrapped = prev_angle is not None and prev_angle > 300.0 and angle < 60.0
        prev_angle = angle

        if wrapped and len(current_scan) > 60:
            scan_count += 1
            points = scan_to_points(current_scan)
            current_scan = []

            if points.shape[0] < ICP_MIN_MATCHES:
                print(f"scan={scan_count:04d} skipped: only {points.shape[0]} valid points")
                continue

            if reference_points is None:
                reference_points = points
                keyframe_pose = pose
                print(f"scan={scan_count:04d} initialized with {points.shape[0]} valid points")
                continue

            result = icp_2d(points, reference_points)
            if result is None:
                print(f"scan={scan_count:04d} ICP failed, valid_points={points.shape[0]}")
                if ICP_MODE == "previous":
                    reference_points = points
                continue

            delta, error, matches, overlap = result
            if not delta_is_plausible(delta):
                rejected_count += 1
                print(
                    "scan={scan:04d} rejected: implausible delta "
                    "dx={dx:+.3f} dy={dy:+.3f} dtheta={dtheta:+.2f} deg "
                    "err={err:.3f} overlap={overlap:.2f}".format(
                        scan=scan_count,
                        dx=delta[0],
                        dy=delta[1],
                        dtheta=math.degrees(delta[2]),
                        err=error,
                        overlap=overlap,
                    )
                )
                if ICP_MODE == "previous":
                    reference_points = points
                continue

            delta = apply_deadband(delta)
            if delta == (0.0, 0.0, 0.0):
                zeroed_count += 1

            if ICP_MODE == "previous":
                pose = apply_robot_delta(pose, delta)
                reference_points = points
                keyframe_pose = pose
            else:
                pose = apply_robot_delta(keyframe_pose, delta)

            step_delta = relative_delta(prev_pose, pose)
            prev_pose = pose
            accepted_count += 1

            print(
                "scan={scan:04d} x={x:+.3f} m y={y:+.3f} m theta={theta:+.1f} deg "
                "dx={dx:+.3f} dy={dy:+.3f} dtheta={dtheta:+.2f} deg "
                "err={err:.3f} m overlap={overlap:.2f} matches={matches} pts={pts}".format(
                    scan=scan_count,
                    x=pose[0],
                    y=pose[1],
                    theta=math.degrees(pose[2]),
                    dx=step_delta[0],
                    dy=step_delta[1],
                    dtheta=math.degrees(step_delta[2]),
                    err=error,
                    overlap=overlap,
                    matches=matches,
                    pts=points.shape[0],
                )
            )

            if ICP_MODE != "previous":
                keyframe_dist = math.hypot(delta[0], delta[1])
                keyframe_rot = abs(math.degrees(delta[2]))
                if (
                    keyframe_dist >= ICP_KEYFRAME_TRANS_M
                    or keyframe_rot >= ICP_KEYFRAME_ROT_DEG
                ):
                    reference_points = points
                    keyframe_pose = pose
                    print(
                        "  keyframe reset at x={x:+.3f} m y={y:+.3f} m theta={theta:+.1f} deg".format(
                            x=pose[0],
                            y=pose[1],
                            theta=math.degrees(pose[2]),
                        )
                    )

        current_scan.append(point)

    lidar.stop_event.set()
    elapsed = max(0.001, time.time() - start_time)
    print(
        "done: scans={scans}, accepted={accepted}, zeroed={zeroed}, rejected={rejected}, elapsed={elapsed:.1f}s, "
        "final_pose=(x={x:+.3f} m, y={y:+.3f} m, theta={theta:+.1f} deg)".format(
            scans=scan_count,
            accepted=accepted_count,
            zeroed=zeroed_count,
            rejected=rejected_count,
            elapsed=elapsed,
            x=pose[0],
            y=pose[1],
            theta=math.degrees(pose[2]),
        )
    )


async def main():
    print(f"Opening RPLidar on {PORT} at {BAUDRATE} baud")
    print(
        "Filters: q>={q}, range={rmin:.2f}..{rmax:.2f} m, angle={amin:.0f}..{amax:.0f} deg, "
        "bin={abin:.2f} deg, voxel={voxel:.2f} m".format(
            q=MIN_QUALITY,
            rmin=MIN_RANGE_M,
            rmax=MAX_RANGE_M,
            amin=ANGLE_MIN_DEG,
            amax=ANGLE_MAX_DEG,
            abin=ANGLE_BIN_DEG,
            voxel=VOXEL_SIZE_M,
        )
    )
    print(
        "ICP: mode={mode}, match_dist<={match:.2f} m, trim={trim:.2f}, "
        "max_error={err:.2f} m, min_overlap={overlap:.2f}, "
        "deadband={db_m:.3f} m/{db_d:.2f} deg, max_delta={max_m:.3f} m/{max_d:.1f} deg".format(
            mode=ICP_MODE,
            match=ICP_MAX_MATCH_DIST_M,
            trim=ICP_TRIM_RATIO,
            err=ICP_MAX_ERROR_M,
            overlap=ICP_MIN_OVERLAP,
            db_m=ICP_DEADBAND_TRANS_M,
            db_d=ICP_DEADBAND_ROT_DEG,
            max_m=ICP_MAX_DELTA_TRANS_M,
            max_d=ICP_MAX_DELTA_ROT_DEG,
        )
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
