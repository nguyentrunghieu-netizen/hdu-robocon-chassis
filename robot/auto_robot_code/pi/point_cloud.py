"""
point_cloud.py
==============
Xu ly point cloud LiDAR:
  - PointCloudFilter: loc, tich luy, decay theo thoi gian
  - PersistentMap: ban do ben vung voi raycast occupancy
"""
import json
import math
import os
import time

import numpy as np

from config import (
    PC_VOXEL_SIZE_M, PC_SOR_K, PC_SOR_STD_RATIO, PC_MAX_POINTS,
    PC_DECAY_PER_SEC, PC_MIN_ALPHA, PC_RANGE_MIN_M, PC_RANGE_MAX_M,
    MAP_VOXEL_SIZE_M, MAP_MAX_POINTS, MAP_MIN_OBS_TO_KEEP,
    MAP_MIN_OBS_TO_DISPLAY, MAP_OBS_MAX,
    MAP_RAYCAST_FREE_DECREMENT, MAP_RAYCAST_HIT_INCREMENT,
    MAP_RAYCAST_MAX_RANGE_M, MAP_RAYCAST_DOWNSAMPLE,
)


class PointCloudFilter:
    """
    Bo loc point cloud hien dai:
    1) Range gating
    2) Voxel-grid downsampling
    3) Statistical Outlier Removal (SOR)
    4) Temporal accumulation + decay
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
        self._cloud = np.zeros((0, 3), dtype=np.float32)
        self._last_update_t = None

    def reset(self):
        self._cloud = np.zeros((0, 3), dtype=np.float32)
        self._last_update_t = None

    def _range_gate(self, pts_local):
        if pts_local.shape[0] == 0:
            return pts_local
        dists = np.linalg.norm(pts_local[:, :2], axis=1)
        mask = (dists >= self.range_min) & (dists <= self.range_max)
        return pts_local[mask]

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

    def update(self, scan_points_local, pose, now_t=None):
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
    Global map ben vung dung occupancy grid voi raycasting.
    Vat di dong tu xoa; tuong tinh tich luy obs_count cao.
    """

    KEY_SCALE = 1000003

    def __init__(self,
                 voxel_size=MAP_VOXEL_SIZE_M,
                 max_points=MAP_MAX_POINTS,
                 min_obs_to_keep=MAP_MIN_OBS_TO_KEEP):
        self.voxel_size = float(voxel_size)
        self.inv_voxel_size = 1.0 / self.voxel_size
        self.max_points = int(max_points)
        self.min_obs_to_keep = float(min_obs_to_keep)
        self._grid = {}
        self.last_pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.last_save_t = 0.0
        self.loaded_from_file = False
        self.source_file = None

    def _xy_to_key(self, ix, iy):
        return int(ix) * self.KEY_SCALE + int(iy)

    def _key_to_xy(self, key):
        ix = key // self.KEY_SCALE
        iy = key - ix * self.KEY_SCALE
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
        if not self._grid:
            return np.zeros((0, 3), dtype=np.float32)
        if min_obs is None:
            min_obs = MAP_MIN_OBS_TO_DISPLAY if only_visible else None
        items = []
        vs = self.voxel_size
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
        arr = self._as_array(only_visible=True)
        if arr.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return arr[:, :2].copy()

    @property
    def size(self):
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

    def _bresenham(self, ix0, iy0, ix1, iy1):
        cells = []
        dx = abs(ix1 - ix0)
        dy = abs(iy1 - iy0)
        sx = 1 if ix0 < ix1 else -1
        sy = 1 if iy0 < iy1 else -1
        err = dx - dy
        x, y = ix0, iy0
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

    def add_scan_with_raycast(self, world_pts, sensor_xy):
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

            d2 = (ix1 - ix0) ** 2 + (iy1 - iy0) ** 2
            if d2 > max_range_idx2:
                continue

            free_cells = self._bresenham(ix0, iy0, ix1, iy1)
            for (cx, cy) in free_cells:
                key = self._xy_to_key(cx, cy)
                cur = self._grid.get(key, 0.0)
                new = cur - MAP_RAYCAST_FREE_DECREMENT
                if new < -3.0:
                    new = -3.0
                self._grid[key] = new

            key_hit = self._xy_to_key(ix1, iy1)
            cur = self._grid.get(key_hit, 0.0)
            new = cur + MAP_RAYCAST_HIT_INCREMENT
            if new > MAP_OBS_MAX:
                new = MAP_OBS_MAX
            self._grid[key_hit] = new

        if len(self._grid) > self.max_points * 1.5:
            self._prune()

    def _prune(self):
        before = len(self._grid)
        self._grid = {k: v for k, v in self._grid.items() if v > -1.5}
        if len(self._grid) > self.max_points:
            items = sorted(self._grid.items(), key=lambda kv: -kv[1])[:self.max_points]
            self._grid = dict(items)
        after = len(self._grid)
        if before != after:
            print(f'[Map] pruned voxels: {before} -> {after}')

    def add_scan_world(self, world_pts, sensor_xy=None):
        if world_pts is None or world_pts.shape[0] == 0:
            return
        if sensor_xy is not None:
            self.add_scan_with_raycast(world_pts, sensor_xy)
            return
        for i in range(world_pts.shape[0]):
            ix, iy = self._world_to_idx(float(world_pts[i, 0]), float(world_pts[i, 1]))
            key = self._xy_to_key(ix, iy)
            cur = self._grid.get(key, 0.0)
            self._grid[key] = min(cur + MAP_RAYCAST_HIT_INCREMENT, MAP_OBS_MAX)

    def save(self, file_path, pose=None, note=''):
        if not self._grid:
            print(f'[Map] save skipped: empty map')
            return False
        try:
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
                obs = np.full(pts.shape[0], float(self.min_obs_to_keep), dtype=np.float32)

            self._grid = {}
            for i in range(pts.shape[0]):
                ix, iy = self._world_to_idx(float(pts[i, 0]), float(pts[i, 1]))
                key = self._xy_to_key(ix, iy)
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
        arr = self._as_array(only_visible=True)
        if arr.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if arr.shape[0] <= max_points:
            return arr
        order = np.argsort(-arr[:, 2])
        return arr[order[:max_points]]
