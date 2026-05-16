"""
planner.py
==========
Lam phang duong di:
  - AStarPlanner: A* tren occupancy grid
  - WaypointStore: luu tru waypoint theo file JSON
"""
import heapq
import json
import math
import os
import re
import threading
import time

import numpy as np

from config import (
    ASTAR_RESOLUTION_M, ASTAR_INFLATE_M,
    WAYPOINTS_FILE, WAYPOINT_LABEL_MAX_LEN,
)


class AStarPlanner:
    """
    A* path planner tren occupancy grid tu PersistentMap.

    Workflow:
      1. Nhan danh sach diem co obstacle trong world frame (tu lidar_map).
      2. Xay dung occupancy grid + tang phong obstacle (Minkowski sum) de robot
         khong di sat vach.
      3. Chay A* tu start_xy den goal_xy tren grid.
      4. Tra ve danh sach sub-waypoints (world frame) da simplify.

    Dung trong navigation loop:
      - Goi plan() khi bat dau navigation hoac can replan.
      - Robot lan luot di den tung sub-waypoint thay vi di thang den goal.
    """

    def __init__(self, resolution=ASTAR_RESOLUTION_M, inflate_m=ASTAR_INFLATE_M):
        self.resolution = float(resolution)
        self.inflate_cells = max(1, round(inflate_m / resolution))

    def plan(self, start_xy, goal_xy, map_points_xy, max_grid_side=350):
        """
        Tra ve list of (x, y) sub-waypoints (world frame) tu start den goal,
        hoac None neu khong tim thay duong.

        map_points_xy: list/ndarray Nx2 cua cac diem obstacle trong world frame.
        """
        sx, sy = float(start_xy[0]), float(start_xy[1])
        gx_w, gy_w = float(goal_xy[0]), float(goal_xy[1])

        if math.hypot(gx_w - sx, gy_w - sy) < self.resolution:
            return [(gx_w, gy_w)]

        if map_points_xy is None or len(map_points_xy) == 0:
            return [(gx_w, gy_w)]

        pts = np.asarray(map_points_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return [(gx_w, gy_w)]
        pts = pts[:, :2]

        res = self.resolution
        margin = max(1.5, self.inflate_cells * res + 0.5)
        min_x = min(float(pts[:, 0].min()), sx, gx_w) - margin
        min_y = min(float(pts[:, 1].min()), sy, gy_w) - margin
        max_x = max(float(pts[:, 0].max()), sx, gx_w) + margin
        max_y = max(float(pts[:, 1].max()), sy, gy_w) + margin

        W = min(max_grid_side, int((max_x - min_x) / res) + 1)
        H = min(max_grid_side, int((max_y - min_y) / res) + 1)
        if W <= 1 or H <= 1:
            return [(gx_w, gy_w)]

        # Build obstacle grid
        grid = np.zeros((H, W), dtype=np.uint8)
        js = np.clip(((pts[:, 0] - min_x) / res).astype(int), 0, W - 1)
        is_ = np.clip(((pts[:, 1] - min_y) / res).astype(int), 0, H - 1)
        grid[is_, js] = 1

        # Inflate obstacles (circular Minkowski sum)
        ic = self.inflate_cells
        if ic > 0:
            inflated = np.zeros_like(grid)
            for di in range(-ic, ic + 1):
                for dj in range(-ic, ic + 1):
                    if di * di + dj * dj <= ic * ic:
                        i0 = max(0, di);   i1 = min(H, H + di)
                        si0 = max(0, -di); si1 = min(H, H - di)
                        j0 = max(0, dj);   j1 = min(W, W + dj)
                        sj0 = max(0, -dj); sj1 = min(W, W - dj)
                        h = min(i1 - i0, si1 - si0)
                        w = min(j1 - j0, sj1 - sj0)
                        if h > 0 and w > 0:
                            inflated[i0:i0 + h, j0:j0 + w] |= grid[si0:si0 + h, sj0:sj0 + w]
            grid = inflated

        si_g = max(0, min(H - 1, int((sy - min_y) / res)))
        sj_g = max(0, min(W - 1, int((sx - min_x) / res)))
        gi_g = max(0, min(H - 1, int((gy_w - min_y) / res)))
        gj_g = max(0, min(W - 1, int((gx_w - min_x) / res)))
        grid[si_g, sj_g] = 0   # dam bao start/goal luon free
        grid[gi_g, gj_g] = 0

        if (si_g, sj_g) == (gi_g, gj_g):
            return [(gx_w, gy_w)]

        SQRT2 = math.sqrt(2)
        DIRS = [
            (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
            (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2),
        ]

        def h(i, j):
            return math.hypot(i - gi_g, j - gj_g)

        g_score = np.full((H, W), np.inf, dtype=np.float32)
        g_score[si_g, sj_g] = 0.0
        came_from = {}
        open_heap = [(h(si_g, sj_g), 0.0, si_g, sj_g)]
        found = False

        while open_heap:
            _, g, ci, cj = heapq.heappop(open_heap)
            if g > g_score[ci, cj] + 1e-6:
                continue
            if ci == gi_g and cj == gj_g:
                found = True
                break
            for di, dj, cost in DIRS:
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < H and 0 <= nj < W):
                    continue
                if grid[ni, nj]:
                    continue
                new_g = g_score[ci, cj] + cost
                if new_g < g_score[ni, nj]:
                    g_score[ni, nj] = new_g
                    came_from[(ni, nj)] = (ci, cj)
                    heapq.heappush(open_heap, (new_g + h(ni, nj), new_g, ni, nj))

        if not found:
            return None  # khong co duong: robot se di thang (fallback)

        # Reconstruct path
        path_cells = []
        cur = (gi_g, gj_g)
        while cur in came_from:
            path_cells.append(cur)
            cur = came_from[cur]
        path_cells.append((si_g, sj_g))
        path_cells.reverse()

        # Convert to world coordinates
        waypoints = [
            (min_x + (cj + 0.5) * res, min_y + (ci + 0.5) * res)
            for ci, cj in path_cells
        ]

        # Simplify: xoa diem thang hang (khong doi huong)
        if len(waypoints) > 2:
            simplified = [waypoints[0]]
            for k in range(1, len(waypoints) - 1):
                px, py = simplified[-1]
                cx, cy = waypoints[k]
                nx, ny = waypoints[k + 1]
                v1 = (cx - px, cy - py)
                v2 = (nx - cx, ny - cy)
                cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
                mag = math.hypot(*v1) * math.hypot(*v2) + 1e-9
                if cross / mag > 0.12:   # goc > ~7 deg -> giu diem nay
                    simplified.append(waypoints[k])
            simplified.append(waypoints[-1])
        else:
            simplified = waypoints

        # Merge: bo diem qua gan nhau
        merged = [simplified[0]]
        for pt in simplified[1:]:
            if math.hypot(pt[0] - merged[-1][0], pt[1] - merged[-1][1]) >= res * 1.5:
                merged.append(pt)
        if math.hypot(merged[-1][0] - simplified[-1][0], merged[-1][1] - simplified[-1][1]) > 1e-3:
            merged.append(simplified[-1])

        return merged


class WaypointStore:
    def __init__(self, file_path=WAYPOINTS_FILE):
        self.file_path = str(file_path)
        self.lock = threading.Lock()
        self._waypoints = []
        self.load()

    def list(self):
        with self.lock:
            return [dict(wp) for wp in self._waypoints]

    def get(self, waypoint_id):
        waypoint_id = str(waypoint_id)
        with self.lock:
            for wp in self._waypoints:
                if wp.get('id') == waypoint_id:
                    return dict(wp)
        return None

    def add(self, label, x, y, theta=None):
        """Them mot waypoint. theta = None nghia la khong dat huong (giu huong cuoi)."""
        with self.lock:
            clean_label = self._clean_label(label, len(self._waypoints) + 1)
            stamp = int(time.time() * 1000)
            waypoint = {
                'id': f'wp_{stamp}_{len(self._waypoints) + 1}',
                'label': clean_label,
                'x': float(x),
                'y': float(y),
                'theta': float(theta) if theta is not None else None,
                'created_at': time.time(),
            }
            self._waypoints.append(waypoint)
            self._save_locked()
            return dict(waypoint)

    def delete(self, waypoint_id):
        waypoint_id = str(waypoint_id)
        with self.lock:
            before = len(self._waypoints)
            self._waypoints = [
                wp for wp in self._waypoints
                if wp.get('id') != waypoint_id
            ]
            deleted = len(self._waypoints) != before
            if deleted:
                self._save_locked()
            return deleted

    def load(self):
        with self.lock:
            self._waypoints = []
            if not os.path.isfile(self.file_path):
                return
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                items = raw.get('waypoints', raw) if isinstance(raw, dict) else raw
                if not isinstance(items, list):
                    raise ValueError('Waypoint file must contain a list')
                loaded = []
                for idx, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    theta_raw = item.get('theta', None)
                    theta_val = None
                    if theta_raw is not None:
                        try:
                            theta_val = float(theta_raw)
                        except (TypeError, ValueError):
                            theta_val = None
                    loaded.append({
                        'id': str(item.get('id') or f'wp_loaded_{idx + 1}'),
                        'label': self._clean_label(item.get('label'), idx + 1),
                        'x': float(item.get('x', 0.0)),
                        'y': float(item.get('y', 0.0)),
                        'theta': theta_val,
                        'created_at': float(item.get('created_at', 0.0) or 0.0),
                    })
                self._waypoints = loaded
                print(f'[Waypoints] LOADED {len(loaded)} points from {self.file_path}')
            except Exception as exc:
                print(f'[Waypoints] LOAD FAILED: {type(exc).__name__}: {exc!r}')
                self._waypoints = []

    def _save_locked(self):
        try:
            parent = os.path.dirname(os.path.abspath(self.file_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp_path = f'{self.file_path}.tmp'
            payload = {
                'version': 1,
                'saved_at': time.time(),
                'frame': 'lidar_map_world',
                'waypoints': self._waypoints,
            }
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.write('\n')
            os.replace(tmp_path, self.file_path)
        except Exception as exc:
            print(f'[Waypoints] SAVE FAILED: {type(exc).__name__}: {exc!r}')
            raise

    @staticmethod
    def _clean_label(label, fallback_idx):
        text = re.sub(r'\s+', ' ', str(label or '').strip())
        if not text:
            text = f'Point {fallback_idx}'
        return text[:WAYPOINT_LABEL_MAX_LEN]
