"""
web_ui.py
=========
Phan giao dien cho Camera Lidar Web.
Chua: ham ve dashboard len frame (OpenCV) va trang HTML cua web server.
"""

import cv2
import numpy as np


# ======================== DASHBOARD (OpenCV overlay) ========================

def draw_bar(frame, x, y, width, height, value, max_val, color, label=""):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), 1)
    mid = x + width // 2
    cv2.line(frame, (mid, y), (mid, y + height), (100, 100, 100), 1)

    ratio = np.clip(value / (max_val + 1e-9), -1.0, 1.0)
    bar_w = int(abs(ratio) * width // 2)
    if ratio >= 0:
        cv2.rectangle(frame, (mid, y + 1), (mid + bar_w, y + height - 1), color, -1)
    else:
        cv2.rectangle(frame, (mid - bar_w, y + 1), (mid, y + height - 1), color, -1)

    if label:
        cv2.putText(frame, label, (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)


def draw_history_graph(frame, x, y, width, height, history, max_val, color, label=""):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), 1)
    mid_y = y + height // 2
    cv2.line(frame, (x, mid_y), (x + width, mid_y), (80, 80, 80), 1)

    if len(history) < 2:
        return

    count = len(history)
    points = []
    for idx, val in enumerate(history):
        px = x + int(idx * width / count)
        ratio = np.clip(val / (max_val + 1e-9), -1.0, 1.0)
        py = mid_y - int(ratio * height / 2)
        points.append((px, py))

    for idx in range(1, len(points)):
        cv2.line(frame, points[idx - 1], points[idx], color, 1)

    if label:
        cv2.putText(frame, label, (x + 3, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def draw_dashboard(frame, info, vx_max=0.35, omega_max=1.5):
    height, width = frame.shape[:2]
    panel_w = 220
    overlay = frame[:, :panel_w].copy()
    cv2.rectangle(frame, (0, 0), (panel_w, height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.3, frame[:, :panel_w], 0.7, 0, frame[:, :panel_w])

    y0 = 20
    dy = 16
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    white = (220, 220, 220)
    green = (0, 255, 0)
    red = (0, 0, 255)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)

    def put(text, y, color=white):
        cv2.putText(frame, text, (8, y), font, font_scale, color, 1)
        return y + dy

    y0 = put("=== CAMERA BASED WEB TEST V2 ===", y0, cyan)
    y0 = put(f"Status: {info.get('status', '---')}", y0, green if info.get('ball_detected') else red)
    y0 += 5

    y0 = put("--- Velocity Command ---", y0, yellow)
    y0 = put(f"vx:    {info.get('vx', 0): .3f} m/s", y0)
    y0 = put(f"vy:    {info.get('vy', 0): .3f} m/s", y0)
    y0 = put(f"omega: {info.get('omega', 0): .3f} rad/s", y0)
    y0 += 5

    draw_bar(frame, 8, y0, 200, 12, info.get('vx', 0), vx_max, (0, 200, 0), 'vx')
    y0 += 28
    draw_bar(frame, 8, y0, 200, 12, info.get('omega', 0), omega_max, (200, 100, 0), 'omega')
    y0 += 28

    y0 = put("--- Error ---", y0, yellow)
    y0 = put(f"err_x:    {info.get('err_x', 0): .3f}", y0)
    y0 = put(f"err_dist: {info.get('err_dist', 0): .3f}", y0)
    y0 += 5

    y0 = put("--- Kalman State ---", y0, yellow)
    y0 = put(f"filt cx:   {info.get('kf_cx', 0): .1f} px", y0)
    y0 = put(f"filt bh:   {info.get('kf_bh', 0): .1f} px", y0)
    y0 = put(f"vel cx:    {info.get('kf_dcx', 0): .1f} px/s", y0)
    y0 = put(f"vel bh:    {info.get('kf_dbh', 0): .1f} px/s", y0)
    y0 += 5

    y0 = put("--- Align V2 ---", y0, yellow)
    y0 = put(f"mode:  {info.get('align_mode', '---')}", y0)
    y0 = put(f"gain:  {info.get('align_gain_scale', 1.0): .2f}", y0)
    y0 = put(f"cross: {info.get('align_overshoots', 0)}", y0)
    y0 = put(f"rate:  {info.get('bearing_rate', 0): .3f}", y0)
    y0 += 5

    y0 = put("--- PID Omega ---", y0, yellow)
    y0 = put(f"P: {info.get('omega_p', 0): .3f}  I: {info.get('omega_i', 0): .3f}", y0)
    y0 = put(f"D: {info.get('omega_d', 0): .3f}", y0)
    y0 += 3

    y0 = put("--- PD Vy ---", y0, yellow)
    y0 = put(f"P: {info.get('vy_p', 0): .3f}", y0)
    y0 = put(f"D: {info.get('vy_d', 0): .3f}", y0)
    y0 += 3

    y0 = put("--- PID Vx ---", y0, yellow)
    y0 = put(f"P: {info.get('vx_p', 0): .3f}  I: {info.get('vx_i', 0): .3f}", y0)
    y0 = put(f"D: {info.get('vx_d', 0): .3f}", y0)
    y0 += 5

    y0 = put("--- FPS ---", y0, yellow)
    y0 = put(f"Vision:  {info.get('fps_vision', 0): .1f} Hz", y0)
    y0 = put(f"Control: {info.get('fps_ctrl', 0): .1f} Hz", y0)

    graph_x = width - 215
    graph_y = 10
    draw_history_graph(frame, graph_x, graph_y, 200, 60, info.get('hist_err_x', []), 1.0, (0, 200, 255), 'err_x')
    draw_history_graph(frame, graph_x, graph_y + 70, 200, 60, info.get('hist_err_dist', []), 1.0, (0, 255, 100), 'err_dist')
    draw_history_graph(frame, graph_x, graph_y + 140, 200, 60, info.get('hist_omega', []), omega_max, (200, 100, 0), 'omega')
    draw_history_graph(frame, graph_x, graph_y + 210, 200, 60, info.get('hist_vx', []), vx_max, (0, 200, 0), 'vx')


# ======================== HTML PAGE ========================

HTML_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Robot Dashboard</title>
  <style>
    :root {
      --bg: #10110f;
      --panel: #181a17;
      --panel-2: #20231f;
      --text: #f3f5ef;
      --muted: #aeb6aa;
      --accent: #2fc5a8;
      --accent-2: #f0b84d;
      --danger: #d96a5f;
      --line: #30362f;
      --cam-w: 480px;
      --cam-h: 360px;
    }
    *, *::before, *::after { box-sizing: border-box; }
    body {
      margin: 0; background: var(--bg); color: var(--text);
      font-family: Inter, "Segoe UI", Arial, sans-serif; font-size: 13px;
    }
    .wrap { max-width: 1400px; margin: 0 auto; padding: 12px 18px; }
    /* HEADER */
    .header { margin-bottom: 12px; }
    .title { font-size: 20px; font-weight: 700; letter-spacing: .05em; }
    .hint { color: var(--muted); font-size: 11px; margin-top: 3px; }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; align-items: center; }
    /* BUTTONS */
    .button {
      appearance: none; border: 1px solid var(--line); background: var(--panel-2);
      color: var(--text); padding: 7px 12px; border-radius: 7px;
      cursor: pointer; font: inherit; font-size: 12px; white-space: nowrap;
    }
        .button[data-kind="start"] {
            border-color: #2d6f46;
        }
    .button[data-kind="stop"]  { border-color: #7d443d; }
    .button.compact { padding: 5px 9px; font-size: 11px; }
    .button.full    { width: 100%; }
    .button.active  { border-color: var(--accent); color: var(--accent); }
    .button.danger  { border-color: var(--danger); color: #ffd8d3; }
    .button:disabled { opacity: .4; cursor: not-allowed; }
    .button.cal-btn  { border-color: #7a5aaa; color: #c8a8f0; }
    /* PANEL */
    .panel {
      background: rgba(24,26,23,.95); border: 1px solid var(--line);
      border-radius: 8px; box-shadow: 0 6px 20px rgba(0,0,0,.3);
    }
    /* MEDIA ROW: Camera + Map same height */
    .media-row { display: flex; gap: 12px; align-items: flex-start; }
    .camera-panel { flex: 0 0 auto; overflow: hidden; }
    img.stream {
      width: var(--cam-w); height: var(--cam-h);
      display: block; background: #05070a; object-fit: contain;
    }
    .map-panel {
      flex: 1; height: var(--cam-h);
      display: flex; flex-direction: column;
    }
    .map-panel > .panel {
      flex: 1; display: flex; flex-direction: column; overflow: hidden;
    }
    .mapbox {
      padding: 8px 10px; display: flex; flex-direction: column;
      flex: 1; min-height: 0;
    }
    .mapbox-head {
      flex: 0 0 auto; display: flex;
      justify-content: space-between; align-items: center; margin-bottom: 6px;
    }
    .mapbox-title {
      font-size: 12px; font-weight: 700; color: var(--accent);
      text-transform: uppercase; letter-spacing: .07em;
    }
    .map-status { color: var(--muted); font-size: 11px; }
    #lidarMap {
      flex: 1; min-height: 0; width: 100%; display: block;
      background: #070a0f; border: 1px solid var(--line); border-radius: 6px;
    }
    #lidarMap.is-marking { cursor: crosshair; }
    /* BOTTOM ROW */
    .bottom-row {
      display: grid; grid-template-columns: 1fr 380px;
      gap: 12px; margin-top: 12px; align-items: start;
    }
    .stats-panel { padding: 10px 14px; }
    .stats-panel h2 {
      margin: 0 0 8px; font-size: 12px; color: var(--accent);
      text-transform: uppercase; letter-spacing: .07em;
    }
    .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); column-gap: 18px; }
    .kv {
      display: grid; grid-template-columns: 1fr auto; gap: 5px;
      padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,.05); font-size: 11px;
    }
    .kv:last-child { border-bottom: none; }
    .label { color: var(--muted); }
    .ok   { color: #83f28f; }
    .warn { color: #ffd16b; }
    /* WAYPOINTS PANEL */
    .waypoints-panel { padding: 10px 12px 12px; }
    .wp-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .wp-head h2 {
      margin: 0; font-size: 12px; color: var(--accent);
      text-transform: uppercase; letter-spacing: .07em;
    }
    .map-selection, .empty { color: var(--muted); font-size: 11px; }
    .map-toolbar, .waypoint-actions { display: flex; gap: 6px; align-items: center; }
    .waypoint-editor {
      display: flex; gap: 6px; flex-wrap: wrap;
      align-items: center; margin-bottom: 6px;
    }
    .heading-input { width: 76px; flex: none; }
    .heading-hint  { font-size: 10px; color: var(--muted); width: 100%; margin-top: -2px; }
    .text-input {
      min-width: 0; flex: 1; border: 1px solid var(--line); border-radius: 7px;
      background: #0d0f0c; color: var(--text); padding: 6px 10px; font: inherit; font-size: 12px;
    }
    .waypoint-list { display: grid; gap: 6px; max-height: 220px; overflow: auto; margin-bottom: 8px; }
    .waypoint-item {
      display: grid; grid-template-columns: minmax(0,1fr) auto;
      gap: 8px; align-items: center; padding: 7px 9px;
      border: 1px solid rgba(255,255,255,.07); border-radius: 7px;
      background: rgba(255,255,255,.03);
    }
    .waypoint-title {
      font-size: 12px; font-weight: 700;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .waypoint-meta { color: var(--muted); font-size: 11px; margin-top: 2px; }
    .mini-button {
      appearance: none; border: 1px solid var(--line); border-radius: 6px;
      background: #11130f; color: var(--text); padding: 4px 7px;
      cursor: pointer; font: inherit; font-size: 11px;
    }
    .mini-button.go     { border-color: var(--accent); color: var(--accent); }
    .mini-button.delete { border-color: var(--danger); color: #ffd8d3; }
    @media (max-width: 900px) {
      :root { --cam-w: 320px; --cam-h: 240px; }
      .bottom-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
      margin-bottom: 4px;
    }
    .map-status,
    .map-selection,
    .empty {
      color: var(--muted);
      font-size: 12px;
    }
    .map-toolbar,
    .waypoint-actions,
    .waypoint-editor {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .waypoint-editor {
      margin-top: 10px;
      flex-wrap: wrap;
    }
    .heading-input {
      width: 90px;
      flex: none;
    }
    .heading-hint {
      font-size: 11px;
      color: var(--muted);
      width: 100%;
      margin-top: -2px;
    }
    .text-input {
      min-width: 0;
      flex: 1;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0d0f0c;
      color: var(--text);
      padding: 10px 12px;
      font: inherit;
      font-size: 13px;
    }
    .map-selection {
      margin-top: 8px;
      min-height: 16px;
    }
    .waypoint-list {
      display: grid;
      gap: 8px;
      margin: 12px 0;
      max-height: 220px;
      overflow: auto;
    }
    .waypoint-item {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 10px;
      border: 1px solid rgba(255, 255, 255, 0.07);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.03);
    }
    .waypoint-title {
      font-size: 13px;
      font-weight: 700;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .waypoint-meta {
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
    }
    .mini-button {
      appearance: none;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #11130f;
      color: var(--text);
      padding: 7px 9px;
      cursor: pointer;
      font: inherit;
      font-size: 12px;
    }
    .mini-button.go {
      border-color: var(--accent);
      color: var(--accent);
    }
    .mini-button.delete {
      border-color: var(--danger);
      color: #ffd8d3;
    }
    #lidarMap {
      width: 100%;
      aspect-ratio: 1 / 1;
      display: block;
      background: #070a0f;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    #lidarMap.is-marking {
      cursor: crosshair;
    }
    @media (max-width: 960px) {
      .grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">

    <!-- HEADER -->
    <div class="header">
      <div class="title">Robot Dashboard</div>
      <div class="hint" id="endpoint"></div>
      <div class="actions">
        <button class="button" data-kind="start" id="startBtn">&#x25B6; Motor On</button>
        <button class="button" data-kind="stop"  id="stopBtn">&#x25A0; Motor Off</button>
        <button class="button" data-kind="start" id="trackingOnBtn">&#x1F4F7; Camera On</button>
        <button class="button" data-kind="stop"  id="trackingOffBtn">&#x1F4F7; Camera Off</button>
        <button class="button" id="saveMapBtn">&#x1F4BE; Save Map</button>
        <button class="button cal-btn" onclick="window.open('/calibration','_blank')">&#x2699; Calibrate &#x2197;</button>
      </div>
    </div>

    <!-- MEDIA ROW: Camera + LiDAR Map (same height) -->
    <div class="media-row">
      <div class="camera-panel panel">
        <img class="stream" src="/stream.mjpg" alt="camera stream">
      </div>
      <div class="map-panel">
        <div class="panel" style="flex:1;display:flex;flex-direction:column;overflow:hidden;">
          <div class="mapbox">
            <div class="mapbox-head">
              <div>
                <div class="mapbox-title">LiDAR Map</div>
                <div class="map-status" id="mapMeta">-</div>
              </div>
              <div class="map-toolbar">
                <button class="button compact" id="addPointBtn">+ Point</button>
                <button class="button compact" id="cancelDraftBtn">Clear</button>
              </div>
            </div>
            <canvas id="lidarMap" width="420" height="420"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- BOTTOM ROW: Stats + Waypoints -->
    <div class="bottom-row">
      <div class="panel stats-panel">
        <h2>Live State</h2>
        <div class="stats-grid" id="stats"></div>
      </div>
      <div class="panel waypoints-panel">
        <div class="wp-head">
          <h2>Waypoints</h2>
        </div>
        <div class="waypoint-editor">
          <input class="text-input" id="waypointLabel" maxlength="48" placeholder="Label">
          <input class="text-input heading-input" id="waypointHeading" type="number" step="1" placeholder="deg">
          <button class="button compact" id="saveWaypointBtn">Save</button>
          <div class="heading-hint">Heading deg (0=+X, 90=+Y). Bo trong = giu huong hien tai.</div>
        </div>
        <div class="map-selection" id="mapSelection">Draft: -</div>
        <div class="waypoint-list" id="waypointList"></div>
        <button class="button compact full danger" id="cancelNavBtn">Cancel Navigation</button>
      </div>
    </div>

  </div>
  <script>
    const statsEl = document.getElementById('stats');
        const lidarCanvas = document.getElementById('lidarMap');
        const lidarCtx = lidarCanvas.getContext('2d');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const trackingOnBtn = document.getElementById('trackingOnBtn');
        const trackingOffBtn = document.getElementById('trackingOffBtn');
        const saveMapBtn = document.getElementById('saveMapBtn');
        const mapMeta = document.getElementById('mapMeta');
        const addPointBtn = document.getElementById('addPointBtn');
        const cancelDraftBtn = document.getElementById('cancelDraftBtn');
        const waypointLabel = document.getElementById('waypointLabel');
        const waypointHeading = document.getElementById('waypointHeading');
        const saveWaypointBtn = document.getElementById('saveWaypointBtn');
        const waypointList = document.getElementById('waypointList');
        const mapSelection = document.getElementById('mapSelection');
        const cancelNavBtn = document.getElementById('cancelNavBtn');
        let latestState = {};
        let addMode = false;
        let draftPoint = null;
        let mapView = { scale: 1, cx: 0, cy: 0, pad: 18, hitboxes: [] };
    document.getElementById('endpoint').textContent = `${location.origin}/stream.mjpg`;

    function row(label, value, cls = '') {
      return `<div class="kv"><div class="label">${label}</div><div class="${cls}">${value}</div></div>`;
    }

    function num(value, digits = 3) {
      const n = Number(value);
      return Number.isFinite(n) ? n.toFixed(digits) : '-';
    }

    function escapeHtml(value) {
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function drawLidarMap(state) {
      const ctx = lidarCtx;
      const w = lidarCanvas.width;
      const h = lidarCanvas.height;
      const pad = 18;
      const cx = w / 2;
      const cy = h / 2;
      const path = Array.isArray(state.lidar_path) ? state.lidar_path : [];
      const cloud = Array.isArray(state.lidar_cloud) ? state.lidar_cloud : [];
      const gmap = Array.isArray(state.lidar_map) ? state.lidar_map : [];
      const waypoints = Array.isArray(state.waypoints) ? state.waypoints : [];
      const navTarget = state.navigation_target || null;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#070a0f';
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = '#1f2a37';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 8; i++) {
        const p = pad + i * (w - 2 * pad) / 8;
        ctx.beginPath();
        ctx.moveTo(p, pad);
        ctx.lineTo(p, h - pad);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(pad, p);
        ctx.lineTo(w - pad, p);
        ctx.stroke();
      }

      // Tinh maxAbs bao gom ca cloud + gmap + path + pose hien tai
      let maxAbs = 0.5;
      for (const p of path) {
        maxAbs = Math.max(maxAbs, Math.abs(p[0] || 0), Math.abs(p[1] || 0));
      }
      for (const wp of waypoints) {
        maxAbs = Math.max(maxAbs, Math.abs(wp.x || 0), Math.abs(wp.y || 0));
      }
      if (draftPoint) {
        maxAbs = Math.max(maxAbs, Math.abs(draftPoint.x || 0), Math.abs(draftPoint.y || 0));
      }
      if (navTarget) {
        maxAbs = Math.max(maxAbs, Math.abs(navTarget.x || 0), Math.abs(navTarget.y || 0));
      }
      maxAbs = Math.max(maxAbs, Math.abs(state.lidar_x || 0), Math.abs(state.lidar_y || 0));
      // Cloud co the rat lon -> lay percentile xap xi (max 95%) de tranh phong to qua muc
      if (cloud.length > 0) {
        let cloudMax = 0;
        for (const p of cloud) {
          const v = Math.max(Math.abs(p[0]), Math.abs(p[1]));
          if (v > cloudMax) cloudMax = v;
        }
        // Khong cho cloud lam scale qua nho (giu pose ro)
        maxAbs = Math.max(maxAbs, Math.min(cloudMax, 8.0));
      }
      // Tinh ca global map: dung 90th percentile de tranh outlier dan
      if (gmap.length > 0) {
        const xs = gmap.map(p => Math.max(Math.abs(p[0]), Math.abs(p[1]))).sort((a, b) => a - b);
        const idx90 = Math.floor(xs.length * 0.9);
        const mapMax = xs[Math.min(idx90, xs.length - 1)] || 0;
        maxAbs = Math.max(maxAbs, Math.min(mapMax, 12.0));
      }
      const scale = (w / 2 - pad) / maxAbs;
      const toPx = (x, y) => [cx + x * scale, cy - y * scale];
      mapView = { scale, cx, cy, pad, hitboxes: [], toPx };

      // ===== Ve global map TRUOC (mau xam, lop nen) =====
      if (gmap.length > 0) {
        ctx.fillStyle = 'rgba(180, 180, 200, 0.55)';
        for (const p of gmap) {
          const px = cx + (p[0] || 0) * scale;
          const py = cy - (p[1] || 0) * scale;
          if (px < pad || px > w - pad || py < pad || py > h - pad) continue;
          ctx.fillRect(px - 1, py - 1, 2, 2);
        }
      }

      // ===== Ve point cloud song (mau cyan, lop tren) =====
      // Moi diem co alpha (cot 2): diem moi sang, diem cu mo dan.
      if (cloud.length > 0) {
        for (const p of cloud) {
          const px = cx + (p[0] || 0) * scale;
          const py = cy - (p[1] || 0) * scale;
          if (px < pad || px > w - pad || py < pad || py > h - pad) continue;
          const a = Math.max(0.1, Math.min(1.0, p[2] || 1.0));
          // Mau xanh cyan voi do trong suot theo alpha
          ctx.fillStyle = `rgba(0, 220, 255, ${a.toFixed(2)})`;
          ctx.fillRect(px - 1, py - 1, 2, 2);
        }
      }

      ctx.strokeStyle = '#2d4258';
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.stroke();

      if (path.length > 1) {
        ctx.strokeStyle = '#f3c969';
        ctx.lineWidth = 2;
        ctx.beginPath();
        path.forEach((p, idx) => {
          const [px, py] = toPx(p[0] || 0, p[1] || 0);
          if (idx === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      }

      const drawPin = (x, y, label, color, active, id, theta) => {
        const [px, py] = toPx(x || 0, y || 0);
        if (px < pad || px > w - pad || py < pad || py > h - pad) return;
        if (id) mapView.hitboxes.push({ id, x: px, y: py });
        ctx.save();
        // Ve mui ten heading neu co theta
        if (theta !== null && theta !== undefined && isFinite(theta)) {
          const arrowLen = 18;
          // Trong he world: theta = 0 -> +X. Tren canvas y dao chieu, nen ve theta=0 -> phai
          const ax = px + Math.cos(theta) * arrowLen;
          const ay = py - Math.sin(theta) * arrowLen;
          ctx.strokeStyle = color;
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(ax, ay);
          ctx.stroke();
          // Dau mui ten
          const headSize = 5;
          const angle = Math.atan2(ay - py, ax - px);
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(ax - headSize * Math.cos(angle - Math.PI / 6),
                     ay - headSize * Math.sin(angle - Math.PI / 6));
          ctx.moveTo(ax, ay);
          ctx.lineTo(ax - headSize * Math.cos(angle + Math.PI / 6),
                     ay - headSize * Math.sin(angle + Math.PI / 6));
          ctx.stroke();
        }
        ctx.fillStyle = color;
        ctx.strokeStyle = active ? '#ffffff' : '#0b0d0a';
        ctx.lineWidth = active ? 3 : 2;
        ctx.beginPath();
        ctx.arc(px, py, active ? 8 : 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.font = '12px "Segoe UI", Arial, sans-serif';
        const fullLabel = String(label || 'Point');
        const safeLabel = fullLabel.length > 18 ? `${fullLabel.slice(0, 17)}.` : fullLabel;
        const labelW = Math.min(120, ctx.measureText(safeLabel).width + 14);
        const lx = Math.min(px + 10, w - pad - labelW);
        const ly = Math.max(pad + 18, py - 9);
        ctx.fillStyle = 'rgba(13, 15, 12, 0.86)';
        ctx.fillRect(lx, ly - 14, labelW, 20);
        ctx.fillStyle = '#f3f5ef';
        ctx.fillText(safeLabel, lx + 7, ly);
        ctx.restore();
      };

      if (navTarget) {
        const [tx, ty] = toPx(navTarget.x || 0, navTarget.y || 0);
        const [rxLine, ryLine] = toPx(state.lidar_x || 0, state.lidar_y || 0);
        ctx.strokeStyle = state.navigation_active ? '#2fc5a8' : '#f0b84d';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 5]);
        ctx.beginPath();
        ctx.moveTo(rxLine, ryLine);
        ctx.lineTo(tx, ty);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      for (const wp of waypoints) {
        const active = state.navigation_active && navTarget && navTarget.id === wp.id;
        drawPin(wp.x, wp.y, wp.label, active ? '#2fc5a8' : '#f0b84d', active, wp.id,
                wp.theta != null ? wp.theta : null);
      }
      if (draftPoint) {
        drawPin(draftPoint.x, draftPoint.y, 'Draft', '#d96a5f', true, null,
                draftPoint.theta != null ? draftPoint.theta : null);
      }

      const x = state.lidar_x || 0;
      const y = state.lidar_y || 0;
      let thetaDeg;
      if (state.heading_effective_deg !== undefined && state.heading_effective_deg !== null) {
        thetaDeg = state.heading_effective_deg;
      } else {
        thetaDeg = state.lidar_theta_deg || 0;
      }
      const theta = thetaDeg * Math.PI / 180;
      const [rx, ry] = toPx(x, y);
      const headingLen = 42;
      const hx = rx + Math.cos(theta) * headingLen;
      const hy = ry - Math.sin(theta) * headingLen;

      const ring1Radius = 0.5 * scale;
      const ring2Radius = 1.0 * scale;
      ctx.strokeStyle = 'rgba(131, 242, 143, 0.22)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(rx, ry, ring1Radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = 'rgba(131, 242, 143, 0.15)';
      ctx.beginPath();
      ctx.arc(rx, ry, ring2Radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(131, 242, 143, 0.55)';
      ctx.font = '9px Consolas, monospace';
      ctx.fillText('0.5m', rx + ring1Radius * 0.71 + 2, ry - ring1Radius * 0.71);
      ctx.fillText('1.0m', rx + ring2Radius * 0.71 + 2, ry - ring2Radius * 0.71);

      const compassR1 = headingLen + 4;
      const compassR2 = headingLen + 10;
      for (let deg = 0; deg < 360; deg += 90) {
        const rad = deg * Math.PI / 180;
        const tx1 = rx + Math.cos(rad) * compassR1;
        const ty1 = ry - Math.sin(rad) * compassR1;
        const tx2 = rx + Math.cos(rad) * compassR2;
        const ty2 = ry - Math.sin(rad) * compassR2;
        ctx.strokeStyle = 'rgba(131, 242, 143, 0.4)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(tx1, ty1);
        ctx.lineTo(tx2, ty2);
        ctx.stroke();
        ctx.fillStyle = 'rgba(131, 242, 143, 0.55)';
        ctx.font = '10px Consolas, monospace';
        const labelR = headingLen + 18;
        const lx = rx + Math.cos(rad) * labelR - 6;
        const ly = ry - Math.sin(rad) * labelR + 4;
        ctx.fillText(`${deg}`, lx, ly);
      }

      ctx.fillStyle = '#83f28f';
      ctx.beginPath();
      ctx.arc(rx, ry, 6, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = '#83f28f';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(rx, ry);
      ctx.lineTo(hx, hy);
      ctx.stroke();
      const arrowHead = 9;
      const arrowAngle = Math.atan2(hy - ry, hx - rx);
      ctx.beginPath();
      ctx.moveTo(hx, hy);
      ctx.lineTo(
        hx - arrowHead * Math.cos(arrowAngle - Math.PI / 6),
        hy - arrowHead * Math.sin(arrowAngle - Math.PI / 6)
      );
      ctx.moveTo(hx, hy);
      ctx.lineTo(
        hx - arrowHead * Math.cos(arrowAngle + Math.PI / 6),
        hy - arrowHead * Math.sin(arrowAngle + Math.PI / 6)
      );
      ctx.stroke();

      let displayDeg = thetaDeg;
      while (displayDeg > 180) displayDeg -= 360;
      while (displayDeg < -180) displayDeg += 360;
      const headingText = `${displayDeg >= 0 ? '+' : ''}${displayDeg.toFixed(1)} deg`;
      ctx.font = 'bold 13px Consolas, monospace';
      const textW = ctx.measureText(headingText).width + 10;
      const textBgX = hx + 6;
      const textBgY = hy - 9;
      ctx.fillStyle = 'rgba(13, 15, 12, 0.85)';
      ctx.fillRect(textBgX, textBgY, textW, 18);
      ctx.fillStyle = '#83f28f';
      ctx.fillText(headingText, textBgX + 5, textBgY + 13);

      const panelW = 130;
      const panelH = 50;
      const panelX = pad;
      const panelY = h - panelH - pad;
      ctx.fillStyle = 'rgba(13, 15, 12, 0.88)';
      ctx.fillRect(panelX, panelY, panelW, panelH);
      ctx.strokeStyle = '#83f28f';
      ctx.lineWidth = 1;
      ctx.strokeRect(panelX, panelY, panelW, panelH);
      ctx.fillStyle = '#9cb0c1';
      ctx.font = '10px Consolas, monospace';
      ctx.fillText('HEADING', panelX + 8, panelY + 14);
      ctx.fillStyle = '#83f28f';
      ctx.font = 'bold 22px Consolas, monospace';
      ctx.fillText(`${displayDeg.toFixed(1)}°`, panelX + 8, panelY + 38);
      let compass = 'E';
      if (displayDeg > 45 && displayDeg <= 135) compass = 'N';
      else if (displayDeg > 135 || displayDeg <= -135) compass = 'W';
      else if (displayDeg > -135 && displayDeg <= -45) compass = 'S';
      ctx.fillStyle = '#9cb0c1';
      ctx.font = 'bold 16px Consolas, monospace';
      ctx.fillText(compass, panelX + panelW - 24, panelY + 38);

      ctx.fillStyle = '#9cb0c1';
      ctx.font = '12px Consolas, monospace';
      ctx.fillText(`${maxAbs.toFixed(1)} m`, pad, h - 8);
      ctx.fillText(state.lidar_status || 'DISABLED', pad, 16);
      ctx.fillText(`live: ${cloud.length}`, w - pad - 80, 16);
      ctx.fillText(`map: ${state.lidar_map_total || 0}`, w - pad - 80, 32);
      if (state.lidar_map_loaded) {
        ctx.fillStyle = state.lidar_map_relocalized ? '#83f28f' : '#f3c969';
        ctx.fillText(state.lidar_map_relocalized ? 'RELOC OK' : 'RELOC...', pad, 32);
      }
      // Hien thi nav phase neu dang navigation
      if (state.navigation_active) {
        ctx.fillStyle = '#2fc5a8';
        ctx.font = 'bold 13px Consolas, monospace';
        ctx.fillText(`NAV: ${state.navigation_phase || '?'}`, pad, h - panelH - pad - 8);
      }
    }

    function setAddMode(enabled) {
      addMode = Boolean(enabled);
      addPointBtn.classList.toggle('active', addMode);
      lidarCanvas.classList.toggle('is-marking', addMode);
    }

    function updateDraftText(text) {
      if (text) {
        mapSelection.textContent = text;
      } else if (draftPoint) {
        mapSelection.textContent = `Draft: ${num(draftPoint.x)}, ${num(draftPoint.y)} m`;
      } else {
        mapSelection.textContent = 'Draft: -';
      }
    }

    function renderWaypointList(state) {
      const waypoints = Array.isArray(state.waypoints) ? state.waypoints : [];
      if (!waypoints.length) {
        waypointList.innerHTML = '<div class="empty">No saved locations</div>';
        return;
      }
      waypointList.innerHTML = waypoints.map((wp) => {
        const active = state.navigation_active && state.navigation_target && state.navigation_target.id === wp.id;
        const headingTxt = (wp.theta != null && isFinite(wp.theta))
          ? `, ${(wp.theta * 180 / Math.PI).toFixed(0)}deg`
          : ', free heading';
        return `
          <div class="waypoint-item">
            <div>
              <div class="waypoint-title">${escapeHtml(wp.label)}${active ? ' - active' : ''}</div>
              <div class="waypoint-meta">${num(wp.x)}, ${num(wp.y)} m${headingTxt}</div>
            </div>
            <div class="waypoint-actions">
              <button class="mini-button go" data-go="${escapeHtml(wp.id)}">Go</button>
              <button class="mini-button delete" data-delete="${escapeHtml(wp.id)}">X</button>
            </div>
          </div>
        `;
      }).join('');
    }

    function canvasPoint(event) {
      const rect = lidarCanvas.getBoundingClientRect();
      return {
        px: (event.clientX - rect.left) * lidarCanvas.width / rect.width,
        py: (event.clientY - rect.top) * lidarCanvas.height / rect.height,
      };
    }

    function worldFromCanvas(point) {
      return {
        x: (point.px - mapView.cx) / mapView.scale,
        y: (mapView.cy - point.py) / mapView.scale,
      };
    }

    function waypointHit(point) {
      let best = null;
      let bestDist = 15;
      for (const hit of mapView.hitboxes || []) {
        const dist = Math.hypot(point.px - hit.x, point.py - hit.y);
        if (dist < bestDist) {
          best = hit;
          bestDist = dist;
        }
      }
      return best;
    }

    async function postJson(url, payload = {}) {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      let data = {};
      try {
        data = await response.json();
      } catch (error) {
        data = {};
      }
      return { ...data, httpOk: response.ok };
    }

    async function startNavigation(id) {
      const result = await postJson('/api/navigation/start', { id });
      updateDraftText(result.ok ? 'Navigation: started' : `Navigation: ${result.reason || 'failed'}`);
      await refresh();
    }

    async function deleteWaypoint(id) {
      await postJson('/api/waypoints/delete', { id });
      await refresh();
    }

    async function cancelNavigation() {
      await postJson('/api/navigation/cancel');
      await refresh();
    }

    async function saveWaypoint() {
      if (!draftPoint) return;
      saveWaypointBtn.disabled = true;
      // Parse heading: bo trong = null (khong dat huong)
      const headingRaw = waypointHeading.value.trim();
      const payload = {
        label: waypointLabel.value,
        x: draftPoint.x,
        y: draftPoint.y,
      };
      if (headingRaw !== '') {
        const headingNum = parseFloat(headingRaw);
        if (!isNaN(headingNum) && isFinite(headingNum)) {
          payload.theta_deg = headingNum;
        }
      }
      const result = await postJson('/api/waypoints', payload);
      if (result.ok) {
        draftPoint = null;
        waypointLabel.value = '';
        waypointHeading.value = '';
        setAddMode(false);
        updateDraftText('Saved point');
      } else {
        updateDraftText(`Save failed: ${result.reason || 'invalid point'}`);
      }
      saveWaypointBtn.disabled = false;
      await refresh();
    }

    async function refresh() {
      try {
        const response = await fetch('/state');
        const state = await response.json();
        latestState = state;
        const statusClass = (state.navigation_active || state.ball_detected || state.status === 'ARRIVED') ? 'ok' : 'warn';
        const waypoints = Array.isArray(state.waypoints) ? state.waypoints : [];
        const navTarget = state.navigation_target || null;
        const headingDeg = (state.heading_effective_deg !== undefined && state.heading_effective_deg !== null)
          ? state.heading_effective_deg
          : (state.lidar_theta_deg || 0);
                startBtn.disabled = !state.serial_enabled || state.motor_enabled;
                stopBtn.disabled = !state.serial_enabled || !state.motor_enabled;
                trackingOnBtn.disabled = !!state.camera_tracking_enabled;
                trackingOffBtn.disabled = !state.camera_tracking_enabled;
                trackingOnBtn.classList.toggle('active', !!state.camera_tracking_enabled);
                trackingOffBtn.classList.toggle('active', !state.camera_tracking_enabled);
                cancelNavBtn.disabled = !state.navigation_active;
                saveWaypointBtn.disabled = !draftPoint;
                mapMeta.textContent = `${state.lidar_status || 'DISABLED'} | ${waypoints.length} saved`;
        statsEl.innerHTML = [
          row('status', state.status, statusClass),
            row('camera tracking', state.camera_tracking_enabled ? 'enabled' : 'paused', state.camera_tracking_enabled ? 'ok' : 'warn'),
          row('serial', state.serial_enabled ? 'enabled' : 'disabled'),
              row('serial port', state.serial_port || '-'),
                    row('motor', state.motor_enabled ? 'armed' : 'stopped', state.motor_enabled ? 'ok' : 'warn'),
                    row('reason', state.motor_reason),
          row('vx / vy', `${num(state.vx)} / ${num(state.vy)}`),
          row('omega', num(state.omega)),
          row('err_x', num(state.err_x)),
          row('err_dist', num(state.err_dist)),
          row('dist m', num(state.dist_m)),
          row('bbox_h', num(state.kf_bh, 1)),
          row('bbox target', num(state.target_bbox_h, 1)),
          row('base vx', num(state.measured_vx)),
          row('base wz', num(state.measured_wz)),
          row('lidar', state.lidar_status, state.lidar_status === 'TRACKING' ? 'ok' : 'warn'),
          row('lidar port', state.lidar_port || '-'),
          row('lidar pose', `${num(state.lidar_x)}, ${num(state.lidar_y)}, ${num(state.lidar_theta_deg, 1)} deg`),
          row('heading', `${num(headingDeg, 1)} deg`),
          row('heading src', state.heading_source || '-'),
          row('heading std', `${num(state.heading_std_deg, 1)} deg`),
          row('lidar baud', state.lidar_baudrate || '-'),
          row('lidar scans', `${state.lidar_accepted}/${state.lidar_scans}`),
          row('lidar err', `${num(state.lidar_error)} m`),
          row('cloud pts', `${state.lidar_cloud_count || 0}`),
          row('map pts', `${state.lidar_map_total || 0}`),
          row('map file', state.lidar_map_loaded ? 'loaded' : 'fresh', state.lidar_map_loaded ? 'ok' : 'warn'),
          row('reloc', state.lidar_map_loaded ? (state.lidar_map_relocalized ? 'success' : 'pending') : 'n/a',
              state.lidar_map_relocalized ? 'ok' : 'warn'),
          row('last save', state.lidar_map_last_save ? new Date(state.lidar_map_last_save * 1000).toLocaleTimeString() : '-'),
          row('waypoints', waypoints.length),
          row('nav', state.navigation_status || 'IDLE', state.navigation_active ? 'ok' : ''),
          row('phase', state.navigation_phase || 'IDLE', state.navigation_active ? 'ok' : ''),
          row('goal', navTarget ? navTarget.label : '-'),
          row('goal dist', `${num(state.navigation_distance)} m`),
          row('heading err', `${num(state.navigation_bearing_error_deg, 1)} deg`),
          row('lidar note', state.lidar_error_text || '-'),
          row('vision fps', num(state.fps_vision, 1)),
          row('control fps', num(state.fps_ctrl, 1)),
          row('last seen', `${num(state.last_seen_age, 2)} s`),
        ].join('');
        drawLidarMap(state);
        renderWaypointList(state);
        updateDraftText();
      } catch (error) {
        statsEl.innerHTML = row('state', 'fetch error', 'warn');
      }
    }

        async function postAction(url) {
            await fetch(url, { method: 'POST' });
            await refresh();
        }

        startBtn.addEventListener('click', () => postAction('/api/motor/start'));
        stopBtn.addEventListener('click', () => postAction('/api/motor/stop'));
        trackingOnBtn.addEventListener('click', () => postAction('/api/camera_tracking/start'));
        trackingOffBtn.addEventListener('click', () => postAction('/api/camera_tracking/stop'));
        saveMapBtn.addEventListener('click', async () => {
          saveMapBtn.disabled = true;
          saveMapBtn.textContent = 'Saving...';
          try {
            const r = await fetch('/api/map/save', { method: 'POST' });
            const j = await r.json();
            saveMapBtn.textContent = j.ok ? 'Saved!' : 'Failed';
          } catch (e) {
            saveMapBtn.textContent = 'Error';
          }
          setTimeout(() => {
            saveMapBtn.textContent = 'Save Map';
            saveMapBtn.disabled = false;
          }, 1500);
        });

        addPointBtn.addEventListener('click', () => {
          setAddMode(!addMode);
        });

        cancelDraftBtn.addEventListener('click', () => {
          draftPoint = null;
          setAddMode(false);
          updateDraftText();
          drawLidarMap(latestState);
        });

        saveWaypointBtn.addEventListener('click', saveWaypoint);
        cancelNavBtn.addEventListener('click', cancelNavigation);

        lidarCanvas.addEventListener('click', async (event) => {
          const point = canvasPoint(event);
          const hit = waypointHit(point);
          if (hit && !addMode) {
            await startNavigation(hit.id);
            return;
          }
          draftPoint = worldFromCanvas(point);
          // Sync heading hien tai vao draft (de live preview)
          updateDraftHeading();
          setAddMode(true);
          updateDraftText();
          drawLidarMap(latestState);
        });

        // Khi user nhap so vao input theta -> cap nhat draftPoint.theta de live preview
        function updateDraftHeading() {
          if (!draftPoint) return;
          const raw = waypointHeading.value.trim();
          if (raw === '') {
            draftPoint.theta = null;
          } else {
            const num = parseFloat(raw);
            draftPoint.theta = (isNaN(num) || !isFinite(num)) ? null : num * Math.PI / 180;
          }
        }
        waypointHeading.addEventListener('input', () => {
          updateDraftHeading();
          drawLidarMap(latestState);
        });

        waypointList.addEventListener('click', async (event) => {
          const goBtn = event.target.closest('[data-go]');
          const deleteBtn = event.target.closest('[data-delete]');
          if (goBtn) {
            await startNavigation(goBtn.dataset.go);
          } else if (deleteBtn) {
            await deleteWaypoint(deleteBtn.dataset.delete);
          }
        });

    refresh();
    setInterval(refresh, 300);
  </script>
</body>
</html>
"""


# ======================== CALIBRATION PAGE ========================

CALIBRATION_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Robot Calibration</title>
  <style>
    :root {
      --bg: #10110f; --panel: #181a17; --panel-2: #1e1e1e;
      --text: #f3f5ef; --muted: #aeb6aa; --accent: #4dd0e1;
      --danger: #d96a5f; --line: #30362f;
    }
    *, *::before, *::after { box-sizing: border-box; }
    body {
      margin: 0; background: var(--bg); color: var(--text);
      font-family: Inter, "Segoe UI", Arial, sans-serif; font-size: 13px;
    }
    .wrap { max-width: 760px; margin: 0 auto; padding: 18px 20px; }
    .page-header {
      display: flex; align-items: center; gap: 14px; margin-bottom: 18px;
    }
    .back-btn {
      appearance: none; border: 1px solid var(--line); background: var(--panel-2);
      color: var(--muted); padding: 6px 12px; border-radius: 7px;
      cursor: pointer; font: inherit; font-size: 12px; text-decoration: none;
    }
    .back-btn:hover { color: var(--text); }
    .page-title { font-size: 20px; font-weight: 700; }
    .panel {
      background: var(--panel-2); border: 1px solid var(--line);
      border-radius: 8px; padding: 18px; margin-bottom: 14px;
    }
    .panel h3 { margin: 0 0 14px; color: var(--accent); font-size: 15px; }
    .panel h4 { margin: 14px 0 6px; color: #999; font-size: 13px; }
    .warning {
      background: #3a2a0a; border: 1px solid #7a5a1a; border-radius: 6px;
      padding: 10px 12px; color: #ffb84d; font-size: 12px; margin-bottom: 14px;
    }
    .cal-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin: 8px 0; }
    .cal-btn {
      appearance: none; background: #1a3a5a; color: #4dd0e1;
      border: 1px solid #2a5a8a; padding: 8px 14px; border-radius: 6px;
      cursor: pointer; font: inherit; font-size: 12px;
    }
    .cal-btn:hover { background: #254a70; }
    .cal-btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .cal-btn.danger { background: #3a1010; color: #ff8877; border-color: #7a2020; }
    .cal-btn.danger:hover { background: #4a1818; }
    .cal-input {
      background: #151715; border: 1px solid var(--line); color: var(--text);
      padding: 7px 10px; border-radius: 6px; width: 110px; font: inherit; font-size: 12px;
    }
    .cal-label { font-size: 12px; color: var(--muted); }
    .cal-progress-bar {
      width: 100%; height: 16px; background: #151715;
      border-radius: 8px; overflow: hidden; margin: 8px 0;
    }
    .cal-progress-fill {
      height: 100%; background: linear-gradient(90deg, #2fc5a8, #4dd0e1);
      width: 0%; transition: width 0.3s;
    }
    .cal-status {
      font-size: 12px; padding: 7px 10px; border-radius: 6px; margin: 6px 0;
    }
    .cal-status.idle    { background: #1a1a1a; color: #888; }
    .cal-status.running { background: #0e2a44; color: #4dd0e1; }
    .cal-status.done    { background: #0e3a1e; color: #4dffaa; }
    .cal-status.error   { background: #3a1010; color: #ff6644; }
    .cal-status.aborted { background: #3a2010; color: #ffb84d; }
    .cal-log {
      background: #050705; border: 1px solid var(--line); border-radius: 6px;
      padding: 8px; font-family: 'Courier New', monospace; font-size: 11px;
      height: 240px; overflow-y: auto; white-space: pre-wrap; color: #9ab0c0;
    }
    .cal-config {
      background: #080a07; border: 1px solid var(--line); border-radius: 6px;
      padding: 10px; font-family: 'Courier New', monospace; font-size: 12px;
      line-height: 1.6; color: #b0c8b0;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="page-header">
      <a class="back-btn" href="javascript:history.back()">&#x2190; Back</a>
      <div class="page-title">&#x2699;&#xFE0F; Robot Calibration</div>
    </div>

    <div class="warning">
      &#x26A0; <b>Motor calibration:</b> NHAC robot len truoc khi bat dau &mdash; banh xe se quay tu do!<br>
      &#x26A0; <b>Wheel / Rotation:</b> Dat robot tren san phang, can &gt;1.5m khong gian xung quanh.
    </div>

    <div class="panel">
      <h3>Cau hinh hien tai</h3>
      <div id="cal-current-config" class="cal-config">Loading&#x2026;</div>
    </div>

    <div class="panel">
      <h3>Chon che do calibration</h3>
      <div class="cal-row">
        <button class="cal-btn" onclick="calStart('motors')">Motor Kv/Ks</button>
        <button class="cal-btn" onclick="calStart('wheel')">Wheel Diameter</button>
        <button class="cal-btn" onclick="calStart('rotation')">Rotation Radius</button>
        <button class="cal-btn" onclick="calStart('all')">Full Calibration</button>
      </div>
      <h4>Optional: Do thu cong</h4>
      <div class="cal-row">
        <span class="cal-label">Distance (m):</span>
        <input type="number" id="cal-manual-dist" class="cal-input" step="0.001" placeholder="e.g. 1.020">
        <span class="cal-label">Rotation (deg):</span>
        <input type="number" id="cal-manual-rot" class="cal-input" step="0.1" placeholder="e.g. 360.5">
      </div>
      <h4>Trang thai</h4>
      <div id="cal-status" class="cal-status idle">Idle &mdash; san sang</div>
      <div class="cal-progress-bar">
        <div id="cal-progress-fill" class="cal-progress-fill"></div>
      </div>
      <div id="cal-progress-text" class="cal-label">0%</div>
      <div class="cal-row" style="margin-top:10px;">
        <button class="cal-btn danger" onclick="calAbort()">&#x23F9; Abort</button>
        <button class="cal-btn" onclick="calRefreshConfig()">&#x1F504; Refresh Config</button>
      </div>
      <h4>Log</h4>
      <div id="cal-log" class="cal-log">(Log se hien thi o day khi chay calibration)</div>
    </div>
  </div>

  <script>
  let calPollTimer = null;

  async function calStart(mode) {
    const distInput = document.getElementById('cal-manual-dist').value;
    const rotInput  = document.getElementById('cal-manual-rot').value;
    const body = { mode };
    if (distInput && parseFloat(distInput) > 0)
      body.manual_distance = parseFloat(distInput);
    if (rotInput && parseFloat(rotInput) !== 0)
      body.manual_rotation = parseFloat(rotInput) * Math.PI / 180;

    const msgs = {
      motors:   'NHAC robot len truoc khi tiep tuc!\\nBanh xe se quay tu do. Xac nhan?',
      wheel:    'Robot se di thang ~1m.\\nDam bao co du khong gian phia truoc. Xac nhan?',
      rotation: 'Robot se xoay ~360 do.\\nDam bao co > 1m khong gian xung quanh. Xac nhan?',
      all:      'FULL CALIBRATION:\\n1. Motor (NHAC robot len!)\\n2. Wheel diameter\\n3. Rotation\\nXac nhan?',
    };
    if (!confirm(msgs[mode] || 'Bat dau calibration. Xac nhan?')) return;

    try {
      const r = await fetch('/api/calibration/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        calSetStatus('error', err.error || err.reason || 'Calibration start failed');
        document.getElementById('cal-log').textContent = err.error || err.reason || 'start failed';
        return;
      }
      calSetStatus('running', 'Dang chay...');
      if (calPollTimer) clearInterval(calPollTimer);
      calPollTimer = setInterval(calPoll, 500);
    } catch (e) { alert('Network error: ' + e.message); }
  }

  async function calAbort() {
    if (!confirm('Huy calibration? Robot se dung lai.')) return;
    try { await fetch('/api/calibration/abort', { method: 'POST' }); } catch (e) {}
  }

  async function calPoll() {
    try {
      const r = await fetch('/api/calibration/status');
      if (!r.ok) return;
      const data = await r.json();
      calSetStatus(data.status, data.message || '');
      const pct = (data.progress || 0) * 100;
      document.getElementById('cal-progress-fill').style.width = pct + '%';
      document.getElementById('cal-progress-text').textContent = pct.toFixed(1) + '%';
      if (data.log_tail && data.log_tail.length > 0) {
        const el = document.getElementById('cal-log');
        el.textContent = data.log_tail.join('\\n');
        el.scrollTop = el.scrollHeight;
      }
      if (data.status === 'done' || data.status === 'error' || data.status === 'aborted') {
        clearInterval(calPollTimer); calPollTimer = null; calRefreshConfig();
      }
    } catch (e) {}
  }

  function calSetStatus(status, message) {
    const el = document.getElementById('cal-status');
    el.className = 'cal-status ' + status;
    const labels = {
      idle: 'Idle', running: 'Dang chay',
      done: '&#x2713; Hoan tat', error: '&#x2717; Loi', aborted: '&#x23F9; Da huy',
    };
    el.innerHTML = (labels[status] || status) + (message ? ' &mdash; ' + message : '');
  }

  async function calRefreshConfig() {
    try {
      const r = await fetch('/api/calibration/config');
      if (!r.ok) return;
      const cfg = await r.json();
      let html = '';
      html += 'wheel_diameter_m  = <b>' + (cfg.wheel_diameter_m * 1000).toFixed(2) + ' mm</b>\\n';
      html += 'rotation_radius_m = <b>' + (cfg.rotation_radius_m * 1000).toFixed(2) + ' mm</b>\\n';
      html += 'last_calibrated   = ' + (cfg.calibration_time || '(never)') + '\\n';
      if (cfg.motor_params) {
        html += '\\n<b>Motor params:</b>\\n';
        for (const [wheel, dirs] of Object.entries(cfg.motor_params)) {
          for (const [d, p] of Object.entries(dirs)) {
            if (p.kv > 0)
              html += '  ' + wheel + ' ' + d + ': Kv=' + p.kv.toFixed(4) + ', Ks=' + p.ks.toFixed(2) + ', RMSE=' + p.rmse.toFixed(2) + '\\n';
          }
        }
      }
      document.getElementById('cal-current-config').innerHTML = html;
    } catch (e) {}
  }

  window.addEventListener('load', () => { calRefreshConfig(); calPoll(); });
  </script>
</body>
</html>
"""
