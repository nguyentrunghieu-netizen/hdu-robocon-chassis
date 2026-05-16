/*
 * TRAJECTORY_FOLLOW.ino
 * Robot Mecanum 4 bánh — Bám theo chuỗi waypoint tuỳ ý
 *
 * Thuật toán tiên tiến áp dụng:
 *  1. Pure Pursuit (hình học) — tính điểm nhìn (lookahead) trên đường path,
 *     tạo lệnh vận tốc tiếp tuyến và góc quay chính xác.
 *  2. Adaptive Lookahead — khoảng nhìn tỷ lệ tốc độ hiện tại (L ∝ v),
 *     giảm cut-corner ở tốc độ thấp, tăng mượt ở tốc độ cao.
 *  3. Trapezoidal Velocity Profile — tăng/giảm tốc tuyến tính dọc theo
 *     đoạn đường → không giật, bảo vệ cơ khí.
 *  4. Cross-track Error Correction — bù lệch ngang bằng PD, kết hợp Pure
 *     Pursuit để không dao động.
 *  5. Kalman Filter trên sai lệch RPM — lọc nhiễu encoder, ước lượng
 *     d(error)/dt cho D-term PID từng bánh (giữ nguyên từ file gốc).
 *  6. PID bánh xe với anti-windup, dead-zone, near-stop logic.
 *  7. Mecanum inverse kinematics — local Vx/Vy/ωz → 4 RPM bánh.
 *
 * Giao tiếp Serial (115200):
 *   TRAJ x1 y1 [th1] ; x2 y2 [th2] ; ...   — nạp quỹ đạo (cm, độ)
 *   GO                                        — chạy quỹ đạo đã nạp
 *   S                                         — dừng khẩn cấp
 *   R                                         — reset odometry
 *   STATUS                                    — in trạng thái
 *   G x y th                                  — đặt 1 target đơn (cm, độ)
 *
 * Ví dụ nạp hình chữ Z 4 điểm:
 *   TRAJ 0 0 0 ; 200 0 0 ; 0 200 0 ; 200 200 0
 *   GO
 */

#include <math.h>
#include <PS2X_lib.h>

// ==================== CHÂN PHẦN CỨNG ====================
const uint8_t M1_L = 46, M1_R = 44;   // FL
const uint8_t M2_L = 10, M2_R = 9;    // FR
const uint8_t M3_L = 6,  M3_R = 7;    // BL
const uint8_t M4_L = 11, M4_R = 12;   // BR

const uint8_t ENC1_A = 18, ENC1_B = 22;
const uint8_t ENC2_A = 19, ENC2_B = 23;
const uint8_t ENC3_A = 2,  ENC3_B = 24;
const uint8_t ENC4_A = 3,  ENC4_B = 25;

const uint8_t PS2_CLK = 50, PS2_CMD = 52, PS2_ATT = 51, PS2_DAT = 53;

// ==================== THÔNG SỐ CƠ KHÍ ====================
const float TICKS_PER_REV      = 500.0;
const float WHEEL_DIAMETER_M   = 0.100;
const float WHEEL_CIRC_M       = PI * WHEEL_DIAMETER_M;
const float ROT_RADIUS_M       = 0.300;   // bán kính quay tại chỗ

// ==================== THÔNG SỐ PID BÁNH ====================
const float RPM_MAX   = 729.0;
const float KP        = 1.0;
const float KI        = 0.8;
const float KD        = 0.05;
const float TAU_FILT  = 0.01;   // time-constant lọc ref (s)
const float Q_VAR     = 8.0;    // Kalman process noise
const float R_VAR     = 30.0;   // Kalman measurement noise
const int   MIN_PWM   = 10;

// ==================== THÔNG SỐ ĐIỀU KHIỂN VỊ TRÍ ====================
const float MAX_LINEAR_MPS     = 0.50;  // vận tốc tịnh tiến tối đa
const float MAX_ANGULAR_RAD_S  = 2.0;  // vận tốc góc tối đa

// Pure Pursuit
const float LOOKAHEAD_MIN_M    = 0.15;  // lookahead tối thiểu
const float LOOKAHEAD_MAX_M    = 0.60;  // lookahead tối đa
const float LOOKAHEAD_K        = 0.40;  // L = K * v + L_min (s)

// Trapezoidal velocity profile
const float ACCEL_MPS2         = 0.40;  // gia tốc tăng tốc (m/s²)
const float DECEL_MPS2         = 0.50;  // gia tốc giảm tốc
const float CRUISE_MPS         = 0.45;  // tốc độ hành trình

// Điều khiển heading
const float KP_TH  = 4.0;
const float KI_TH  = 0.4;
const float KD_TH  = 0.5;

// Cross-track correction
const float KP_CROSS = 2.5;
const float KD_CROSS = 0.3;

// Dung sai dừng
const float POS_TOL_M          = 0.06;   // 6 cm
const float THETA_TOL_RAD      = 0.052;  // 3°
const float SPEED_TOL_MPS      = 0.08;
const uint8_t HOLD_CYCLES      = 5;

const uint32_t CTRL_DT_MS      = 10;

// ==================== WAYPOINT ====================
#define MAX_WAYPOINTS 32

struct Waypoint {
  float x, y, theta;   // m, m, rad
};

Waypoint traj[MAX_WAYPOINTS];
uint8_t  traj_len    = 0;
uint8_t  traj_idx    = 0;   // waypoint đích hiện tại (segment end)
bool     traj_active = false;
bool     traj_done   = false;

// Điểm bắt đầu segment hiện tại (cho lookahead & cross-track)
float seg_start_x = 0.0, seg_start_y = 0.0;
// Tổng chiều dài path còn lại (cho velocity profile)
float path_remaining_m = 0.0;
// Tốc độ profile hiện tại
float profile_v_mps = 0.0;

// Trạng thái điều khiển
float I_theta      = 0.0;
float prev_cross   = 0.0;   // cross-track error trước (cho D-term)
uint8_t hold_cnt   = 0;

// ==================== ODOMETRY ====================
float odom_x   = 0.0, odom_y = 0.0, odom_th = 0.0;
float robot_vx = 0.0, robot_vy = 0.0, robot_wz = 0.0;

// ==================== ENCODER & PID ====================
volatile long enc[4] = {0, 0, 0, 0};
long   prev_enc[4]   = {0};
float  rpm[4]        = {0};
float  target_rpm[4] = {0};
float  I_term[4]     = {0};
float  filt_ref[4]   = {0};

struct KalmanD {
  float e_hat, de_hat;
  float P[2][2];
};
KalmanD kf[4];

PS2X ps2x;
bool ps2_ready = false;

// =========================================================
// TIỆN ÍCH
// =========================================================
float clampAbs(float v, float lim) {
  if (v >  lim) return  lim;
  if (v < -lim) return -lim;
  return v;
}

float wrapAngle(float a) {
  while (a >  PI) a -= 2.0f * PI;
  while (a < -PI) a += 2.0f * PI;
  return a;
}

float wheelMpsToRpm(float mps) { return mps * 60.0f / WHEEL_CIRC_M; }
float rpmToMps(float r)        { return r * WHEEL_CIRC_M / 60.0f; }

// =========================================================
// KALMAN FILTER
// =========================================================
void initKalman(KalmanD &k) {
  k.e_hat = k.de_hat = 0;
  k.P[0][0] = k.P[1][1] = 1;
  k.P[0][1] = k.P[1][0] = 0;
}

void kalmanUpdate(KalmanD &k, float e_meas, float dt) {
  k.e_hat += dt * k.de_hat;

  float dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt2 * dt2;
  float P00 = k.P[0][0] + dt * (k.P[0][1] + k.P[1][0]) + dt2 * k.P[1][1] + Q_VAR * dt4 / 4.0f;
  float P01 = k.P[0][1] + dt * k.P[1][1] + Q_VAR * dt3 / 2.0f;
  float P10 = k.P[1][0] + dt * k.P[1][1] + Q_VAR * dt3 / 2.0f;
  float P11 = k.P[1][1] + Q_VAR * dt2;
  k.P[0][0] = P00; k.P[0][1] = P01;
  k.P[1][0] = P10; k.P[1][1] = P11;

  float y = e_meas - k.e_hat;
  float S = P00 + R_VAR;
  float K0 = P00 / S, K1 = P10 / S;

  k.e_hat  += K0 * y;
  k.de_hat += K1 * y;

  k.P[0][0] = (1 - K0) * P00;
  k.P[0][1] = (1 - K0) * P01;
  k.P[1][0] = P10 - K1 * P00;
  k.P[1][1] = P11 - K1 * P01;
}

// =========================================================
// ISR ENCODER
// =========================================================
#define READ_B0 (PINA & (1 << PA0))
#define READ_B1 (PINA & (1 << PA1))
#define READ_B2 (PINA & (1 << PA2))
#define READ_B3 (PINA & (1 << PA3))
#define READ_A0 (PIND & (1 << 3))
#define READ_A1 (PIND & (1 << 2))
#define READ_A2 (PINE & (1 << 4))
#define READ_A3 (PINE & (1 << 5))

void enc_isr0() { enc[0] += ((READ_A0 ? 1 : 0) ^ (READ_B0 ? 1 : 0)) ? +1 : -1; }
void enc_isr1() { enc[1] += ((READ_A1 ? 1 : 0) ^ (READ_B1 ? 1 : 0)) ? -1 : +1; }
void enc_isr2() { enc[2] += ((READ_A2 ? 1 : 0) ^ (READ_B2 ? 1 : 0)) ? -1 : +1; }
void enc_isr3() { enc[3] += ((READ_A3 ? 1 : 0) ^ (READ_B3 ? 1 : 0)) ? +1 : -1; }

// =========================================================
// PID TỪNG BÁNH (P/I trên raw error, D qua Kalman)
// =========================================================
float pidWheel(float ref, float meas, float &integral, float &ref_f, KalmanD &k, float dt) {
  ref_f += (ref - ref_f) * dt / TAU_FILT;
  ref_f = constrain(ref_f, -RPM_MAX, RPM_MAX);

  float err = ref_f - meas;
  kalmanUpdate(k, err, dt);

  float d_term = KD * k.de_hat;
  float u_pre  = KP * err + integral + d_term;

  if (fabsf(ref) < 2.0f && fabsf(meas) < 8.0f) {
    integral *= 0.8f;
  } else {
    integral += KI * err * dt;
  }
  // Anti-windup
  if (u_pre > RPM_MAX)  integral -= KI * (u_pre - RPM_MAX) * dt;
  if (u_pre < -RPM_MAX) integral += KI * (-RPM_MAX - u_pre) * dt;
  integral = constrain(integral, -RPM_MAX, RPM_MAX);

  return constrain(KP * err + integral + d_term, -RPM_MAX, RPM_MAX);
}

// =========================================================
// MOTOR OUTPUT
// =========================================================
void setMotorLR(uint8_t L, uint8_t R, float u) {
  int pwm = (int)(fabsf(u) * 255.0f / RPM_MAX);
  pwm = constrain(pwm, 0, 255);
  if (pwm < MIN_PWM) { analogWrite(L, 0); analogWrite(R, 0); return; }
  if (u >= 0) { analogWrite(L, pwm); analogWrite(R, 0); }
  else        { analogWrite(L, 0);   analogWrite(R, pwm); }
}

void stopAllMotors() {
  for (int i = 0; i < 4; i++) { target_rpm[i] = 0; }
  analogWrite(M1_L, 0); analogWrite(M1_R, 0);
  analogWrite(M2_L, 0); analogWrite(M2_R, 0);
  analogWrite(M3_L, 0); analogWrite(M3_R, 0);
  analogWrite(M4_L, 0); analogWrite(M4_R, 0);
}

// =========================================================
// ODOMETRY
// =========================================================
void updateRPM(float dt) {
  long cur[4];
  noInterrupts();
  for (int i = 0; i < 4; i++) cur[i] = enc[i];
  interrupts();
  float k = 60.0f / (TICKS_PER_REV * dt);
  for (int i = 0; i < 4; i++) {
    long d = cur[i] - prev_enc[i];
    if (abs(d) > 10000) d = 0;
    rpm[i] = d * k;
    prev_enc[i] = cur[i];
  }
}

void updateOdometry(float dt) {
  float vfl = rpmToMps(rpm[0]), vfr = rpmToMps(rpm[1]);
  float vbl = rpmToMps(rpm[2]), vbr = rpmToMps(rpm[3]);

  float vx_l = ( vfl + vfr - vbl - vbr) * 0.25f;
  float vy_l = (-vfl + vfr - vbl + vbr) * 0.25f;
  float wz   = ( vfl - vfr - vbl + vbr) / (4.0f * ROT_RADIUS_M);

  float c = cosf(odom_th), s = sinf(odom_th);
  odom_x  += (vx_l * c - vy_l * s) * dt;
  odom_y  += (vx_l * s + vy_l * c) * dt;
  odom_th  = wrapAngle(odom_th + wz * dt);

  robot_vx = vx_l;
  robot_vy = vy_l;
  robot_wz = wz;
}

void resetOdom() {
  noInterrupts();
  for (int i = 0; i < 4; i++) enc[i] = 0;
  interrupts();
  for (int i = 0; i < 4; i++) {
    prev_enc[i] = 0; rpm[i] = 0;
    target_rpm[i] = 0; I_term[i] = 0; filt_ref[i] = 0;
    initKalman(kf[i]);
  }
  odom_x = odom_y = odom_th = 0;
  robot_vx = robot_vy = robot_wz = 0;
  I_theta = prev_cross = 0;
  hold_cnt = 0;
  profile_v_mps = 0;
}

// =========================================================
// TRAJECTORY MANAGEMENT
// =========================================================

// Tính tổng độ dài path từ điểm hiện tại đến cuối
float calcPathRemaining() {
  float total = 0;
  float px = odom_x, py = odom_y;
  for (uint8_t i = traj_idx; i < traj_len; i++) {
    float dx = traj[i].x - px;
    float dy = traj[i].y - py;
    total += sqrtf(dx * dx + dy * dy);
    px = traj[i].x;
    py = traj[i].y;
  }
  return total;
}

void startTrajectory() {
  if (traj_len == 0) { Serial.println("ERR: no waypoints"); return; }
  traj_idx = 0;
  traj_active = true;
  traj_done   = false;
  hold_cnt    = 0;
  I_theta     = 0;
  prev_cross  = 0;
  profile_v_mps = 0;
  seg_start_x = odom_x;
  seg_start_y = odom_y;
  path_remaining_m = calcPathRemaining();

  Serial.print("TRAJ START: "); Serial.print(traj_len);
  Serial.print(" wps, dist "); Serial.print(path_remaining_m, 2); Serial.println(" m");
}

void stopTrajectory() {
  traj_active = false;
  I_theta = prev_cross = 0;
  hold_cnt = 0;
  profile_v_mps = 0;
  stopAllMotors();
  Serial.println("TRAJ STOP");
}

void advanceWaypoint() {
  // Ghi điểm đích vừa qua làm seg_start mới
  seg_start_x = traj[traj_idx].x;
  seg_start_y = traj[traj_idx].y;
  I_theta = 0;
  prev_cross = 0;
  hold_cnt = 0;

  traj_idx++;
  if (traj_idx >= traj_len) {
    traj_active = false;
    traj_done   = true;
    profile_v_mps = 0;
    stopAllMotors();
    Serial.println("TRAJ DONE");
    return;
  }
  Serial.print("WP "); Serial.print(traj_idx + 1);
  Serial.print("/"); Serial.print(traj_len);
  Serial.print(" -> ("); Serial.print(traj[traj_idx].x, 2);
  Serial.print(", "); Serial.print(traj[traj_idx].y, 2); Serial.println(")");
}

// =========================================================
// PURE PURSUIT + VELOCITY PROFILE
// =========================================================
/*
 * Pure Pursuit cho robot holonomic:
 *  - Tính lookahead point trên đường thẳng segment hiện tại.
 *  - Nếu lookahead vượt qua điểm đích → dùng chính điểm đích.
 *  - Tính lệnh Vx, Vy trong frame robot hướng đến lookahead.
 *  - Kết hợp PD heading để giữ theta đúng.
 *  - Velocity profile: tăng tốc + hãm trapezoidal theo path_remaining.
 */
void computeWheelRPM(float dt) {
  if (!traj_active) {
    for (int i = 0; i < 4; i++) target_rpm[i] = 0;
    return;
  }

  Waypoint &wp = traj[traj_idx];
  float dx = wp.x - odom_x;
  float dy = wp.y - odom_y;
  float dist_to_wp = sqrtf(dx * dx + dy * dy);

  // ---- Trapezoidal velocity profile ----
  // path_remaining ước lượng theo dist_to_wp cộng các đoạn sau
  float dist_after = 0;
  {
    float px = wp.x, py = wp.y;
    for (uint8_t i = traj_idx + 1; i < traj_len; i++) {
      float ex = traj[i].x - px, ey = traj[i].y - py;
      dist_after += sqrtf(ex * ex + ey * ey);
      px = traj[i].x; py = traj[i].y;
    }
  }
  path_remaining_m = dist_to_wp + dist_after;

  // Tốc độ tối đa cho phép theo gia tốc (không vượt CRUISE_MPS)
  float v_accel = profile_v_mps + ACCEL_MPS2 * dt;
  // Tốc độ tối đa để kịp hãm dừng ở cuối path (v² = 2*a*s)
  float v_decel = sqrtf(2.0f * DECEL_MPS2 * path_remaining_m + 1e-6f);
  float v_target = min(min(v_accel, v_decel), CRUISE_MPS);
  v_target = max(v_target, 0.0f);

  // Giảm tốc khi gần waypoint trung gian (lookahead sẽ "nhìn qua" wp tiếp)
  float v_max_seg = MAX_LINEAR_MPS;
  if (dist_to_wp < 0.25f && traj_idx < traj_len - 1) {
    v_max_seg = MAX_LINEAR_MPS * (dist_to_wp / 0.25f);
    if (v_max_seg < 0.05f) v_max_seg = 0.05f;
  }
  v_target = min(v_target, v_max_seg);
  profile_v_mps = v_target;

  // ---- Adaptive lookahead ----
  float speed_cur = sqrtf(robot_vx * robot_vx + robot_vy * robot_vy);
  float L = LOOKAHEAD_K * speed_cur + LOOKAHEAD_MIN_M;
  L = constrain(L, LOOKAHEAD_MIN_M, LOOKAHEAD_MAX_M);

  // ---- Pure Pursuit: tìm lookahead point trên segment ----
  // Segment: seg_start → wp
  float seg_dx = wp.x - seg_start_x;
  float seg_dy = wp.y - seg_start_y;
  float seg_len = sqrtf(seg_dx * seg_dx + seg_dy * seg_dy);

  float lx, ly;  // lookahead point (global)

  if (seg_len < 0.01f) {
    // Đoạn quá ngắn → dùng thẳng điểm đích
    lx = wp.x; ly = wp.y;
  } else {
    // Chiếu robot lên đường thẳng segment
    float t = ((odom_x - seg_start_x) * seg_dx + (odom_y - seg_start_y) * seg_dy) / (seg_len * seg_len);
    t = constrain(t, 0.0f, 1.0f);
    float proj_x = seg_start_x + t * seg_dx;
    float proj_y = seg_start_y + t * seg_dy;

    // Lookahead = proj + L dọc theo đường
    float remain = (1.0f - t) * seg_len;  // khoảng còn lại trên segment
    if (L >= remain) {
      // Lookahead vượt qua điểm đích → kẹp tại điểm đích
      lx = wp.x; ly = wp.y;
    } else {
      float unit_x = seg_dx / seg_len;
      float unit_y = seg_dy / seg_len;
      lx = proj_x + L * unit_x;
      ly = proj_y + L * unit_y;
    }
  }

  // ---- Cross-track error (signed distance từ robot đến đường segment) ----
  float cross_err = 0;
  if (seg_len > 0.02f) {
    float rx = odom_x - seg_start_x;
    float ry = odom_y - seg_start_y;
    cross_err = (ry * seg_dx - rx * seg_dy) / seg_len;
  }
  float d_cross = (cross_err - prev_cross) / dt;
  prev_cross = cross_err;

  // ---- Vector từ robot đến lookahead (global) ----
  float goal_dx = lx - odom_x;
  float goal_dy = ly - odom_y;
  float goal_d  = sqrtf(goal_dx * goal_dx + goal_dy * goal_dy);

  float c = cosf(odom_th), s = sinf(odom_th);

  // Chuyển sang local frame của robot
  float goal_lx, goal_ly;
  if (goal_d > 0.005f) {
    float ux = goal_dx / goal_d, uy = goal_dy / goal_d;
    goal_lx = ux * c + uy * s;
    goal_ly = -ux * s + uy * c;
  } else {
    goal_lx = 0; goal_ly = 0;
  }

  // Lệnh vận tốc tịnh tiến (local frame), tỷ lệ theo profile
  float vx_cmd = v_target * goal_lx;
  float vy_cmd = v_target * goal_ly;

  // Cross-track correction trong global → local
  if (seg_len > 0.02f) {
    float corr_gx =  KP_CROSS * cross_err * (seg_dy / seg_len) - KD_CROSS * d_cross * (seg_dy / seg_len);
    float corr_gy = -KP_CROSS * cross_err * (seg_dx / seg_len) + KD_CROSS * d_cross * (seg_dx / seg_len);
    vx_cmd += clampAbs( corr_gx * c + corr_gy * s, 0.15f);
    vy_cmd += clampAbs(-corr_gx * s + corr_gy * c, 0.15f);
  }

  vx_cmd = clampAbs(vx_cmd, MAX_LINEAR_MPS);
  vy_cmd = clampAbs(vy_cmd, MAX_LINEAR_MPS);

  // ---- Heading PID (đặt theta của waypoint đích) ----
  float dtheta = wrapAngle(wp.theta - odom_th);
  I_theta += KI_TH * dtheta * dt;
  I_theta = clampAbs(I_theta, MAX_ANGULAR_RAD_S * 0.4f);
  float wz_cmd = clampAbs(KP_TH * dtheta + I_theta - KD_TH * robot_wz, MAX_ANGULAR_RAD_S);

  // ---- Kiểm tra đến waypoint ----
  float speed = sqrtf(robot_vx * robot_vx + robot_vy * robot_vy);
  bool at_pos   = dist_to_wp < POS_TOL_M;
  bool at_angle = fabsf(dtheta) < THETA_TOL_RAD;
  bool stopped  = speed < SPEED_TOL_MPS;

  // Với waypoint trung gian: chỉ cần at_pos (không cần dừng hẳn)
  bool is_last = (traj_idx == traj_len - 1);
  bool arrived = is_last ? (at_pos && at_angle && stopped) : at_pos;

  if (arrived) {
    hold_cnt++;
    if (hold_cnt >= HOLD_CYCLES) {
      advanceWaypoint();
      return;
    }
  } else {
    hold_cnt = 0;
  }

  // ---- Mecanum inverse kinematics ----
  float ws[4];
  ws[0] =  vx_cmd - vy_cmd + ROT_RADIUS_M * wz_cmd;   // FL
  ws[1] =  vx_cmd + vy_cmd - ROT_RADIUS_M * wz_cmd;   // FR
  ws[2] = -vx_cmd - vy_cmd - ROT_RADIUS_M * wz_cmd;   // BL  (đảo dấu x,y)
  ws[3] = -vx_cmd + vy_cmd + ROT_RADIUS_M * wz_cmd;   // BR

  // Scale nếu vượt RPM_MAX
  float max_r = 0;
  for (int i = 0; i < 4; i++) {
    float ar = fabsf(wheelMpsToRpm(ws[i]));
    if (ar > max_r) max_r = ar;
  }
  float scale = (max_r > RPM_MAX) ? RPM_MAX / max_r : 1.0f;
  for (int i = 0; i < 4; i++) target_rpm[i] = wheelMpsToRpm(ws[i] * scale);
}

// =========================================================
// SERIAL COMMAND PARSER
// =========================================================
static bool nextFloat(const char *&p, float &out) {
  char *end;
  while (*p == ' ' || *p == '\t') p++;
  out = (float)strtod(p, &end);
  if (end == p) return false;
  p = end;
  return true;
}

/*
 * TRAJ x1 y1 [th1] ; x2 y2 [th2] ; ...
 * Toạ độ cm, góc độ (tùy chọn, mặc định 0)
 */
void parseTrajCommand(const char *buf) {
  traj_len = 0;
  const char *p = buf;
  while (*p && traj_len < MAX_WAYPOINTS) {
    while (*p == ' ' || *p == '\t') p++;
    float x, y, th = 0;
    if (!nextFloat(p, x)) break;
    if (!nextFloat(p, y)) break;
    // theta tùy chọn: nếu ký tự tiếp theo là ';' hoặc cuối chuỗi thì bỏ qua
    const char *q = p;
    while (*q == ' ' || *q == '\t') q++;
    if (*q != ';' && *q != '\0') {
      float tmp;
      if (nextFloat(p, tmp)) th = tmp;
    }
    traj[traj_len].x     = x / 100.0f;       // cm → m
    traj[traj_len].y     = y / 100.0f;
    traj[traj_len].theta = wrapAngle(th * PI / 180.0f);
    traj_len++;
    // Bỏ qua dấu ';'
    while (*p == ' ' || *p == '\t') p++;
    if (*p == ';') p++;
  }
  Serial.print("TRAJ LOADED: "); Serial.print(traj_len); Serial.println(" waypoints");
  for (uint8_t i = 0; i < traj_len; i++) {
    Serial.print("  WP"); Serial.print(i + 1);
    Serial.print(" ("); Serial.print(traj[i].x * 100.0f, 1);
    Serial.print("cm, "); Serial.print(traj[i].y * 100.0f, 1);
    Serial.print("cm, "); Serial.print(traj[i].theta * 180.0f / PI, 1);
    Serial.println(" deg)");
  }
}

void handleSerial() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  if (line.equalsIgnoreCase("S") || line.equalsIgnoreCase("STOP")) {
    stopTrajectory();
    return;
  }
  if (line.equalsIgnoreCase("R") || line.equalsIgnoreCase("RESET")) {
    stopTrajectory();
    resetOdom();
    Serial.println("ODOM RESET");
    return;
  }
  if (line.equalsIgnoreCase("GO")) {
    startTrajectory();
    return;
  }
  if (line.equalsIgnoreCase("STATUS")) {
    Serial.print("ODOM: "); Serial.print(odom_x * 100, 1); Serial.print("cm ");
    Serial.print(odom_y * 100, 1); Serial.print("cm ");
    Serial.print(odom_th * 180.0f / PI, 1); Serial.println("deg");
    Serial.print("TRAJ: len="); Serial.print(traj_len);
    Serial.print(" idx="); Serial.print(traj_idx);
    Serial.print(" active="); Serial.println(traj_active ? "YES" : "NO");
    return;
  }
  // TRAJ ...
  if (line.startsWith("TRAJ ") || line.startsWith("traj ")) {
    parseTrajCommand(line.c_str() + 5);
    return;
  }
  // G x_cm y_cm theta_deg — điểm đơn
  if (line.startsWith("G ") || line.startsWith("g ")) {
    const char *p = line.c_str() + 2;
    float x, y, th = 0;
    if (nextFloat(p, x) && nextFloat(p, y)) {
      nextFloat(p, th);
      traj[0] = { x / 100.0f, y / 100.0f, wrapAngle(th * PI / 180.0f) };
      traj_len = 1;
      startTrajectory();
    } else {
      Serial.println("ERR: G x_cm y_cm [th_deg]");
    }
    return;
  }
  Serial.println("CMD: TRAJ x1 y1 [th1] ; x2 y2 ... | GO | S | R | STATUS | G x y [th]");
}

// =========================================================
// GAMEPAD
// =========================================================
void handleGamepad() {
  if (!ps2_ready) return;
  ps2x.read_gamepad(false, 0);
  if (ps2x.ButtonPressed(PSB_START)) {
    if (traj_len > 0) startTrajectory();
    else Serial.println("No trajectory loaded. Use TRAJ command first.");
  }
  if (ps2x.ButtonPressed(PSB_SELECT)) {
    stopTrajectory();
  }
}

// =========================================================
// SETUP
// =========================================================
void setup() {
  Serial.begin(115200);

  byte err = ps2x.config_gamepad(PS2_CLK, PS2_CMD, PS2_ATT, PS2_DAT, false, true);
  if (err == 0) { ps2_ready = true; Serial.println("PS2 OK"); }
  else          { Serial.print("PS2 ERR: "); Serial.println(err); }

  // Motor pins
  pinMode(M1_L, OUTPUT); pinMode(M1_R, OUTPUT);
  pinMode(M2_L, OUTPUT); pinMode(M2_R, OUTPUT);
  pinMode(M3_L, OUTPUT); pinMode(M3_R, OUTPUT);
  pinMode(M4_L, OUTPUT); pinMode(M4_R, OUTPUT);

  // Encoder pins
  pinMode(ENC1_A, INPUT_PULLUP); pinMode(ENC1_B, INPUT_PULLUP);
  pinMode(ENC2_A, INPUT_PULLUP); pinMode(ENC2_B, INPUT_PULLUP);
  pinMode(ENC3_A, INPUT_PULLUP); pinMode(ENC3_B, INPUT_PULLUP);
  pinMode(ENC4_A, INPUT_PULLUP); pinMode(ENC4_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENC1_A), enc_isr0, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC2_A), enc_isr1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC3_A), enc_isr2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC4_A), enc_isr3, CHANGE);

  for (int i = 0; i < 4; i++) initKalman(kf[i]);
  resetOdom();

  Serial.println("=== TRAJECTORY FOLLOWER READY ===");
  Serial.println("Algorithms: Pure Pursuit + Adaptive Lookahead + Trapezoidal Velocity Profile");
  Serial.println("  + Cross-track PD + Heading PID + Kalman D-term per wheel");
  Serial.println("CMD: TRAJ x1 y1 [th1] ; x2 y2 ... | GO | S | R | STATUS | G x y [th]");
  Serial.println("Example (square 2m): TRAJ 0 0 0 ; 200 0 0 ; 200 200 0 ; 0 200 0 ; 0 0 0");
}

// =========================================================
// MAIN LOOP
// =========================================================
void loop() {
  static uint32_t last_ctrl = 0;
  static uint32_t last_dbg  = 0;

  handleGamepad();
  handleSerial();

  uint32_t now = millis();
  if (now - last_ctrl < CTRL_DT_MS) return;
  float dt = CTRL_DT_MS / 1000.0f;
  last_ctrl = now;

  updateRPM(dt);
  updateOdometry(dt);
  computeWheelRPM(dt);

  float cmd[4];
  for (int i = 0; i < 4; i++) {
    cmd[i] = pidWheel(target_rpm[i], rpm[i], I_term[i], filt_ref[i], kf[i], dt);
  }
  setMotorLR(M1_L, M1_R, cmd[0]);
  setMotorLR(M2_L, M2_R, cmd[1]);
  setMotorLR(M3_L, M3_R, cmd[2]);
  setMotorLR(M4_L, M4_R, cmd[3]);

  // Debug 5 Hz
  if (now - last_dbg >= 200) {
    last_dbg = now;
    Serial.print("X="); Serial.print(odom_x * 100, 1);
    Serial.print("cm Y="); Serial.print(odom_y * 100, 1);
    Serial.print("cm TH="); Serial.print(odom_th * 180.0f / PI, 1);
    Serial.print("deg | WP="); Serial.print(traj_active ? traj_idx + 1 : 0);
    Serial.print("/"); Serial.print(traj_len);
    Serial.print(" v="); Serial.print(profile_v_mps, 2);
    Serial.print("m/s path="); Serial.print(path_remaining_m, 2); Serial.println("m");
  }
}
