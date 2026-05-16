#include <math.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <PS2X_lib.h>

// ==================== CẤU HÌNH CHÂN ====================
const uint8_t M1_L = 46, M1_R = 44;   // FL
const uint8_t M2_L = 10, M2_R = 9;    // FR
const uint8_t M3_L = 6,  M3_R = 7;    // BL
const uint8_t M4_L = 11, M4_R = 12;   // BR

const uint8_t ENC1_A = 18, ENC1_B = 22;  // FL
const uint8_t ENC2_A = 19, ENC2_B = 23;  // FR
const uint8_t ENC3_A = 2,  ENC3_B = 24;  // BL
const uint8_t ENC4_A = 3,  ENC4_B = 25;  // BR

const uint8_t PS2_CLK = 50, PS2_CMD = 52, PS2_ATT = 51, PS2_DAT = 53;

// ==================== [IMU] CẤU HÌNH BNO085 ====================
// BNO085 nối: SDA -> pin 20, SCL -> pin 21, VCC -> 3.3V, GND -> GND
#define BNO08X_RESET      -1
#define BNO08X_ADDR_LOW   0x4A
#define BNO08X_ADDR_HIGH  0x4B
const unsigned long IMU_REPORT_INTERVAL_US = 20000;   // 50 Hz, nhẹ hơn và ổn định hơn trên Mega
const uint32_t IMU_I2C_CLOCK_HZ = 100000;
const uint8_t IMU_MAX_EVENTS_PER_UPDATE = 2;
const bool ENABLE_IMU_GYRO = true;
// Nếu khi robot quay trái (CCW) mà yaw GIẢM thay vì TĂNG, đổi sang -1.0f
const float IMU_YAW_SIGN = +1.0f;

// ==================== THÔNG SỐ CƠ KHÍ ====================
const float TICKS_PER_REV = 500.0;
const float WHEEL_DIAMETER_M = 0.100;
const float WHEEL_CIRCUMFERENCE_M = PI * WHEEL_DIAMETER_M;
const float ROTATION_RADIUS_M = 0.300;

// ==================== THÔNG SỐ ĐIỀU KHIỂN ====================
const float RPM_MAX = 729.0;
const float TAU_P_FILT = 0.01;
const float TAU_WHEEL_MEAS = 0.04;
const float Q_var = 8.0;
const float R_var = 30.0;
const uint32_t CTRL_DT_MS = 10;

const float WHEEL_KP_PWM_PER_RPM = 0.20f;
const float WHEEL_KI_PWM_PER_RPM_S = 0.75f;
const float WHEEL_KV_PWM_PER_RPM = 255.0f / RPM_MAX;
const float WHEEL_KS_PWM = 22.0f;
const float WHEEL_I_LIMIT_PWM = 110.0f;
const float WHEEL_REF_ZERO_RPM = 12.0f;
const float WHEEL_MEAS_ZERO_RPM = 10.0f;
const float PWM_ZERO_DB = 14.0f;

const float KP_X = 1.5;
const float KP_Y = 1.5;
const float KP_TH = 4.0;
const float KI_TH = 0.5;
const float KD_POS = 0.3;
const float KD_THETA = 0.5;
const float KP_CROSS = 3.0;
const float MAX_LINEAR_MPS = 0.50;
const float MAX_ANGULAR_RAD_S = 2.0;
const float POS_TOL_M = 0.02;
const float SQUARE_WP_SWITCH_RADIUS_M = 0.06;
const float THETA_TOL_RAD = 3.0 * PI / 180.0;
const float SPEED_TOL_MPS = 0.08;
const float DECEL_RADIUS_M = 0.50;
const uint16_t TARGET_HOLD_CYCLES = 5;
const float TARGET_RPM_SLEW_RPM_PER_S = 1800.0f;

const bool AUTO_START_DEMO = true;
const uint32_t AUTO_START_DELAY_MS = 3000;   // Tăng lên 3s để BNO085 ổn định trước khi chạy
const float SQUARE_SIDE_M = 2.0;
const uint8_t SQUARE_WP_COUNT = 4;
const float SQUARE_WP[4][3] = {
  {  1.5,  0.0,  0.0 },
  {  1.5,  1.5,  0.0 },
  {  0.0,  1.5,  0.0 },
  {  0.0,  0.0,  0.0 },
};

// ==================== BIẾN TOÀN CỤC ====================
volatile long enc[4] = {0, 0, 0, 0};
long prev_enc[4] = {0, 0, 0, 0};

float rpm[4] = {0};
float rpm_filt[4] = {0};
float target_rpm[4] = {0};
float I_term[4] = {0};
float filt_ref[4] = {0};

float odom_x_m = 0.0;
float odom_y_m = 0.0;
float odom_theta_rad = 0.0;
float robot_vx_mps = 0.0;
float robot_vy_mps = 0.0;
float robot_wz_rad_s = 0.0;

struct MotionTarget {
  float x_m;
  float y_m;
  float theta_rad;
  bool active;
  bool finished;
};

MotionTarget motion = {0.0, 0.0, 0.0, false, false};
uint16_t target_hold_counter = 0;
bool has_stored_target = false;
float prev_wp_x = 0.0, prev_wp_y = 0.0;
float I_theta = 0.0;

uint8_t square_wp_idx = 0;
bool square_running = false;
bool square_finished = false;

PS2X ps2x;
bool ps2_ready = false;

// ==================== [IMU] BIẾN TOÀN CỤC BNO085 ====================
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t imu_value;

bool  imu_connected = false;     // true nếu IMU kết nối OK
bool  imu_has_yaw = false;       // true khi đã có yaw lần đầu
bool  imu_has_gyro = false;      // true khi đã có gyro lần đầu
bool  imu_pending_offset = true; // cần chụp offset yaw lần tới
float imu_yaw_raw_rad = 0.0f;    // yaw nguyên gốc từ cảm biến (đã nhân SIGN)
float imu_yaw_offset_rad = 0.0f; // offset chụp lúc reset (gốc tọa độ)
float imu_yaw_rad = 0.0f;        // yaw tương đối dùng cho odometry
float imu_wz_rad_s = 0.0f;       // vận tốc góc quanh trục Z từ gyro

// ==================== TIỆN ÍCH ====================
float clampAbs(float value, float limit) {
  if (value > limit) return limit;
  if (value < -limit) return -limit;
  return value;
}

float wrapAngle(float angle_rad) {
  while (angle_rad > PI) angle_rad -= 2.0 * PI;
  while (angle_rad < -PI) angle_rad += 2.0 * PI;
  return angle_rad;
}

float rpmToWheelMps(float wheel_rpm) {
  return wheel_rpm * WHEEL_CIRCUMFERENCE_M / 60.0;
}

float wheelMpsToRpm(float wheel_mps) {
  return wheel_mps * 60.0 / WHEEL_CIRCUMFERENCE_M;
}

float ticksToMeters(long ticks) {
  return ticks * (WHEEL_CIRCUMFERENCE_M / TICKS_PER_REV);
}

float signNonZero(float value) {
  if (value > 0.0f) return 1.0f;
  if (value < 0.0f) return -1.0f;
  return 0.0f;
}

float slewToward(float current, float target, float maxDelta) {
  float delta = constrain(target - current, -maxDelta, maxDelta);
  return current + delta;
}

// ==================== [IMU] CÁC HÀM HỖ TRỢ BNO085 ====================
// Chuyển quaternion sang yaw (góc quay quanh trục Z), kết quả tính theo radian.
// Công thức ZYX Euler: yaw = atan2(2(qw*qz + qx*qy), 1 - 2(qy^2 + qz^2))
static float quatToYaw(float qw, float qx, float qy, float qz) {
  return atan2(2.0f * (qw * qz + qx * qy),
               1.0f - 2.0f * (qy * qy + qz * qz));
}

// Bật các report cần thiết trên BNO085:
// - GAME_ROTATION_VECTOR: quaternion tự fusion, không dùng từ trường -> ổn định
// - GYROSCOPE_CALIBRATED: vận tốc góc quanh trục Z (rad/s), dùng làm wz
void setIMUReports() {
  if (!bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, IMU_REPORT_INTERVAL_US)) {
    Serial.println("[IMU] Khong bat duoc game rotation vector");
  }
  if (ENABLE_IMU_GYRO && !bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, IMU_REPORT_INTERVAL_US)) {
    Serial.println("[IMU] Khong bat duoc gyroscope");
  }
}

// Thử kết nối BNO085 ở cả 2 địa chỉ 0x4A và 0x4B
bool beginIMU() {
  Serial.println("[IMU] Thu ket noi BNO085 tai 0x4A...");
  if (bno08x.begin_I2C(BNO08X_ADDR_LOW, &Wire)) {
    Serial.println("[IMU] Da ket noi BNO085 tai 0x4A");
    return true;
  }
  Serial.println("[IMU] Thu ket noi BNO085 tai 0x4B...");
  if (bno08x.begin_I2C(BNO08X_ADDR_HIGH, &Wire)) {
    Serial.println("[IMU] Da ket noi BNO085 tai 0x4B");
    return true;
  }
  return false;
}

// Đọc và xử lý các sự kiện mới từ BNO085.
// Gọi hàm này càng nhanh càng tốt trong loop để không bị tràn hàng đợi.
void updateIMU() {
  if (!imu_connected) return;

  // Nếu chip bị reset (do nhiễu nguồn, ...) thì bật lại report
  if (bno08x.wasReset()) {
    Serial.println("[IMU] BNO085 vua reset, bat lai reports");
    setIMUReports();
    imu_pending_offset = true;
    imu_has_yaw = false;
    imu_has_gyro = false;
  }

  for (uint8_t eventsRead = 0; eventsRead < IMU_MAX_EVENTS_PER_UPDATE; ++eventsRead) {
    if (!bno08x.getSensorEvent(&imu_value)) break;

    switch (imu_value.sensorId) {
      case SH2_GAME_ROTATION_VECTOR: {
        float yaw = quatToYaw(
          imu_value.un.gameRotationVector.real,
          imu_value.un.gameRotationVector.i,
          imu_value.un.gameRotationVector.j,
          imu_value.un.gameRotationVector.k);
        imu_yaw_raw_rad = IMU_YAW_SIGN * yaw;

        // Lần đầu có yaw sau reset -> chụp offset để yaw tương đối = 0
        if (imu_pending_offset) {
          imu_yaw_offset_rad = imu_yaw_raw_rad;
          imu_pending_offset = false;
          Serial.print("[IMU] Chup offset yaw = ");
          Serial.print(imu_yaw_offset_rad * 180.0 / PI, 2);
          Serial.println(" deg");
        }

        imu_yaw_rad = wrapAngle(imu_yaw_raw_rad - imu_yaw_offset_rad);
        imu_has_yaw = true;
        break;
      }

      case SH2_GYROSCOPE_CALIBRATED:
        if (ENABLE_IMU_GYRO) {
          imu_wz_rad_s = IMU_YAW_SIGN * imu_value.un.gyroscope.z;
          imu_has_gyro = true;
        }
        break;
    }
  }
}

// Yêu cầu IMU chụp offset yaw hiện tại làm "hướng 0"
// -> sau khi gọi, imu_yaw_rad ~ 0 khi robot đứng yên
void captureIMUYawOffset() {
  if (imu_has_yaw) {
    imu_yaw_offset_rad = imu_yaw_raw_rad;
    imu_yaw_rad = 0.0f;
  } else {
    // Chưa có yaw -> đánh dấu chờ, sẽ chụp ở lần reading tiếp theo
    imu_pending_offset = true;
  }
}

// ==================== ĐIỀU KHIỂN QUỸ ĐẠO ====================
void stopMotion() {
  motion.active = false;
  motion.finished = true;
  target_hold_counter = 0;
  I_theta = 0.0;
  for (int i = 0; i < 4; i++) {
    target_rpm[i] = 0.0;
    I_term[i] = 0.0f;
    filt_ref[i] = 0.0f;
  }
  hardStopAllMotors();
}

void setTargetPose(float x_m, float y_m, float theta_rad) {
  prev_wp_x = odom_x_m;
  prev_wp_y = odom_y_m;
  I_theta = 0.0;
  motion.x_m = x_m;
  motion.y_m = y_m;
  motion.theta_rad = wrapAngle(theta_rad);
  motion.active = true;
  motion.finished = false;
  target_hold_counter = 0;
  has_stored_target = true;
}

bool startStoredTarget() {
  if (!has_stored_target) {
    startSquare();
    return true;
  }
  setTargetPose(motion.x_m, motion.y_m, motion.theta_rad);
  return true;
}

void startSquare() {
  square_wp_idx = 0;
  square_running = true;
  square_finished = false;
  setTargetPose(SQUARE_WP[0][0], SQUARE_WP[0][1], SQUARE_WP[0][2]);
  Serial.print("SQUARE START: WP 1/");
  Serial.println(SQUARE_WP_COUNT);
}

void advanceSquareWaypoint() {
  if (!square_running || square_finished) return;

  prev_wp_x = SQUARE_WP[square_wp_idx][0];
  prev_wp_y = SQUARE_WP[square_wp_idx][1];
  I_theta = 0.0;

  square_wp_idx++;
  if (square_wp_idx >= SQUARE_WP_COUNT) {
    square_running = false;
    square_finished = true;
    stopMotion();
    Serial.println("SQUARE DONE -- da hoan thanh hinh vuong!");
    return;
  }

  target_hold_counter = 0;
  motion.x_m = SQUARE_WP[square_wp_idx][0];
  motion.y_m = SQUARE_WP[square_wp_idx][1];
  motion.theta_rad = wrapAngle(SQUARE_WP[square_wp_idx][2]);
  motion.active = true;
  motion.finished = false;

  Serial.print("SQUARE WP ");
  Serial.print(square_wp_idx + 1);
  Serial.print("/");
  Serial.print(SQUARE_WP_COUNT);
  Serial.print(" -> (");
  Serial.print(SQUARE_WP[square_wp_idx][0], 2);
  Serial.print(", ");
  Serial.print(SQUARE_WP[square_wp_idx][1], 2);
  Serial.println(")");
}

void resetOdometry() {
  noInterrupts();
  for (int i = 0; i < 4; i++) enc[i] = 0;
  interrupts();

  for (int i = 0; i < 4; i++) prev_enc[i] = 0;

  odom_x_m = 0.0;
  odom_y_m = 0.0;
  odom_theta_rad = 0.0;
  robot_vx_mps = 0.0;
  robot_vy_mps = 0.0;
  robot_wz_rad_s = 0.0;

  for (int i = 0; i < 4; i++) {
    rpm[i] = 0.0;
    rpm_filt[i] = 0.0;
    target_rpm[i] = 0.0;
    filt_ref[i] = 0.0;
    I_term[i] = 0.0;
  }

  target_hold_counter = 0;
  motion.active = false;
  motion.finished = false;
  prev_wp_x = 0.0;
  prev_wp_y = 0.0;
  I_theta = 0.0;

  // [IMU] Chụp lại offset yaw -> hướng hiện tại = 0 rad
  captureIMUYawOffset();
}

void setMotorSignedPwm(uint8_t pinL, uint8_t pinR, float pwmSigned) {
  float pwmAbs = fabs(pwmSigned);
  if (pwmAbs < PWM_ZERO_DB) {
    analogWrite(pinL, 0);
    analogWrite(pinR, 0);
    return;
  }

  int pwm = (int)constrain(pwmAbs, 0.0f, 255.0f);
  if (pwmSigned >= 0.0f) {
    analogWrite(pinL, pwm);
    analogWrite(pinR, 0);
  } else {
    analogWrite(pinL, 0);
    analogWrite(pinR, pwm);
  }
}

void applyMotorPwm(float pwm0, float pwm1, float pwm2, float pwm3) {
  setMotorSignedPwm(M1_L, M1_R, pwm0);
  setMotorSignedPwm(M2_L, M2_R, pwm1);
  setMotorSignedPwm(M3_L, M3_R, pwm2);
  setMotorSignedPwm(M4_L, M4_R, pwm3);
}

void hardStopAllMotors() {
  applyMotorPwm(0.0f, 0.0f, 0.0f, 0.0f);

  noInterrupts();
  for (int i = 0; i < 4; ++i) {
    prev_enc[i] = enc[i];
  }
  interrupts();

  for (int i = 0; i < 4; ++i) {
    rpm[i] = 0.0f;
    rpm_filt[i] = 0.0f;
    target_rpm[i] = 0.0f;
    I_term[i] = 0.0f;
    filt_ref[i] = 0.0f;
  }
}

// ==================== ISR ENCODER ====================
#define READ_B0 (PINA & (1 << PA0))
#define READ_B1 (PINA & (1 << PA1))
#define READ_B2 (PINA & (1 << PA2))
#define READ_B3 (PINA & (1 << PA3))

#define READ_A0 (PIND & (1 << 3))
#define READ_A1 (PIND & (1 << 2))
#define READ_A2 (PINE & (1 << 4))
#define READ_A3 (PINE & (1 << 5))

void enc_isr0() { enc[0] += (READ_A0 ? 1 : 0) ^ (READ_B0 ? 1 : 0) ? +1 : -1; }
void enc_isr1() { enc[1] += (READ_A1 ? 1 : 0) ^ (READ_B1 ? 1 : 0) ? -1 : +1; }
void enc_isr2() { enc[2] += (READ_A2 ? 1 : 0) ^ (READ_B2 ? 1 : 0) ? -1 : +1; }
void enc_isr3() { enc[3] += (READ_A3 ? 1 : 0) ^ (READ_B3 ? 1 : 0) ? +1 : -1; }

// ==================== PID TỪNG BÁNH ====================
float PID_WHEEL_RPM(float rpm_ref, float rpm_meas, float &integral, float &ref_filter, float dt) {

  ref_filter += (rpm_ref - ref_filter) * dt / TAU_P_FILT;
  ref_filter = constrain(ref_filter, -RPM_MAX, RPM_MAX);

  if (fabs(ref_filter) < WHEEL_REF_ZERO_RPM && fabs(rpm_meas) < WHEEL_MEAS_ZERO_RPM) {
    integral = 0.0f;
    ref_filter = 0.0f;
    return 0.0f;
  }

  float err = ref_filter - rpm_meas;
  float uFF = WHEEL_KV_PWM_PER_RPM * ref_filter;
  if (fabs(ref_filter) > WHEEL_REF_ZERO_RPM) {
    uFF += signNonZero(ref_filter) * WHEEL_KS_PWM;
  }

  float p = WHEEL_KP_PWM_PER_RPM * err;
  float uPre = uFF + p + integral;
  float uSat = constrain(uPre, -255.0f, 255.0f);

  bool allowIntegrate = (fabs(uPre - uSat) < 0.5f) ||
                        ((uPre > uSat) && (err < 0.0f)) ||
                        ((uPre < uSat) && (err > 0.0f));

  if (allowIntegrate) {
    integral += WHEEL_KI_PWM_PER_RPM_S * err * dt;
    integral = constrain(integral, -WHEEL_I_LIMIT_PWM, WHEEL_I_LIMIT_PWM);
  }

  uPre = uFF + p + integral;
  return constrain(uPre, -255.0f, 255.0f);
}

// ==================== ODOMETRY ====================
void updateMeasuredWheelRPM(float dt) {
  long cur_enc[4];

  noInterrupts();
  for (int i = 0; i < 4; i++) cur_enc[i] = enc[i];
  interrupts();

  float ticks_to_rpm = 60.0 / (TICKS_PER_REV * dt);
  float alpha = dt / (TAU_WHEEL_MEAS + dt);
  for (int i = 0; i < 4; i++) {
    long delta = cur_enc[i] - prev_enc[i];
    if (abs(delta) > 10000) delta = 0;
    rpm[i] = delta * ticks_to_rpm;
    rpm_filt[i] += alpha * (rpm[i] - rpm_filt[i]);
    prev_enc[i] = cur_enc[i];
  }
}

// [IMU] Hàm này đã được sửa để dùng yaw và wz từ BNO085 thay cho encoder.
// - theta: lấy trực tiếp từ imu_yaw_rad (tuyệt đối, không drift).
// - wz: lấy từ gyro.z của BNO085 (chính xác hơn tính từ chênh lệch bánh).
// - x, y: vẫn tích phân từ vận tốc bánh (chuẩn mecanum), nhưng xoay theo theta IMU.
// Nếu IMU chưa có dữ liệu, tự động fallback sang cách tính cũ (từ encoder).
void updateOdometry(float dt) {
  float v_fl = rpmToWheelMps(rpm_filt[0]);
  float v_fr = rpmToWheelMps(rpm_filt[1]);
  float v_bl = rpmToWheelMps(rpm_filt[2]);
  float v_br = rpmToWheelMps(rpm_filt[3]);

  float vx_local = (v_fl + v_fr - v_bl - v_br) * 0.25;
  float vy_local = (-v_fl + v_fr - v_bl + v_br) * 0.25;

  // [IMU] wz ưu tiên từ gyro, fallback về encoder nếu IMU chưa sẵn sàng
  float wz_encoder = (v_fl - v_fr - v_bl + v_br) / (4.0 * ROTATION_RADIUS_M);
  float wz = imu_has_gyro ? imu_wz_rad_s : wz_encoder;

  // [IMU] theta ưu tiên từ IMU (tuyệt đối), fallback về tích phân wz
  if (imu_has_yaw) {
    odom_theta_rad = imu_yaw_rad;
  } else {
    odom_theta_rad = wrapAngle(odom_theta_rad + wz * dt);
  }

  float cos_th = cos(odom_theta_rad);
  float sin_th = sin(odom_theta_rad);

  float vx_global = vx_local * cos_th - vy_local * sin_th;
  float vy_global = vx_local * sin_th + vy_local * cos_th;

  odom_x_m += vx_global * dt;
  odom_y_m += vy_global * dt;

  robot_vx_mps = vx_local;
  robot_vy_mps = vy_local;
  robot_wz_rad_s = wz;
}

// ==================== ĐIỀU KHIỂN TỌA ĐỘ ====================
void computeTargetWheelRPM() {
  if (!motion.active) {
    for (int i = 0; i < 4; i++) target_rpm[i] = 0.0;
    return;
  }

  float dx = motion.x_m - odom_x_m;
  float dy = motion.y_m - odom_y_m;
  float dtheta = wrapAngle(motion.theta_rad - odom_theta_rad);

  float cos_th = cos(odom_theta_rad);
  float sin_th = sin(odom_theta_rad);

  float ex_local = dx * cos_th + dy * sin_th;
  float ey_local = -dx * sin_th + dy * cos_th;

  float pos_err = sqrt(dx * dx + dy * dy);
  float speed = sqrt(robot_vx_mps * robot_vx_mps + robot_vy_mps * robot_vy_mps);

  if (!square_running && pos_err < POS_TOL_M && fabs(dtheta) < THETA_TOL_RAD && speed < SPEED_TOL_MPS) {
    target_hold_counter++;
    if (target_hold_counter >= TARGET_HOLD_CYCLES) {
      stopMotion();
      return;
    }
  } else {
    target_hold_counter = 0;
  }

  float max_speed = MAX_LINEAR_MPS;
  if (pos_err < DECEL_RADIUS_M) {
    max_speed = MAX_LINEAR_MPS * (pos_err / DECEL_RADIUS_M);
    if (max_speed < 0.03) max_speed = 0.03;
  }

  float vx_cmd = clampAbs(KP_X * ex_local - KD_POS * robot_vx_mps, max_speed);
  float vy_cmd = clampAbs(KP_Y * ey_local - KD_POS * robot_vy_mps, max_speed);

  float line_dx = motion.x_m - prev_wp_x;
  float line_dy = motion.y_m - prev_wp_y;
  float line_len = sqrt(line_dx * line_dx + line_dy * line_dy);
  if (line_len > 0.02) {
    float rx = odom_x_m - prev_wp_x;
    float ry = odom_y_m - prev_wp_y;
    float cross_err = (ry * line_dx - rx * line_dy) / line_len;
    float corr_gx =  KP_CROSS * cross_err * line_dy / line_len;
    float corr_gy = -KP_CROSS * cross_err * line_dx / line_len;
    float corr_lx =  corr_gx * cos_th + corr_gy * sin_th;
    float corr_ly = -corr_gx * sin_th + corr_gy * cos_th;
    vx_cmd = clampAbs(vx_cmd + corr_lx, MAX_LINEAR_MPS);
    vy_cmd = clampAbs(vy_cmd + corr_ly, MAX_LINEAR_MPS);
  }

  float dt_pos = CTRL_DT_MS / 1000.0;
  I_theta += KI_TH * dtheta * dt_pos;
  I_theta = clampAbs(I_theta, MAX_ANGULAR_RAD_S * 0.5);
  float wz_cmd = clampAbs(KP_TH * dtheta + I_theta - KD_THETA * robot_wz_rad_s, MAX_ANGULAR_RAD_S);

  float wheel_mps[4];
  wheel_mps[0] = vx_cmd - vy_cmd + ROTATION_RADIUS_M * wz_cmd;
  wheel_mps[1] = vx_cmd + vy_cmd - ROTATION_RADIUS_M * wz_cmd;
  wheel_mps[2] = -vx_cmd - vy_cmd - ROTATION_RADIUS_M * wz_cmd;
  wheel_mps[3] = -vx_cmd + vy_cmd + ROTATION_RADIUS_M * wz_cmd;

  float max_wheel_rpm = 0.0;
  for (int i = 0; i < 4; i++) {
    float abs_rpm = fabs(wheelMpsToRpm(wheel_mps[i]));
    if (abs_rpm > max_wheel_rpm) max_wheel_rpm = abs_rpm;
  }

  float scale = 1.0;
  if (max_wheel_rpm > RPM_MAX) scale = RPM_MAX / max_wheel_rpm;

  float maxDeltaRpm = TARGET_RPM_SLEW_RPM_PER_S * dt_pos;
  for (int i = 0; i < 4; i++) {
    float desiredTargetRpm = wheelMpsToRpm(wheel_mps[i] * scale);
    target_rpm[i] = slewToward(target_rpm[i], desiredTargetRpm, maxDeltaRpm);
  }
}

void handleGamepadCommand() {
  if (!ps2_ready) return;

  ps2x.read_gamepad(false, 0);

  if (ps2x.ButtonPressed(PSB_START)) {
    startStoredTarget();
    Serial.print("GAMEPAD START m/rad: ");
    Serial.print(motion.x_m, 2); Serial.print(", ");
    Serial.print(motion.y_m, 2); Serial.print(", ");
    Serial.println(motion.theta_rad, 2);
  }

  if (ps2x.ButtonPressed(PSB_SELECT)) {
    square_running = false;
    stopMotion();
    Serial.println("GAMEPAD STOP");
  }
}

// ==================== SERIAL COMMAND ====================
static bool parseThreeFloats(const char *str, float &a, float &b, float &c) {
  char *end;
  a = (float)strtod(str, &end);
  if (end == str) return false;
  str = end;
  b = (float)strtod(str, &end);
  if (end == str) return false;
  str = end;
  c = (float)strtod(str, &end);
  if (end == str) return false;
  return true;
}

void handleSerialCommand() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  if (line.equalsIgnoreCase("S")) {
    square_running = false;
    stopMotion();
    Serial.println("STOP");
    return;
  }

  if (line.equalsIgnoreCase("R")) {
    resetOdometry();
    Serial.println("ODOM RESET (+ IMU yaw offset)");
    return;
  }

  if (line.equalsIgnoreCase("SQ")) {
    resetOdometry();
    startSquare();
    return;
  }

  if (line.startsWith("G ") || line.startsWith("g ")) {
    float x_cm, y_cm, theta_deg;
    if (parseThreeFloats(line.c_str() + 2, x_cm, y_cm, theta_deg)) {
      setTargetPose(x_cm / 100.0, y_cm / 100.0, theta_deg * PI / 180.0);
      Serial.print("TARGET SET cm/deg: ");
      Serial.print(x_cm, 1); Serial.print(", ");
      Serial.print(y_cm, 1); Serial.print(", ");
      Serial.println(theta_deg, 1);
    } else {
      Serial.println("ERR: use G x_cm y_cm theta_deg");
    }
    return;
  }

  Serial.println("CMD: SQ | G x_cm y_cm theta_deg | S | R");
}

// ==================== KHỞI TẠO ====================
void setup() {
  Serial.begin(115200);
  delay(300);

  // [IMU] Khởi tạo I2C cho BNO085
  Wire.begin();
  Wire.setClock(IMU_I2C_CLOCK_HZ);

  // [IMU] Thử kết nối BNO085
  if (beginIMU()) {
    imu_connected = true;
    setIMUReports();
    Serial.println("[IMU] San sang, cho du lieu dau tien...");
  } else {
    imu_connected = false;
    Serial.println("[IMU] KHONG KET NOI DUOC -- se dung odometry thuan tuy encoder");
    Serial.println("[IMU] Kiem tra: VCC 3.3V, GND, SDA=20, SCL=21, dia chi 0x4A/0x4B");
  }

  byte ps2_error = ps2x.config_gamepad(PS2_CLK, PS2_CMD, PS2_ATT, PS2_DAT, false, true);
  if (ps2_error == 0) {
    ps2_ready = true;
    Serial.println("PS2 gamepad ready: START run, SELECT stop");
  } else {
    Serial.print("PS2 unavailable, code: ");
    Serial.println(ps2_error);
  }

  pinMode(ENC1_A, INPUT_PULLUP);
  pinMode(ENC2_A, INPUT_PULLUP);
  pinMode(ENC3_A, INPUT_PULLUP);
  pinMode(ENC4_A, INPUT_PULLUP);
  pinMode(ENC1_B, INPUT_PULLUP);
  pinMode(ENC2_B, INPUT_PULLUP);
  pinMode(ENC3_B, INPUT_PULLUP);
  pinMode(ENC4_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENC1_A), enc_isr0, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC2_A), enc_isr1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC3_A), enc_isr2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC4_A), enc_isr3, CHANGE);

  pinMode(M1_L, OUTPUT); pinMode(M1_R, OUTPUT);
  pinMode(M2_L, OUTPUT); pinMode(M2_R, OUTPUT);
  pinMode(M3_L, OUTPUT); pinMode(M3_R, OUTPUT);
  pinMode(M4_L, OUTPUT); pinMode(M4_R, OUTPUT);

  // [IMU] Đọc IMU một vài lần trước khi reset odometry -> chụp offset đúng
  for (int i = 0; i < 50; i++) {
    updateIMU();
    delay(10);
  }

  resetOdometry();
  hardStopAllMotors();

  Serial.println("Coordinate control ready (+ IMU yaw fusion)");
  Serial.println("CMD: SQ | G x_cm y_cm theta_deg | S | R");
  Serial.println("Gamepad: START run target/demo | SELECT stop");
  Serial.print("Wheel circumference m: ");
  Serial.println(WHEEL_CIRCUMFERENCE_M, 6);
  Serial.print("Distance per tick mm: ");
  Serial.println(ticksToMeters(1) * 1000.0, 4);
}

// ==================== MAIN LOOP ====================
void loop() {
  static uint32_t last_ctrl = 0;
  static uint32_t last_dbg = 0;
  static bool demo_started = false;

  // [IMU] Đọc IMU trong mọi vòng loop để không bị tràn queue
  updateIMU();

  handleGamepadCommand();
  handleSerialCommand();

  uint32_t now = millis();
  if (AUTO_START_DEMO && !ps2_ready && !demo_started && now >= AUTO_START_DELAY_MS) {
    // Reset lại odometry + IMU offset ngay trước khi khởi động demo
    resetOdometry();
    startSquare();
    demo_started = true;
    Serial.println("AUTO DEMO START: hinh vuong 2m x 4 canh");
  }

  if (now - last_ctrl < CTRL_DT_MS) return;
  float dt = CTRL_DT_MS / 1000.0;
  last_ctrl = now;

  updateMeasuredWheelRPM(dt);
  updateOdometry(dt);

  if (square_running && motion.active) {
    float dx = motion.x_m - odom_x_m;
    float dy = motion.y_m - odom_y_m;
    float dist = sqrt(dx * dx + dy * dy);
    if (dist < SQUARE_WP_SWITCH_RADIUS_M) {
      advanceSquareWaypoint();
    }
  }

  computeTargetWheelRPM();

  float motor_cmd[4];
  for (int i = 0; i < 4; i++) {
    motor_cmd[i] = PID_WHEEL_RPM(target_rpm[i], rpm_filt[i], I_term[i], filt_ref[i], dt);
  }

  bool nearStopCmd = (fabs(target_rpm[0]) + fabs(target_rpm[1]) + fabs(target_rpm[2]) + fabs(target_rpm[3])) < 20.0f;
  bool nearStopMeas = true;
  for (int i = 0; i < 4; ++i) {
    if (fabs(rpm_filt[i]) > WHEEL_MEAS_ZERO_RPM) {
      nearStopMeas = false;
      break;
    }
  }

  if (!motion.active && nearStopCmd && nearStopMeas) {
    hardStopAllMotors();
  } else {
    applyMotorPwm(motor_cmd[0], motor_cmd[1], motor_cmd[2], motor_cmd[3]);
  }

  if (now - last_dbg >= 200) {
    last_dbg = now;
    Serial.print("ODOM m: ");
    Serial.print(odom_x_m, 3); Serial.print(", ");
    Serial.print(odom_y_m, 3); Serial.print(" | th deg: ");
    Serial.print(odom_theta_rad * 180.0 / PI, 2);
    // [IMU] Cho biết nguồn theta: I = IMU, E = encoder
    Serial.print(imu_has_yaw ? " [I]" : " [E]");
    Serial.print(" | tgt m: ");
    Serial.print(motion.x_m, 2); Serial.print(", ");
    Serial.print(motion.y_m, 2); Serial.print(", ");
    Serial.print(motion.theta_rad * 180.0 / PI, 1);
    Serial.print(" | active: ");
    Serial.print(motion.active ? 1 : 0);
    Serial.print(" | rpm meas: ");
    for (int i = 0; i < 4; i++) {
      Serial.print(rpm[i], 1);
      Serial.print(i == 3 ? ' ' : '/');
    }
    Serial.print("| rpm tgt: ");
    for (int i = 0; i < 4; i++) {
      Serial.print(target_rpm[i], 1);
      Serial.print(i == 3 ? ' ' : '/');
    }
    Serial.println();
  }
}
