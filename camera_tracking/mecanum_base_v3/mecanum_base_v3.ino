#include <math.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>

// ==================== CẤU HÌNH CHÂN ====================
const uint8_t M1_L = 46, M1_R = 44;   // FL
const uint8_t M2_L = 10, M2_R = 9;    // FR
const uint8_t M3_L = 6,  M3_R = 7;    // BL
const uint8_t M4_L = 11, M4_R = 12;   // BR

const uint8_t ENC1_A = 18, ENC1_B = 22;  // FL
const uint8_t ENC2_A = 19, ENC2_B = 23;  // FR
const uint8_t ENC3_A = 2,  ENC3_B = 24;  // BL
const uint8_t ENC4_A = 3,  ENC4_B = 25;  // BR

// Kênh B: Port A (đọc nhanh qua thanh ghi)
#define READ_B0 (PINA & (1 << PA0))  // pin 22 - FL
#define READ_B1 (PINA & (1 << PA1))  // pin 23 - FR
#define READ_B2 (PINA & (1 << PA2))  // pin 24 - BL
#define READ_B3 (PINA & (1 << PA3))  // pin 25 - BR

// Kênh A: Port D (pin 18,19) và Port E (pin 2,3)
#define READ_A0 (PIND & (1 << 3))    // pin 18 = ENC1_A (PD3)
#define READ_A1 (PIND & (1 << 2))    // pin 19 = ENC2_A (PD2)
#define READ_A2 (PINE & (1 << 4))    // pin 2  = ENC3_A (PE4)
#define READ_A3 (PINE & (1 << 5))    // pin 3  = ENC4_A (PE5)

// ==================== THÔNG SỐ CƠ KHÍ ====================
const float TICKS_PER_REV       = 500.0;  // 250 CPR × 2x quadrature
const float WHEEL_DIAMETER_M    = 0.100;
const float WHEEL_CIRCUMFERENCE = PI * WHEEL_DIAMETER_M;
const float ROTATION_RADIUS_M   = 0.300;  // Bán kính quay (tâm robot → bánh)

// ==================== [IMU] CẤU HÌNH BNO085 ====================
#define BNO08X_RESET      -1
#define BNO08X_ADDR_LOW   0x4A
#define BNO08X_ADDR_HIGH  0x4B
const unsigned long IMU_REPORT_INTERVAL_US = 20000;
const uint32_t IMU_I2C_CLOCK_HZ = 100000;
const uint8_t IMU_MAX_EVENTS_PER_UPDATE = 2;
const bool ENABLE_IMU_GYRO = true;
const float IMU_YAW_SIGN = +1.0f;

// ==================== THÔNG SỐ ĐIỀU KHIỂN ====================
const float RPM_MAX = 729.0;
const uint32_t CTRL_DT_MS = 10;
const float TAU_P_FILT = 0.01;
const float TAU_WHEEL_MEAS = 0.04;

const float WHEEL_KP_PWM_PER_RPM = 0.20f;
const float WHEEL_KI_PWM_PER_RPM_S = 0.75f;
const float WHEEL_KV_PWM_PER_RPM = 255.0f / RPM_MAX;
const float WHEEL_KS_PWM = 22.0f;
const float WHEEL_I_LIMIT_PWM = 110.0f;
const float WHEEL_REF_ZERO_RPM = 12.0f;
const float WHEEL_MEAS_ZERO_RPM = 10.0f;
const float PWM_ZERO_DB = 14.0f;

const float CMD_SLEW_VX_MPS_S = 1.2f;
const float CMD_SLEW_VY_MPS_S = 1.0f;
const float CMD_SLEW_WZ_RAD_S2 = 3.0f;

const float OMEGA_CMD_DEADBAND = 0.05f;
const float LIN_MOVE_THRESHOLD = 0.01f;
const float KP_HEADING_HOLD = 3.0f;
const float KI_HEADING_HOLD = 0.5f;
const float KD_HEADING_HOLD = 0.5f;
const float MAX_HEADING_INTEGRAL = 1.0f;
const float KP_YAW_RATE = 0.5f;

// Timeout an toàn: dừng robot nếu không nhận lệnh trong 500ms
const unsigned long CMD_TIMEOUT_MS = 500;

// ==================== BIẾN TOÀN CỤC ====================
volatile long enc[4] = {0, 0, 0, 0};
long encPrev[4] = {0, 0, 0, 0};

float targetRPM[4]   = {0, 0, 0, 0};
float actualRPM[4]   = {0, 0, 0, 0};
float actualRPMFilt[4] = {0, 0, 0, 0};
float wheelIntegralPwm[4] = {0, 0, 0, 0};
float filtRef[4]     = {0, 0, 0, 0};
float pwmOutput[4]   = {0, 0, 0, 0};

float vxTarget = 0.0f;
float vyTarget = 0.0f;
float wzTarget = 0.0f;
float vxCmd = 0.0f;
float vyCmd = 0.0f;
float wzCmd = 0.0f;

// [IMU] BNO085
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t imu_value;
bool imu_connected = false;
bool imu_has_yaw = false;
bool imu_has_gyro = false;
bool imu_pending_offset = true;
float imu_yaw_raw_rad = 0.0f;
float imu_yaw_offset_rad = 0.0f;
float imu_yaw_rad = 0.0f;
float imu_wz_rad_s = 0.0f;

// Odometry
float odom_x_m = 0.0;
float odom_y_m = 0.0;
float odom_theta_rad = 0.0;
float robot_vx_mps = 0.0f;
float robot_vy_mps = 0.0f;
float robot_wz_rad_s = 0.0f;

bool headingLocked = false;
float headingTargetRad = 0.0f;
float headingIntegral = 0.0f;

unsigned long lastCtrlTime = 0;
unsigned long lastCmdTime = 0;

// Serial input buffer
char serialBuf[64];
uint8_t serialIdx = 0;

// ==================== TIỆN ÍCH ====================
float rpmToWheelMps(float wheel_rpm) {
  return wheel_rpm * WHEEL_CIRCUMFERENCE / 60.0;
}

float wheelMpsToRpm(float wheel_mps) {
  return wheel_mps * 60.0 / WHEEL_CIRCUMFERENCE;
}

float wrapAngle(float a) {
  while (a >  PI) a -= 2.0 * PI;
  while (a < -PI) a += 2.0 * PI;
  return a;
}

float signNonZero(float value) {
  if (value > 0.0f) return 1.0f;
  if (value < 0.0f) return -1.0f;
  return 0.0f;
}

float clampAbs(float value, float limit) {
  if (value > limit) return limit;
  if (value < -limit) return -limit;
  return value;
}

float slewToward(float current, float target, float maxDelta) {
  float delta = constrain(target - current, -maxDelta, maxDelta);
  return current + delta;
}

static float quatToYaw(float qw, float qx, float qy, float qz) {
  return atan2(2.0f * (qw * qz + qx * qy),
               1.0f - 2.0f * (qy * qy + qz * qz));
}

void setIMUReports() {
  if (!bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, IMU_REPORT_INTERVAL_US)) {
    Serial.println("[IMU] Khong bat duoc game rotation vector");
  }
  if (ENABLE_IMU_GYRO && !bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, IMU_REPORT_INTERVAL_US)) {
    Serial.println("[IMU] Khong bat duoc gyroscope");
  }
}

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

void updateIMU() {
  if (!imu_connected) return;

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

void captureIMUYawOffset() {
  if (imu_has_yaw) {
    imu_yaw_offset_rad = imu_yaw_raw_rad;
    imu_yaw_rad = 0.0f;
  } else {
    imu_pending_offset = true;
  }
  headingLocked = false;
  headingIntegral = 0.0f;
}

void resetOdometry() {
  noInterrupts();
  for (int i = 0; i < 4; ++i) {
    enc[i] = 0;
    encPrev[i] = 0;
  }
  interrupts();

  odom_x_m = 0.0f;
  odom_y_m = 0.0f;
  odom_theta_rad = 0.0f;
  robot_vx_mps = 0.0f;
  robot_vy_mps = 0.0f;
  robot_wz_rad_s = 0.0f;
  captureIMUYawOffset();
}

float applyHeadingControl(float vxCmdLocal, float vyCmdLocal, float wzCmdLocal, float dt) {
  if (!imu_has_yaw || !imu_has_gyro) {
    headingLocked = false;
    headingIntegral = 0.0f;
    return wzCmdLocal;
  }

  bool hasLinear = (fabs(vxCmdLocal) > LIN_MOVE_THRESHOLD) ||
                   (fabs(vyCmdLocal) > LIN_MOVE_THRESHOLD);
  bool omegaNearZero = fabs(wzCmdLocal) < OMEGA_CMD_DEADBAND;

  if (omegaNearZero && hasLinear) {
    if (!headingLocked) {
      headingTargetRad = imu_yaw_rad;
      headingLocked = true;
      headingIntegral = 0.0f;
    }

    float headingErr = wrapAngle(headingTargetRad - imu_yaw_rad);
    headingIntegral += KI_HEADING_HOLD * headingErr * dt;
    headingIntegral = clampAbs(headingIntegral, MAX_HEADING_INTEGRAL);

    float omegaFix = KP_HEADING_HOLD * headingErr +
                     headingIntegral -
                     KD_HEADING_HOLD * imu_wz_rad_s;
    return clampAbs(omegaFix, 3.14f);
  }

  if (!omegaNearZero) {
    headingLocked = false;
    headingIntegral = 0.0f;
    float wzErr = wzCmdLocal - imu_wz_rad_s;
    return clampAbs(wzCmdLocal + KP_YAW_RATE * wzErr, 3.14f);
  }

  headingLocked = false;
  headingIntegral = 0.0f;
  return 0.0f;
}

// ==================== ĐIỀU KHIỂN MOTOR ====================
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

void hardStopAll() {
  applyMotorPwm(0.0f, 0.0f, 0.0f, 0.0f);

  noInterrupts();
  for (int i = 0; i < 4; ++i) encPrev[i] = enc[i];
  interrupts();

  vxTarget = 0.0f; vyTarget = 0.0f; wzTarget = 0.0f;
  vxCmd = 0.0f; vyCmd = 0.0f; wzCmd = 0.0f;
  headingLocked = false;
  headingIntegral = 0.0f;
  for (int i = 0; i < 4; ++i) {
    targetRPM[i] = 0.0f;
    actualRPM[i] = 0.0f;
    actualRPMFilt[i] = 0.0f;
    wheelIntegralPwm[i] = 0.0f;
    filtRef[i] = 0.0f;
    pwmOutput[i] = 0.0f;
  }
}

// ==================== ĐỘNG HỌC NGHỊCH MECANUM ====================
// IK khớp với robot đã test thực tế (dấu BL/BR đảo vì lắp ngược)
//   vx    = tiến/lùi (m/s, dương = tiến)
//   vy    = trái/phải (m/s, dương = sang trái)
//   omega = xoay (rad/s, dương = ngược chiều kim đồng hồ)
void mecanumIK(float vx, float vy, float omega) {
  float v_fl =  vx - vy + ROTATION_RADIUS_M * omega;
  float v_fr =  vx + vy - ROTATION_RADIUS_M * omega;
  float v_bl = -vx - vy - ROTATION_RADIUS_M * omega;
  float v_br = -vx + vy + ROTATION_RADIUS_M * omega;

  float factor = 60.0 / WHEEL_CIRCUMFERENCE;
  targetRPM[0] = v_fl * factor;
  targetRPM[1] = v_fr * factor;
  targetRPM[2] = v_bl * factor;
  targetRPM[3] = v_br * factor;

  // Scale nếu vượt quá RPM_MAX
  float maxReq = 0;
  for (int i = 0; i < 4; i++) {
    if (fabs(targetRPM[i]) > maxReq) maxReq = fabs(targetRPM[i]);
  }
  if (maxReq > RPM_MAX) {
    float scale = RPM_MAX / maxReq;
    for (int i = 0; i < 4; i++) targetRPM[i] *= scale;
  }
}

// ==================== ISR ENCODER ====================
// 2x Quadrature: CHANGE trên kênh A, đọc cả A+B → (A XOR B) cho chiều.
// Gấp đôi phân giải so với RISING-only (250→500 tick/rev).
// Đọc PORT trực tiếp (~0.1µs) thay vì digitalRead (~4-5µs).
void enc_isr0() { enc[0] += (READ_A0 ? 1 : 0) ^ (READ_B0 ? 1 : 0) ? +1 : -1; }
void enc_isr1() { enc[1] += (READ_A1 ? 1 : 0) ^ (READ_B1 ? 1 : 0) ? -1 : +1; }
void enc_isr2() { enc[2] += (READ_A2 ? 1 : 0) ^ (READ_B2 ? 1 : 0) ? -1 : +1; }
void enc_isr3() { enc[3] += (READ_A3 ? 1 : 0) ^ (READ_B3 ? 1 : 0) ? +1 : -1; }

float updateWheelController(float rpmRef, float rpmMeas, float &integralPwm, float &refFilt, float dt) {
  refFilt += (rpmRef - refFilt) * dt / TAU_P_FILT;
  refFilt = constrain(refFilt, -RPM_MAX, RPM_MAX);

  if (fabs(refFilt) < WHEEL_REF_ZERO_RPM && fabs(rpmMeas) < WHEEL_MEAS_ZERO_RPM) {
    integralPwm = 0.0f;
    refFilt = 0.0f;
    return 0.0f;
  }

  float err = refFilt - rpmMeas;
  float uFF = WHEEL_KV_PWM_PER_RPM * refFilt;
  if (fabs(refFilt) > WHEEL_REF_ZERO_RPM) {
    uFF += signNonZero(refFilt) * WHEEL_KS_PWM;
  }

  float p = WHEEL_KP_PWM_PER_RPM * err;
  float uPre = uFF + p + integralPwm;
  float uSat = constrain(uPre, -255.0f, 255.0f);

  bool allowIntegrate = (fabs(uPre - uSat) < 0.5f) ||
                        ((uPre > uSat) && (err < 0.0f)) ||
                        ((uPre < uSat) && (err > 0.0f));

  if (allowIntegrate) {
    integralPwm += WHEEL_KI_PWM_PER_RPM_S * err * dt;
    integralPwm = constrain(integralPwm, -WHEEL_I_LIMIT_PWM, WHEEL_I_LIMIT_PWM);
  }

  uPre = uFF + p + integralPwm;
  return constrain(uPre, -255.0f, 255.0f);
}

// ==================== ĐO ENCODER + ODOMETRY ====================
void updateMeasuredRPM(float dt) {
  long encNow[4];
  noInterrupts();
  for (int i = 0; i < 4; i++) encNow[i] = enc[i];
  interrupts();

  float ticks_to_rpm = 60.0 / (TICKS_PER_REV * dt);
  float alpha = dt / (TAU_WHEEL_MEAS + dt);
  for (int i = 0; i < 4; i++) {
    long delta = encNow[i] - encPrev[i];
    if (abs(delta) > 10000) delta = 0;  // Spike rejection
    actualRPM[i] = delta * ticks_to_rpm;
    actualRPMFilt[i] += alpha * (actualRPM[i] - actualRPMFilt[i]);
    encPrev[i] = encNow[i];
  }
}

void updateOdometry(float dt) {
  float v_fl = rpmToWheelMps(actualRPMFilt[0]);
  float v_fr = rpmToWheelMps(actualRPMFilt[1]);
  float v_bl = rpmToWheelMps(actualRPMFilt[2]);
  float v_br = rpmToWheelMps(actualRPMFilt[3]);

  // Forward kinematics (khớp với IK đã test)
  float vx_local = ( v_fl + v_fr - v_bl - v_br) * 0.25;
  float vy_local = (-v_fl + v_fr - v_bl + v_br) * 0.25;
  float wz_encoder = ( v_fl - v_fr - v_bl + v_br) / (4.0 * ROTATION_RADIUS_M);
  float wz = imu_has_gyro ? imu_wz_rad_s : wz_encoder;

  if (imu_has_yaw) {
    odom_theta_rad = imu_yaw_rad;
  } else {
    odom_theta_rad = wrapAngle(odom_theta_rad + wz * dt);
  }

  float cos_th = cos(odom_theta_rad);
  float sin_th = sin(odom_theta_rad);

  odom_x_m += (vx_local * cos_th - vy_local * sin_th) * dt;
  odom_y_m += (vx_local * sin_th + vy_local * cos_th) * dt;
  robot_vx_mps = vx_local;
  robot_vy_mps = vy_local;
  robot_wz_rad_s = wz;
}

// ==================== XỬ LÝ LỆNH SERIAL ====================
// Fix: sscanf %f không hoạt động trên AVR → dùng strtod
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

void processCommand(const char* cmd) {
  if (cmd[0] == 'V' && cmd[1] == ' ') {
    float vx = 0, vy = 0, omega = 0;
    if (parseThreeFloats(cmd + 2, vx, vy, omega)) {
      vx    = constrain(vx, -1.0f, 1.0f);
      vy    = constrain(vy, -1.0f, 1.0f);
      omega = constrain(omega, -3.14f, 3.14f);
      vxTarget = vx;
      vyTarget = vy;
      wzTarget = omega;
      lastCmdTime = millis();
    }
  }
  else if (cmd[0] == 'S') {
    hardStopAll();
    lastCmdTime = millis();
  }
  else if (cmd[0] == 'R') {
    resetOdometry();
    hardStopAll();
    lastCmdTime = millis();
    Serial.println("ODOM_RESET");
  }
}

// ==================== KHỞI TẠO ====================
void setup() {
  Serial.begin(115200);

  Wire.begin();
  Wire.setClock(IMU_I2C_CLOCK_HZ);

  if (beginIMU()) {
    imu_connected = true;
    setIMUReports();
    Serial.println("[IMU] San sang, cho du lieu dau tien...");
  } else {
    Serial.println("[IMU] KHONG KET NOI DUOC -- se dung odometry thuan tuy encoder");
    Serial.println("[IMU] Kiem tra: VCC 3.3V, GND, SDA=20, SCL=21, dia chi 0x4A/0x4B");
  }

  // Motor pins
  pinMode(M1_L, OUTPUT); pinMode(M1_R, OUTPUT);
  pinMode(M2_L, OUTPUT); pinMode(M2_R, OUTPUT);
  pinMode(M3_L, OUTPUT); pinMode(M3_R, OUTPUT);
  pinMode(M4_L, OUTPUT); pinMode(M4_R, OUTPUT);

  // Encoder A (interrupt) + B (port read)
  pinMode(ENC1_A, INPUT_PULLUP); pinMode(ENC1_B, INPUT_PULLUP);
  pinMode(ENC2_A, INPUT_PULLUP); pinMode(ENC2_B, INPUT_PULLUP);
  pinMode(ENC3_A, INPUT_PULLUP); pinMode(ENC3_B, INPUT_PULLUP);
  pinMode(ENC4_A, INPUT_PULLUP); pinMode(ENC4_B, INPUT_PULLUP);

  // 2x Quadrature: CHANGE (không phải RISING)
  attachInterrupt(digitalPinToInterrupt(ENC1_A), enc_isr0, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC2_A), enc_isr1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC3_A), enc_isr2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC4_A), enc_isr3, CHANGE);

  for (int i = 0; i < 50; ++i) {
    updateIMU();
    delay(10);
  }

  hardStopAll();
  resetOdometry();
  lastCtrlTime = millis();
  lastCmdTime  = millis();

  Serial.println("MECANUM_READY");
}

// ==================== VÒNG LẶP CHÍNH ====================
void loop() {
  updateIMU();

  // --- Đọc Serial ---
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (serialIdx > 0) {
        serialBuf[serialIdx] = '\0';
        processCommand(serialBuf);
        serialIdx = 0;
      }
    } else {
      if (serialIdx < sizeof(serialBuf) - 1) {
        serialBuf[serialIdx++] = c;
      } else {
        serialIdx = 0;
      }
    }
  }

  // --- Vòng PID 100Hz (10ms) ---
  unsigned long now = millis();
  if (now - lastCtrlTime < CTRL_DT_MS) return;
  float dt = CTRL_DT_MS / 1000.0;
  lastCtrlTime = now;

  updateMeasuredRPM(dt);
  updateOdometry(dt);

  if (now - lastCmdTime > CMD_TIMEOUT_MS) {
    vxTarget = 0.0f;
    vyTarget = 0.0f;
    wzTarget = 0.0f;
  }

  vxCmd = slewToward(vxCmd, vxTarget, CMD_SLEW_VX_MPS_S * dt);
  vyCmd = slewToward(vyCmd, vyTarget, CMD_SLEW_VY_MPS_S * dt);
  wzCmd = slewToward(wzCmd, wzTarget, CMD_SLEW_WZ_RAD_S2 * dt);

  float wzApplied = applyHeadingControl(vxCmd, vyCmd, wzCmd, dt);
  mecanumIK(vxCmd, vyCmd, wzApplied);

  // Điều khiển từng bánh theo PWM-domain
  for (int i = 0; i < 4; i++) {
    pwmOutput[i] = updateWheelController(targetRPM[i], actualRPMFilt[i], wheelIntegralPwm[i], filtRef[i], dt);
  }

  bool nearStopCmd = (fabs(targetRPM[0]) + fabs(targetRPM[1]) + fabs(targetRPM[2]) + fabs(targetRPM[3])) < 20.0f;
  bool nearStopMeas = true;
  for (int i = 0; i < 4; ++i) {
    if (fabs(actualRPMFilt[i]) > WHEEL_MEAS_ZERO_RPM) {
      nearStopMeas = false;
      break;
    }
  }

  if (nearStopCmd && nearStopMeas) {
    applyMotorPwm(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 4; ++i) wheelIntegralPwm[i] = 0.0f;
  } else {
    applyMotorPwm(pwmOutput[0], pwmOutput[1], pwmOutput[2], pwmOutput[3]);
  }

  // Gửi phản hồi mỗi 10 chu kỳ PID (~100ms)
  static uint8_t fbCount = 0;
  if (++fbCount >= 10) {
    fbCount = 0;
    Serial.print("T ");
    Serial.print(robot_vx_mps, 3); Serial.print(' ');
    Serial.print(robot_vy_mps, 3); Serial.print(' ');
    Serial.print(robot_wz_rad_s, 3); Serial.print(' ');
    Serial.print(odom_x_m, 3); Serial.print(' ');
    Serial.print(odom_y_m, 3); Serial.print(' ');
    Serial.println(odom_theta_rad * 180.0 / PI, 2);
  }
}