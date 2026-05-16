#include <math.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <PS2X_lib.h>

// -----------------------------------------------------------------------------
// Modern control architecture for a 4-wheel X-drive robot with:
// - 4 quadrature encoders
// - BNO085 IMU
// - PS2 gamepad
//
// Design goals:
// 1. Keep wheel speed control robust and smooth.
// 2. Use a clean outer heading-hold loop instead of directly fighting joystick.
// 3. Use a simple modern estimator stack that is realistic on Arduino Mega:
//    - encoder-based wheel velocity estimation
//    - IMU yaw complementary estimator (gyro integration + fused yaw correction)
//    - body-level heading controller producing desired yaw rate
// 4. Avoid low-speed chatter with feedforward, anti-windup and clean stop logic.
// -----------------------------------------------------------------------------

// ==================== Pin configuration ====================
const uint8_t M1_L = 46, M1_R = 44;   // Front-left
const uint8_t M2_L = 10, M2_R = 9;    // Front-right
const uint8_t M3_L = 6,  M3_R = 7;    // Back-left
const uint8_t M4_L = 11, M4_R = 12;   // Back-right

const uint8_t ENC_A[4] = {18, 19, 2, 3};

#define READ_B0 (PINA & (1 << PA0))
#define READ_B1 (PINA & (1 << PA1))
#define READ_B2 (PINA & (1 << PA2))
#define READ_B3 (PINA & (1 << PA3))

#define READ_A0 (PIND & (1 << PD3))
#define READ_A1 (PIND & (1 << PD2))
#define READ_A2 (PINE & (1 << PE4))
#define READ_A3 (PINE & (1 << PE5))

const uint8_t PS2_CLK = 50, PS2_CMD = 52, PS2_ATT = 51, PS2_DAT = 53;

// ==================== IMU ====================
#define BNO08X_RESET     -1
#define BNO08X_ADDR_LOW  0x4A
#define BNO08X_ADDR_HIGH 0x4B
const unsigned long IMU_REPORT_INTERVAL_US = 20000;  // 50 Hz to keep I2C load light on Mega
const uint32_t IMU_I2C_CLOCK_HZ = 100000;
const uint8_t IMU_MAX_EVENTS_PER_UPDATE = 2;
const float IMU_YAW_SIGN = +1.0f;
const bool ENABLE_IMU = true;
const bool ENABLE_IMU_GYRO = true;

// ==================== Robot and control tuning ====================
// [V2-KALMAN] Encoder spec: x2 quadrature decode (chỉ ISR trên kênh A mode CHANGE)
// đo được phóng đại 4x -> PID phải kéo PWM xuống để cân bằng -> rung.)
const float TICKS_PER_REV = 500.0f;
const float RPM_MAX = 729.0f;
const float WHEEL_RPM_REF_MAX = RPM_MAX * 0.88f;

const uint32_t CONTROL_PERIOD_US = 10000;  // 100 Hz nominal
const uint32_t COMMAND_TIMEOUT_MS = 250;

const float AXIS_DEADZONE = 0.18f;
const float TRANSLATE_ACTIVE_DB = 0.06f;
const float ROTATE_ENTER_DB = 0.09f;
const float ROTATE_EXIT_DB = 0.04f;
const float HEADING_CAPTURE_DELAY_S = 0.03f;
const float HEADING_ERR_DB_RAD = 0.010f;

const float MAX_ROTATE_RATE_RAD_S = 2.2f;
const float MAX_HOLD_CORR_RATE_RAD_S = 1.5f;
const bool ENABLE_HEADING_HOLD = true;

const float CMD_ACCEL_XY_START = 1.2f;  // softer initial takeoff from standstill
const float CMD_ACCEL_XY_RUN = 4.4f;    // faster once the robot is already moving
const float CMD_ACCEL_WZ_START = 1.8f;
const float CMD_ACCEL_WZ_RUN = 5.2f;
const float CMD_SOFT_START_BLEND = 0.20f;

const float TAU_GYRO = 0.025f;
const float IMU_CORR_GAIN = 0.10f;  // complementary correction on yaw estimate

// [V2-KALMAN] Kalman 2-state cho RPM bánh xe.
// Lý do thêm: dù encoder 500 PPR cho resolution tốt (~17 tick/period ở 200 RPM),
// quantization noise vẫn còn (~12 RPM RMS) cộng với rung cơ khí, ma sát động không
// đều. Low-pass cũ phải đánh đổi giữa lọc-mượt và trễ-pha. Kalman có MÔ HÌNH
// "RPM thay đổi mượt theo gia tốc" -> lọc tốt hơn cùng độ trễ, đồng thời cho ra
// drpm_hat làm D-term trên measurement (damping tốt hơn).
//
// Tune:
//   Q lớn  -> tin đo hơn, nhanh, ồn hơn
//   R lớn  -> tin model hơn, mượt, trễ hơn
//   Tỉ số Q/R quyết định tốc độ phản ứng. Khởi đầu hợp lý cho 500 PPR:
const float KALMAN_Q_RPM = 80000.0f;   // (RPM/s^2)^2 - cho phép gia tốc ~280 RPM/s
const float KALMAN_R_RPM = 50.0f;      // (RPM)^2 - tương đương noise ~7 RPM RMS (500 PPR)

// Wheel velocity controller in signed PWM units.
const float WHEEL_KP_PWM_PER_RPM = 0.20f;
const float WHEEL_KI_PWM_PER_RPM_S = 0.75f;
const float WHEEL_KD_PWM_PER_RPM_S = 0.008f;  // [V2-KALMAN] D-term trên measurement, dùng drpm_hat
const float WHEEL_KV_PWM_PER_RPM = 255.0f / RPM_MAX;
const float WHEEL_KS_PWM = 22.0f;
const float WHEEL_KS_MIN_BLEND = 0.55f;
const float WHEEL_KS_FULL_BLEND_RPM = 90.0f;
const float WHEEL_I_LIMIT_PWM = 110.0f;
const float WHEEL_REF_ZERO_RPM = 12.0f;
const float WHEEL_MEAS_ZERO_RPM = 10.0f;
const float PWM_ZERO_DB = 14.0f;

// Outer heading controller in yaw-rate units.
const float HEADING_KP = 3.6f;
const float HEADING_KI = 0.55f;
const float HEADING_KD = 0.24f;  // D on measured yaw rate
const float HEADING_I_LIMIT = 0.50f;

const float DPAD_SPEED = 0.65f;
const float TURN_IN_PLACE_SCALE = 0.20f;
const float TURN_WHILE_MOVING_SCALE = 0.65f;

const bool ENABLE_DEBUG = true;
const uint32_t DEBUG_PRINT_MS = 200;

// ==================== Utility ====================
float clampUnit(float x) {
  return constrain(x, -1.0f, 1.0f);
}

float wrapAngle(float rad) {
  while (rad > PI) rad -= 2.0f * PI;
  while (rad < -PI) rad += 2.0f * PI;
  return rad;
}

float angleDiff(float target, float current) {
  return wrapAngle(target - current);
}

float signNonZero(float x) {
  if (x > 0.0f) return 1.0f;
  if (x < 0.0f) return -1.0f;
  return 0.0f;
}

float applyDeadzone(int raw) {
  float x = (raw - 128) / 128.0f;
  if (fabs(x) < AXIS_DEADZONE) return 0.0f;
  return clampUnit(x);
}

float slewToward(float current, float target, float maxDelta) {
  float delta = constrain(target - current, -maxDelta, maxDelta);
  return current + delta;
}

float shapeErrorWithDeadband(float err, float deadband) {
  if (fabs(err) <= deadband) return 0.0f;
  return err - signNonZero(err) * deadband;
}

float vecAbs1(float a, float b) {
  return fabs(a) + fabs(b);
}

float max4(float a, float b, float c, float d) {
  return max(max(fabs(a), fabs(b)), max(fabs(c), fabs(d)));
}

float smoothstep01(float x) {
  x = constrain(x, 0.0f, 1.0f);
  return x * x * (3.0f - 2.0f * x);
}

// ==================== Controllers and state ====================
// [V2-KALMAN] WheelState chứa Kalman 2-state thay vì rpm_filt low-pass.
//   rpm_meas : RPM thô tính từ delta encoder (chỉ để debug/diagnostic)
//   rpm_hat  : RPM ước lượng bởi Kalman (dùng cho PID, odometry, near-stop)
//   drpm_hat : gia tốc bánh ước lượng (dùng cho D-term)
//   P[2][2]  : ma trận hiệp phương sai Kalman (2x2)
struct WheelState {
  float rpm_meas;
  float rpm_hat;
  float drpm_hat;
  float P[2][2];
  float rpm_ref;
  float integrator_pwm;
};

struct ImuState {
  bool connected;
  bool hasQuat;
  bool hasGyro;
  bool pendingZero;
  bool yawEstimatorInit;
  float yawRaw;
  float yawOffset;
  float yawBno;
  float yawEst;
  float gyroZ;
  float gyroZFilt;
};

struct HeadingState {
  bool active;
  bool manualRotate;
  float targetYaw;
  float integrator;
  float engageTimer;
};

struct CommandState {
  float vxTarget;
  float vyTarget;
  float wzTarget;
  float vx;
  float vy;
  float wz;
};

float updateWheelController(WheelState &w, float rpmRef, float dt);

PS2X ps2x;
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t imuEvent;

volatile long encCount[4] = {0, 0, 0, 0};
long prevEncCount[4] = {0, 0, 0, 0};

WheelState wheel[4];
ImuState imu = {false, false, false, true, false, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
HeadingState heading = {false, false, 0.0f, 0.0f, 0.0f};
CommandState cmd = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

int debugRawLX = 128;
int debugRawLY = 128;
int debugRawRX = 128;
float debugRef[4] = {0.0f, 0.0f, 0.0f, 0.0f};
float debugPwm[4] = {0.0f, 0.0f, 0.0f, 0.0f};

uint32_t lastCmdMs = 0;
uint32_t lastDebugMs = 0;
uint32_t lastControlUs = 0;

// ==================== Encoder ISR ====================
void enc_isr0() { bool a = READ_A0; bool b = READ_B0; encCount[0] += (a != b) ? +1 : -1; }
void enc_isr1() { bool a = READ_A1; bool b = READ_B1; encCount[1] += (a != b) ? -1 : +1; }
void enc_isr2() { bool a = READ_A2; bool b = READ_B2; encCount[2] += (a != b) ? -1 : +1; }
void enc_isr3() { bool a = READ_A3; bool b = READ_B3; encCount[3] += (a != b) ? +1 : -1; }

// ==================== IMU support ====================
static float quatToYaw(float qw, float qx, float qy, float qz) {
  return atan2(2.0f * (qw * qz + qx * qy),
               1.0f - 2.0f * (qy * qy + qz * qz));
}

void enableImuReports() {
  if (!bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, IMU_REPORT_INTERVAL_US)) {
    Serial.println(F("[IMU] Failed to enable game rotation vector"));
  }
  if (ENABLE_IMU_GYRO && !bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, IMU_REPORT_INTERVAL_US)) {
    Serial.println(F("[IMU] Failed to enable calibrated gyro"));
  }
}

bool beginImu() {
  Serial.println(F("[IMU] Trying BNO085 at 0x4A"));
  if (bno08x.begin_I2C(BNO08X_ADDR_LOW, &Wire)) {
    Serial.println(F("[IMU] Connected at 0x4A"));
    return true;
  }

  Serial.println(F("[IMU] Trying BNO085 at 0x4B"));
  if (bno08x.begin_I2C(BNO08X_ADDR_HIGH, &Wire)) {
    Serial.println(F("[IMU] Connected at 0x4B"));
    return true;
  }

  return false;
}

void zeroYawReference() {
  if (imu.hasQuat) {
    imu.yawOffset = imu.yawRaw;
    imu.yawBno = 0.0f;
    imu.yawEst = 0.0f;
    imu.yawEstimatorInit = true;
  } else {
    imu.pendingZero = true;
  }

  heading.active = false;
  heading.manualRotate = false;
  heading.targetYaw = 0.0f;
  heading.integrator = 0.0f;
  heading.engageTimer = 0.0f;
}

void updateImu() {
  if (!ENABLE_IMU) return;
  if (!imu.connected) return;

  if (bno08x.wasReset()) {
    Serial.println(F("[IMU] Sensor reset, re-enabling reports"));
    enableImuReports();
    imu.pendingZero = true;
    imu.hasQuat = false;
    imu.hasGyro = false;
    imu.yawEstimatorInit = false;
  }

  for (uint8_t eventsRead = 0; eventsRead < IMU_MAX_EVENTS_PER_UPDATE; ++eventsRead) {
    if (!bno08x.getSensorEvent(&imuEvent)) break;

    switch (imuEvent.sensorId) {
      case SH2_GAME_ROTATION_VECTOR: {
        float yaw = quatToYaw(
          imuEvent.un.gameRotationVector.real,
          imuEvent.un.gameRotationVector.i,
          imuEvent.un.gameRotationVector.j,
          imuEvent.un.gameRotationVector.k);

        imu.yawRaw = IMU_YAW_SIGN * yaw;

        if (imu.pendingZero) {
          imu.yawOffset = imu.yawRaw;
          imu.pendingZero = false;
        }

        imu.yawBno = wrapAngle(imu.yawRaw - imu.yawOffset);
        imu.hasQuat = true;

        if (!imu.yawEstimatorInit) {
          imu.yawEst = imu.yawBno;
          imu.yawEstimatorInit = true;
        }
        break;
      }

      case SH2_GYROSCOPE_CALIBRATED:
        if (ENABLE_IMU_GYRO) {
          imu.gyroZ = IMU_YAW_SIGN * imuEvent.un.gyroscope.z;
          imu.hasGyro = true;
        }
        break;
    }
  }
}

void stepYawEstimator(float dt) {
  if (!ENABLE_IMU) return;
  if (!imu.connected || !imu.hasQuat) return;

  if (!imu.yawEstimatorInit) {
    imu.yawEst = imu.yawBno;
    imu.yawEstimatorInit = true;
  }

  if (imu.hasGyro) {
    float alphaGyro = dt / (TAU_GYRO + dt);
    imu.gyroZFilt += alphaGyro * (imu.gyroZ - imu.gyroZFilt);
    imu.yawEst = wrapAngle(imu.yawEst + imu.gyroZFilt * dt);
  }

  imu.yawEst = wrapAngle(imu.yawEst + IMU_CORR_GAIN * angleDiff(imu.yawBno, imu.yawEst));
}

// ==================== Motor and wheel control ====================
void resetWheelControllers() {
  for (int i = 0; i < 4; ++i) {
    wheel[i].rpm_meas = 0.0f;
    wheel[i].rpm_hat = 0.0f;
    wheel[i].drpm_hat = 0.0f;
    // [V2-KALMAN] Init covariance "rộng" để filter hội tụ nhanh ban đầu
    wheel[i].P[0][0] = 100.0f; wheel[i].P[0][1] = 0.0f;
    wheel[i].P[1][0] = 0.0f;   wheel[i].P[1][1] = 100.0f;
    wheel[i].rpm_ref = 0.0f;
    wheel[i].integrator_pwm = 0.0f;
  }
}

// [V2-KALMAN] Reset chỉ state (không đụng covariance) - dùng khi tạm dừng motor
// vẫn giữ "độ tin cậy" của filter để khi chạy lại không cần hội tụ lại.
void kalmanResetWheelStateOnly(WheelState &w) {
  w.rpm_hat = 0.0f;
  w.drpm_hat = 0.0f;
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

void hardStopAll() {
  applyMotorPwm(0.0f, 0.0f, 0.0f, 0.0f);
  resetWheelControllers();

  noInterrupts();
  for (int i = 0; i < 4; ++i) {
    prevEncCount[i] = encCount[i];
  }
  interrupts();
}

// [V2-KALMAN] Thay thế low-pass filter bằng Kalman 2-state.
// Mô hình động học:
//   rpm(k+1)  = rpm(k) + dt * drpm(k)         (gia tốc tích phân thành RPM)
//   drpm(k+1) = drpm(k) + w                   (gia tốc là random walk)
//   F = [[1, dt], [0, 1]]
// Đo lường: y = rpm + v (chỉ đo được RPM, H = [1, 0])
// Q matrix derived từ continuous-time process noise KALMAN_Q_RPM.
void updateWheelMeasurements(float dt) {
  long current[4];
  noInterrupts();
  for (int i = 0; i < 4; ++i) current[i] = encCount[i];
  interrupts();

  float ticksToRpm = 60.0f / (TICKS_PER_REV * dt);

  float dt2 = dt * dt;
  float dt3 = dt2 * dt;
  float dt4 = dt2 * dt2;
  float Q11 = KALMAN_Q_RPM * dt4 / 4.0f;
  float Q12 = KALMAN_Q_RPM * dt3 / 2.0f;
  float Q22 = KALMAN_Q_RPM * dt2;

  for (int i = 0; i < 4; ++i) {
    long delta = current[i] - prevEncCount[i];
    prevEncCount[i] = current[i];

    if (labs(delta) > 10000) delta = 0;

    float rpm_meas_raw = delta * ticksToRpm;
    wheel[i].rpm_meas = rpm_meas_raw;   // giữ lại để debug

    WheelState &w = wheel[i];

    // ----- PREDICT -----
    // x_pred: rpm += dt * drpm; drpm giữ nguyên
    w.rpm_hat += dt * w.drpm_hat;

    // P_pred = F * P * F^T + Q
    float P00 = w.P[0][0] + dt * (w.P[0][1] + w.P[1][0]) + dt2 * w.P[1][1] + Q11;
    float P01 = w.P[0][1] + dt * w.P[1][1] + Q12;
    float P10 = w.P[1][0] + dt * w.P[1][1] + Q12;
    float P11 = w.P[1][1] + Q22;

    // ----- UPDATE -----
    float y = rpm_meas_raw - w.rpm_hat;
    float S = P00 + KALMAN_R_RPM;
    float K0 = P00 / S;
    float K1 = P10 / S;

    w.rpm_hat  += K0 * y;
    w.drpm_hat += K1 * y;

    // P = (I - K*H) * P_pred
    w.P[0][0] = (1.0f - K0) * P00;
    w.P[0][1] = (1.0f - K0) * P01;
    w.P[1][0] = P10 - K1 * P00;
    w.P[1][1] = P11 - K1 * P01;
  }
}

// [V2-KALMAN] PID = uFF + KS + KP*err + KD*(-drpm_hat) + I
//   - rpm_meas đầu vào giờ là rpm_hat (đã lọc Kalman, mượt + ít trễ)
//   - D-term DÙNG drpm_hat (derivative on measurement) - không dùng derivative on error
//     -> tránh "derivative kick" khi target nhảy bậc
//     -> drpm_hat đã được Kalman lọc nên không cần thêm low-pass
//     -> Dấu âm: khi bánh đang tăng tốc nhanh, ta giảm command để damping
float updateWheelController(WheelState &w, float rpmRef, float dt) {
  w.rpm_ref = rpmRef;

  if (fabs(rpmRef) < WHEEL_REF_ZERO_RPM && fabs(w.rpm_hat) < WHEEL_MEAS_ZERO_RPM) {
    w.integrator_pwm = 0.0f;
    return 0.0f;
  }

  float err = rpmRef - w.rpm_hat;

  float uFF = WHEEL_KV_PWM_PER_RPM * rpmRef;
  float refAbs = fabs(rpmRef);
  if (refAbs > WHEEL_REF_ZERO_RPM) {
    float ksBlend = smoothstep01((refAbs - WHEEL_REF_ZERO_RPM) /
                                 max(WHEEL_KS_FULL_BLEND_RPM - WHEEL_REF_ZERO_RPM, 1e-3f));
    float ksScale = WHEEL_KS_MIN_BLEND + (1.0f - WHEEL_KS_MIN_BLEND) * ksBlend;
    uFF += signNonZero(rpmRef) * (WHEEL_KS_PWM * ksScale);
  }

  float p = WHEEL_KP_PWM_PER_RPM * err;
  float d = -WHEEL_KD_PWM_PER_RPM_S * w.drpm_hat;   // [V2-KALMAN] D trên measurement
  float uPre = uFF + p + d + w.integrator_pwm;
  float uSat = constrain(uPre, -255.0f, 255.0f);

  bool allowIntegrate = (fabs(uPre - uSat) < 0.5f) ||
                        ((uPre > uSat) && (err < 0.0f)) ||
                        ((uPre < uSat) && (err > 0.0f));

  if (allowIntegrate) {
    w.integrator_pwm += WHEEL_KI_PWM_PER_RPM_S * err * dt;
    w.integrator_pwm = constrain(w.integrator_pwm, -WHEEL_I_LIMIT_PWM, WHEEL_I_LIMIT_PWM);
  }

  uPre = uFF + p + d + w.integrator_pwm;
  return constrain(uPre, -255.0f, 255.0f);
}

// ==================== Command shaping and heading control ====================
void readCommandTargets() {
  ps2x.read_gamepad(false, 0);

  debugRawLY = ps2x.Analog(PSS_LY);
  debugRawLX = ps2x.Analog(PSS_LX);
  debugRawRX = ps2x.Analog(PSS_RX);

  float vx = applyDeadzone(debugRawLY);
  float vy = applyDeadzone(debugRawLX);
  float wz = -applyDeadzone(debugRawRX);

  bool dpadUp = ps2x.Button(PSB_PAD_UP);
  bool dpadDown = ps2x.Button(PSB_PAD_DOWN);
  bool dpadLeft = ps2x.Button(PSB_PAD_LEFT);
  bool dpadRight = ps2x.Button(PSB_PAD_RIGHT);

  if (dpadUp || dpadDown || dpadLeft || dpadRight) {
    vx = 0.0f;
    vy = 0.0f;
    if (dpadUp) vx -= DPAD_SPEED;
    if (dpadDown) vx += DPAD_SPEED;
    if (dpadLeft) vy -= DPAD_SPEED;
    if (dpadRight) vy += DPAD_SPEED;

    float mag = sqrt(vx * vx + vy * vy);
    if (mag > 1.0f) {
      vx /= mag;
      vy /= mag;
    }
  }

  cmd.vxTarget = vx;
  cmd.vyTarget = vy;
  cmd.wzTarget = wz;
}

void shapeCommands(float dt) {
  float xyLevel = max(fabs(cmd.vx), fabs(cmd.vy));
  float xyBlend = smoothstep01(xyLevel / CMD_SOFT_START_BLEND);
  float wzBlend = smoothstep01(fabs(cmd.wz) / CMD_SOFT_START_BLEND);

  float accelXY = CMD_ACCEL_XY_START + (CMD_ACCEL_XY_RUN - CMD_ACCEL_XY_START) * xyBlend;
  float accelWZ = CMD_ACCEL_WZ_START + (CMD_ACCEL_WZ_RUN - CMD_ACCEL_WZ_START) * wzBlend;

  cmd.vx = slewToward(cmd.vx, cmd.vxTarget, accelXY * dt);
  cmd.vy = slewToward(cmd.vy, cmd.vyTarget, accelXY * dt);
  cmd.wz = slewToward(cmd.wz, cmd.wzTarget, accelWZ * dt);
}

void resetHeadingControllerToCurrentYaw() {
  heading.active = false;
  heading.integrator = 0.0f;
  heading.engageTimer = 0.0f;
  heading.targetYaw = imu.yawEst;
}

float computeYawRateCommand(float vxCmd, float vyCmd, float wzRaw, float wzCmd, float dt) {
  float manualRate = wzCmd * MAX_ROTATE_RATE_RAD_S;

  if (!ENABLE_HEADING_HOLD) {
    heading.active = false;
    heading.manualRotate = false;
    heading.integrator = 0.0f;
    return manualRate;
  }

  if (!imu.connected || !imu.hasQuat || !imu.yawEstimatorInit) {
    heading.active = false;
    heading.manualRotate = false;
    heading.integrator = 0.0f;
    return manualRate;
  }

  if (heading.manualRotate) {
    if (fabs(wzRaw) < ROTATE_EXIT_DB) heading.manualRotate = false;
  } else {
    if (fabs(wzRaw) > ROTATE_ENTER_DB) heading.manualRotate = true;
  }

  bool translating = vecAbs1(vxCmd, vyCmd) > TRANSLATE_ACTIVE_DB;

  if (heading.manualRotate || !translating) {
    resetHeadingControllerToCurrentYaw();
    return manualRate;
  }

  heading.engageTimer += dt;
  if (!heading.active) {
    if (heading.engageTimer < HEADING_CAPTURE_DELAY_S) {
      heading.targetYaw = imu.yawEst;
      return manualRate;
    }

    heading.active = true;
    heading.targetYaw = imu.yawEst;
    heading.integrator = 0.0f;
    return manualRate;
  }

  float yawErr = shapeErrorWithDeadband(angleDiff(heading.targetYaw, imu.yawEst), HEADING_ERR_DB_RAD);

  if (yawErr == 0.0f) {
    heading.integrator *= 0.97f;
  } else {
    heading.integrator += HEADING_KI * yawErr * dt;
    heading.integrator = constrain(heading.integrator, -HEADING_I_LIMIT, HEADING_I_LIMIT);
  }

  float holdRate = HEADING_KP * yawErr + heading.integrator - HEADING_KD * imu.gyroZFilt;
  holdRate = constrain(holdRate, -MAX_HOLD_CORR_RATE_RAD_S, MAX_HOLD_CORR_RATE_RAD_S);

  return manualRate + holdRate;
}

// ==================== Setup ====================
void setup() {
  Serial.begin(115200);
  delay(300);

  if (ENABLE_IMU) {
    Wire.begin();
    Wire.setClock(IMU_I2C_CLOCK_HZ);

    imu.connected = beginImu();
    if (imu.connected) {
      enableImuReports();
      Serial.println(F("[IMU] Ready"));
    } else {
      Serial.println(F("[IMU] Not found. Robot can still drive without heading hold."));
    }
  } else {
    imu.connected = false;
    imu.hasQuat = false;
    imu.hasGyro = false;
    imu.yawEstimatorInit = false;
    Serial.println(F("[IMU] Disabled for controller isolation test"));
  }

  byte ps2Error = ps2x.config_gamepad(PS2_CLK, PS2_CMD, PS2_ATT, PS2_DAT, false, true);
  if (ps2Error != 0) {
    Serial.print(F("[PS2] Config error code: "));
    Serial.println(ps2Error);
    while (1) {
    }
  }

  for (int i = 0; i < 4; ++i) {
    pinMode(ENC_A[i], INPUT_PULLUP);
  }

  pinMode(22, INPUT_PULLUP);
  pinMode(23, INPUT_PULLUP);
  pinMode(24, INPUT_PULLUP);
  pinMode(25, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENC_A[0]), enc_isr0, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_A[1]), enc_isr1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_A[2]), enc_isr2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_A[3]), enc_isr3, CHANGE);

  pinMode(M1_L, OUTPUT); pinMode(M1_R, OUTPUT);
  pinMode(M2_L, OUTPUT); pinMode(M2_R, OUTPUT);
  pinMode(M3_L, OUTPUT); pinMode(M3_R, OUTPUT);
  pinMode(M4_L, OUTPUT); pinMode(M4_R, OUTPUT);

  resetWheelControllers();
  hardStopAll();

  if (ENABLE_IMU) {
    for (int i = 0; i < 60; ++i) {
      updateImu();
      delay(5);
    }
    zeroYawReference();
  }

  lastCmdMs = millis();
  lastControlUs = micros();

  Serial.println(F("Modern X-drive controller ready [V2-KALMAN]"));
  Serial.println(F("SELECT: recapture yaw zero"));
  Serial.println(F("START : hard stop"));
}

// ==================== Main loop ====================
void loop() {
  if (ENABLE_IMU) updateImu();

  uint32_t nowUs = micros();
  uint32_t elapsedUs = nowUs - lastControlUs;
  if (elapsedUs < CONTROL_PERIOD_US) return;

  float dt = elapsedUs * 1.0e-6f;
  lastControlUs = nowUs;

  if (dt <= 0.0f || dt > 0.05f) {
    dt = CONTROL_PERIOD_US * 1.0e-6f;
  }

  if (ENABLE_IMU) stepYawEstimator(dt);
  readCommandTargets();

  if (ps2x.ButtonPressed(PSB_SELECT)) {
    zeroYawReference();
    Serial.println(F("[IMU] Yaw zero recaptured"));
  }

  if (ps2x.ButtonPressed(PSB_START)) {
    cmd.vxTarget = 0.0f;
    cmd.vyTarget = 0.0f;
    cmd.wzTarget = 0.0f;
    cmd.vx = 0.0f;
    cmd.vy = 0.0f;
    cmd.wz = 0.0f;
    hardStopAll();
    resetHeadingControllerToCurrentYaw();
    lastCmdMs = millis();
    return;
  }

  bool inputActive = (fabs(cmd.vxTarget) + fabs(cmd.vyTarget) + fabs(cmd.wzTarget)) > 0.01f;
  if (!inputActive) {
    cmd.vxTarget = 0.0f;
    cmd.vyTarget = 0.0f;
    cmd.wzTarget = 0.0f;
    cmd.vx = 0.0f;
    cmd.vy = 0.0f;
    cmd.wz = 0.0f;
    hardStopAll();
    resetHeadingControllerToCurrentYaw();
    lastCmdMs = millis();
    return;
  }

  if (inputActive) {
    lastCmdMs = millis();
  } else if (millis() - lastCmdMs > COMMAND_TIMEOUT_MS) {
    cmd.vxTarget = 0.0f;
    cmd.vyTarget = 0.0f;
    cmd.wzTarget = 0.0f;
  }

  shapeCommands(dt);
  updateWheelMeasurements(dt);

  float yawRateCmd = computeYawRateCommand(cmd.vx, cmd.vy, cmd.wzTarget, cmd.wz, dt);
  float turnScale = (vecAbs1(cmd.vx, cmd.vy) < 0.06f) ? TURN_IN_PLACE_SCALE : TURN_WHILE_MOVING_SCALE;
  float wzMix = clampUnit((yawRateCmd / MAX_ROTATE_RATE_RAD_S) * turnScale);

  float fl =  cmd.vx - cmd.vy + wzMix;
  float fr =  cmd.vx + cmd.vy - wzMix;
  float bl = -cmd.vx - cmd.vy - wzMix;
  float br = -cmd.vx + cmd.vy + wzMix;

  float norm = max4(fl, fr, bl, br);
  if (norm > 1.0f) {
    fl /= norm;
    fr /= norm;
    bl /= norm;
    br /= norm;
  }

  float ref0 = fl * WHEEL_RPM_REF_MAX;
  float ref1 = fr * WHEEL_RPM_REF_MAX;
  float ref2 = bl * WHEEL_RPM_REF_MAX;
  float ref3 = br * WHEEL_RPM_REF_MAX;

  debugRef[0] = ref0;
  debugRef[1] = ref1;
  debugRef[2] = ref2;
  debugRef[3] = ref3;

  float pwm0 = updateWheelController(wheel[0], ref0, dt);
  float pwm1 = updateWheelController(wheel[1], ref1, dt);
  float pwm2 = updateWheelController(wheel[2], ref2, dt);
  float pwm3 = updateWheelController(wheel[3], ref3, dt);

  debugPwm[0] = pwm0;
  debugPwm[1] = pwm1;
  debugPwm[2] = pwm2;
  debugPwm[3] = pwm3;

  bool nearStopCmd = (fabs(ref0) + fabs(ref1) + fabs(ref2) + fabs(ref3)) < 20.0f;
  bool nearStopMeas = true;
  for (int i = 0; i < 4; ++i) {
    if (fabs(wheel[i].rpm_hat) > WHEEL_MEAS_ZERO_RPM) {   // [V2-KALMAN] dùng Kalman estimate
      nearStopMeas = false;
      break;
    }
  }

  if (nearStopCmd && nearStopMeas) {
    applyMotorPwm(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 4; ++i) {
      wheel[i].integrator_pwm = 0.0f;
      kalmanResetWheelStateOnly(wheel[i]);   // [V2-KALMAN] reset state, giữ P
    }
  } else {
    applyMotorPwm(pwm0, pwm1, pwm2, pwm3);
  }

  // if (ENABLE_DEBUG && (millis() - lastDebugMs >= DEBUG_PRINT_MS)) {
  //   lastDebugMs = millis();
  //   Serial.print(F("raw_lx="));
  //   Serial.print(debugRawLX);
  //   Serial.print(F(" raw_ly="));
  //   Serial.print(debugRawLY);
  //   Serial.print(F(" raw_rx="));
  //   Serial.print(debugRawRX);
  //   Serial.print(F(" tgt_vx="));
  //   Serial.print(cmd.vxTarget, 3);
  //   Serial.print(F(" tgt_vy="));
  //   Serial.print(cmd.vyTarget, 3);
  //   Serial.print(F(" tgt_wz="));
  //   Serial.print(cmd.wzTarget, 3);
  //   Serial.print(F(" cmd_vx="));
  //   Serial.print(cmd.vx, 3);
  //   Serial.print(F(" cmd_vy="));
  //   Serial.print(cmd.vy, 3);
  //   Serial.print(F(" cmd_wz="));
  //   Serial.print(cmd.wz, 3);
  //   Serial.print(F(" ref0="));
  //   Serial.print(debugRef[0], 1);
  //   Serial.print(F(" ref1="));
  //   Serial.print(debugRef[1], 1);
  //   Serial.print(F(" pwm0="));
  //   Serial.print(debugPwm[0], 1);
  //   Serial.print(F(" pwm1="));
  //   Serial.print(debugPwm[1], 1);
  //   Serial.print(F("yaw_est_deg="));
  //   Serial.print(imu.yawEst * 180.0f / PI, 2);
  //   Serial.print(F(" yaw_bno_deg="));
  //   Serial.print(imu.yawBno * 180.0f / PI, 2);
  //   Serial.print(F(" gyro_z="));
  //   Serial.print(imu.gyroZFilt, 3);
  //   Serial.print(F(" tgt_deg="));
  //   Serial.print(heading.targetYaw * 180.0f / PI, 2);
  //   Serial.print(F(" hold="));
  //   Serial.print(heading.active ? 1 : 0);
  //   Serial.print(F(" cmd_wz="));
  //   Serial.print(cmd.wz, 3);
  //   Serial.print(F(" rpm0="));
  //   Serial.print(wheel[0].rpm_hat, 1);
  //   Serial.print(F(" rpm1="));
  //   Serial.print(wheel[1].rpm_hat, 1);
  //   Serial.println();
  // }
}
