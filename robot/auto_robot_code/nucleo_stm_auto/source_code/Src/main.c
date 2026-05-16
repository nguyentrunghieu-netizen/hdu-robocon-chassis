/*
 * main.c - STM32F446RE bare-metal cho Mecanum Robot
 * Ported logic from mecanum_base_v3.ino
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "bno08x.h"

#ifndef PI
#define PI 3.141592653589793f
#endif

// ==================== CẤU HÌNH THANH GHI (BARE-METAL) ====================
#define RCC_BASE        0x40023800U
#define RCC_AHB1ENR     (*(volatile uint32_t *)(RCC_BASE + 0x30U))
#define RCC_APB1ENR     (*(volatile uint32_t *)(RCC_BASE + 0x40U))
#define RCC_APB2ENR     (*(volatile uint32_t *)(RCC_BASE + 0x44U))

#define GPIOA_BASE      0x40020000U
#define GPIOB_BASE      0x40020400U
#define GPIOC_BASE      0x40020800U

#define GPIOA_MODER     (*(volatile uint32_t *)(GPIOA_BASE + 0x00U))
#define GPIOA_PUPDR     (*(volatile uint32_t *)(GPIOA_BASE + 0x0CU))
#define GPIOA_IDR       (*(volatile uint32_t *)(GPIOA_BASE + 0x10U))
#define GPIOA_ODR       (*(volatile uint32_t *)(GPIOA_BASE + 0x14U))
#define GPIOA_AFRL      (*(volatile uint32_t *)(GPIOA_BASE + 0x20U))

#define GPIOB_MODER     (*(volatile uint32_t *)(GPIOB_BASE + 0x00U))
#define GPIOB_PUPDR     (*(volatile uint32_t *)(GPIOB_BASE + 0x0CU))
#define GPIOB_IDR       (*(volatile uint32_t *)(GPIOB_BASE + 0x10U))
#define GPIOB_BSRR      (*(volatile uint32_t *)(GPIOB_BASE + 0x18U))

#define GPIOC_MODER     (*(volatile uint32_t *)(GPIOC_BASE + 0x00U))
#define GPIOC_PUPDR     (*(volatile uint32_t *)(GPIOC_BASE + 0x0CU))
#define GPIOC_IDR       (*(volatile uint32_t *)(GPIOC_BASE + 0x10U))
#define GPIOC_BSRR      (*(volatile uint32_t *)(GPIOC_BASE + 0x18U))

#define SYSCFG_BASE      0x40013800U
#define SYSCFG_EXTICR1   (*(volatile uint32_t *)(SYSCFG_BASE + 0x08U))

#define EXTI_BASE       0x40013C00U
#define EXTI_IMR        (*(volatile uint32_t *)(EXTI_BASE + 0x00U))
#define EXTI_RTSR       (*(volatile uint32_t *)(EXTI_BASE + 0x08U))
#define EXTI_FTSR       (*(volatile uint32_t *)(EXTI_BASE + 0x0CU))
#define EXTI_PR         (*(volatile uint32_t *)(EXTI_BASE + 0x14U))

#define NVIC_ISER0      (*(volatile uint32_t *)0xE000E100U)

#define USART2_BASE     0x40004400U
#define USART2_SR       (*(volatile uint32_t *)(USART2_BASE + 0x00U))
#define USART2_DR       (*(volatile uint32_t *)(USART2_BASE + 0x04U))
#define USART2_BRR      (*(volatile uint32_t *)(USART2_BASE + 0x08U))
#define USART2_CR1      (*(volatile uint32_t *)(USART2_BASE + 0x0CU))

#define DWT_CTRL        (*(volatile uint32_t *)0xE0001000U)
#define DWT_CYCCNT      (*(volatile uint32_t *)0xE0001004U)
#define DEMCR           (*(volatile uint32_t *)0xE000EDFCU)

// ==================== THÔNG SỐ CƠ KHÍ & ĐIỀU KHIỂN ====================
const float TICKS_PER_REV       = 500.0f;  
float WHEEL_DIAMETER_M          = 0.100f;           
float WHEEL_CIRCUMFERENCE       = PI * 0.100f;     
float ROTATION_RADIUS_M         = 0.300f;          

float RPM_MAX = 729.0f;  
const uint32_t CTRL_DT_MS = 10;
const float TAU_P_FILT = 0.01f;
const float TAU_WHEEL_MEAS = 0.04f;

const float WHEEL_KP_PWM_PER_RPM = 0.20f;
const float WHEEL_KI_PWM_PER_RPM_S = 0.75f;
float WHEEL_KV_PWM_PER_RPM = 255.0f / 729.0f;  
float WHEEL_KS_PWM = 22.0f;                      
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

const uint32_t CMD_TIMEOUT_MS = 500;

// ==================== BIẾN TOÀN CỤC ====================
static volatile long enc_count[4] = {0, 0, 0, 0};
static long prev_enc_count[4]     = {0, 0, 0, 0};

float targetRPM[4]       = {0, 0, 0, 0};
float actualRPM[4]       = {0, 0, 0, 0};
float actualRPMFilt[4]   = {0, 0, 0, 0};
float wheelIntegralPwm[4]= {0, 0, 0, 0};
float filtRef[4]         = {0, 0, 0, 0};
float pwmOutput[4]       = {0, 0, 0, 0};

float vxTarget = 0.0f, vyTarget = 0.0f, wzTarget = 0.0f;
float vxCmd = 0.0f, vyCmd = 0.0f, wzCmd = 0.0f;

bool imu_connected = false;
bool imu_has_yaw = false;
bool imu_has_gyro = false;
bool imu_pending_offset = true;
float imu_yaw_raw_rad = 0.0f;
float imu_yaw_offset_rad = 0.0f;
float imu_yaw_rad = 0.0f;
float imu_wz_rad_s = 0.0f;

float odom_x_m = 0.0f, odom_y_m = 0.0f, odom_theta_rad = 0.0f;
float robot_vx_mps = 0.0f, robot_vy_mps = 0.0f, robot_wz_rad_s = 0.0f, robot_wz_enc_rad_s = 0.0f;

bool headingLocked = false;
float headingTargetRad = 0.0f;
float headingIntegral = 0.0f;
bool externalHeadingControl = false;

uint32_t lastCmdTime = 0;

bool calibrationMode = false;
float calibrationPwm[4] = {0, 0, 0, 0};

char serialBuf[64];
uint8_t serialIdx = 0;
static char print_buffer[128];

BNO08x_Data imu;
BNO08x_Estimator est;

// ==================== TIỆN ÍCH CƠ BẢN ====================
static uint32_t micros(void) { return DWT_CYCCNT / 16U; }
static uint32_t millis(void) { return DWT_CYCCNT / 16000U; }

static char *f2s(float val, int dec, char *buf) {
    char sign = (val < 0.0f) ? '-' : ' ';
    val = fabsf(val);
    int i_part = (int)val;
    if (dec == 1)      sprintf(buf, "%c%d.%d", sign, i_part, (int)(val * 10.0f) % 10);
    else if (dec == 2) sprintf(buf, "%c%d.%02d", sign, i_part, (int)(val * 100.0f) % 100);
    else               sprintf(buf, "%c%d.%04d", sign, i_part, (int)(val * 10000.0f) % 10000);
    if (buf[0] == ' ') memmove(buf, buf + 1, strlen(buf));
    return buf;
}

static void serial_print(const char *str) {
    while (*str) {
        while (!(USART2_SR & (1U << 7))) {}
        USART2_DR = (uint32_t)(uint8_t)(*str++);
    }
}

static bool serial_receive(uint8_t *c) {
    if (USART2_SR & (1U << 5)) {
        *c = (uint8_t)USART2_DR;
        return true;
    }
    return false;
}

float wrapAngle(float a) {
    while (a >  PI) a -= 2.0f * PI;
    while (a < -PI) a += 2.0f * PI;
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
    float delta = clampAbs(target - current, maxDelta);
    return current + delta;
}

float rpmToWheelMps(float wheel_rpm) { return wheel_rpm * WHEEL_CIRCUMFERENCE / 60.0f; }
float wheelMpsToRpm(float wheel_mps) { return wheel_mps * 60.0f / WHEEL_CIRCUMFERENCE; }

// ==================== KHỞI TẠO PHẦN CỨNG ====================
static void hardware_init(void) {
    RCC_AHB1ENR |= (1U << 0) | (1U << 1) | (1U << 2);
    RCC_APB1ENR |= (1U << 17);
    RCC_APB2ENR |= (1U << 14);
    (void)RCC_AHB1ENR;

    DEMCR |= (1U << 24);
    DWT_CYCCNT = 0U;
    DWT_CTRL |= 1U;

    // USART2: PA2 (TX), PA3 (RX)
    GPIOA_MODER &= ~(0xFU << 4);
    GPIOA_MODER |= (0xAU << 4);
    GPIOA_AFRL &= ~(0xFFU << 8);
    GPIOA_AFRL |= (0x77U << 8);
    USART2_BRR = 0x008BU; // 115200 at 16MHz
    USART2_CR1 = (1U << 13) | (1U << 3) | (1U << 2);

    // Encoders: A = PA0, PA1, PC2, PC3 | B = PB4, PB5, PB6, PB7
    GPIOA_MODER &= ~(0xFU);
    GPIOA_PUPDR &= ~(0xFU); GPIOA_PUPDR |= (0x5U);

    GPIOC_MODER &= ~(0xFU << 4);
    GPIOC_PUPDR &= ~(0xFU << 4); GPIOC_PUPDR |= (0x5U << 4);

    GPIOB_MODER &= ~(0xFF00U);
    GPIOB_PUPDR &= ~(0xFF00U); GPIOB_PUPDR |= (0x5500U);

    // EXTI cho Encoder A
    SYSCFG_EXTICR1 &= ~(0xFFFFU);
    SYSCFG_EXTICR1 |= (0x2200U); // PA0, PA1, PC2, PC3
    EXTI_IMR |= 0x0FU;
    EXTI_RTSR |= 0x0FU;
    EXTI_FTSR |= 0x0FU;
    NVIC_ISER0 |= (1U << 6) | (1U << 7) | (1U << 8) | (1U << 9);

    // Motor Pins (Output cho logic PWM, TODO: Cấu hình Hardware Timer sau)
    // Motor 1: PC0, PC1 | Motor 2: PC4, PC5 | Motor 3: PB0, PB1 | Motor 4: PB2, PB10
    GPIOC_MODER &= ~((3U << 0) | (3U << 2) | (3U << 8) | (3U << 10));
    GPIOC_MODER |=  ((1U << 0) | (1U << 2) | (1U << 8) | (1U << 10));

    GPIOB_MODER &= ~((3U << 0) | (3U << 2) | (3U << 4) | (3U << 20));
    GPIOB_MODER |=  ((1U << 0) | (1U << 2) | (1U << 4) | (1U << 20));
}

// ==================== NGẮT ENCODER ====================
void EXTI0_IRQHandler(void) {
    if (EXTI_PR & (1U << 0)) {
        EXTI_PR = (1U << 0);
        uint8_t a = (GPIOA_IDR & (1U << 0)) ? 1U : 0U;
        uint8_t b = (GPIOB_IDR & (1U << 4)) ? 1U : 0U;
        enc_count[0] += (a != b) ? 1 : -1;
    }
}
void EXTI1_IRQHandler(void) {
    if (EXTI_PR & (1U << 1)) {
        EXTI_PR = (1U << 1);
        uint8_t a = (GPIOA_IDR & (1U << 1)) ? 1U : 0U;
        uint8_t b = (GPIOB_IDR & (1U << 5)) ? 1U : 0U;
        enc_count[1] += (a != b) ? -1 : 1;
    }
}
void EXTI2_IRQHandler(void) {
    if (EXTI_PR & (1U << 2)) {
        EXTI_PR = (1U << 2);
        uint8_t a = (GPIOC_IDR & (1U << 2)) ? 1U : 0U;
        uint8_t b = (GPIOB_IDR & (1U << 6)) ? 1U : 0U;
        enc_count[2] += (a != b) ? -1 : 1;
    }
}
void EXTI3_IRQHandler(void) {
    if (EXTI_PR & (1U << 3)) {
        EXTI_PR = (1U << 3);
        uint8_t a = (GPIOC_IDR & (1U << 3)) ? 1U : 0U;
        uint8_t b = (GPIOB_IDR & (1U << 7)) ? 1U : 0U;
        enc_count[3] += (a != b) ? 1 : -1;
    }
}

// ==================== ĐIỀU KHIỂN MOTOR ====================
void setMotorSignedPwm(uint8_t motor_id, float pwmSigned) {
    float pwmAbs = fabsf(pwmSigned);
    if (pwmAbs < PWM_ZERO_DB) pwmAbs = 0;
    
    // TODO: Áp dụng pwmAbs vào thanh ghi CCR của Timer tương ứng.
    // Dưới đây là mô phỏng set HIGH/LOW (Digital On/Off) để giữ đúng logic hàm.
    bool fwd = (pwmSigned >= 0.0f && pwmAbs > 0);
    bool rev = (pwmSigned < 0.0f && pwmAbs > 0);

    switch(motor_id) {
        case 0:
            if(fwd)      { GPIOC_BSRR = (1U << 0); GPIOC_BSRR = (1U << 17); } // PC0=1, PC1=0
            else if(rev) { GPIOC_BSRR = (1U << 16); GPIOC_BSRR = (1U << 1); } // PC0=0, PC1=1
            else         { GPIOC_BSRR = (1U << 16); GPIOC_BSRR = (1U << 17); }
            break;
        case 1:
            if(fwd)      { GPIOC_BSRR = (1U << 4); GPIOC_BSRR = (1U << 21); } 
            else if(rev) { GPIOC_BSRR = (1U << 20); GPIOC_BSRR = (1U << 5); }
            else         { GPIOC_BSRR = (1U << 20); GPIOC_BSRR = (1U << 21); }
            break;
        case 2:
            if(fwd)      { GPIOB_BSRR = (1U << 0); GPIOB_BSRR = (1U << 17); } 
            else if(rev) { GPIOB_BSRR = (1U << 16); GPIOB_BSRR = (1U << 1); }
            else         { GPIOB_BSRR = (1U << 16); GPIOB_BSRR = (1U << 17); }
            break;
        case 3:
            if(fwd)      { GPIOB_BSRR = (1U << 2); GPIOB_BSRR = (1U << 26); } // PB2=1, PB10=0
            else if(rev) { GPIOB_BSRR = (1U << 18); GPIOB_BSRR = (1U << 10); }
            else         { GPIOB_BSRR = (1U << 18); GPIOB_BSRR = (1U << 26); }
            break;
    }
}

void applyMotorPwm(float p0, float p1, float p2, float p3) {
    setMotorSignedPwm(0, p0);
    setMotorSignedPwm(1, p1);
    setMotorSignedPwm(2, p2);
    setMotorSignedPwm(3, p3);
}

void captureIMUYawOffset(void) {
    if (imu_has_yaw) {
        imu_yaw_offset_rad = imu_yaw_raw_rad;
        imu_yaw_rad = 0.0f;
    } else {
        imu_pending_offset = true;
    }
    headingLocked = false;
    headingIntegral = 0.0f;
}

void hardStopAll(void) {
    applyMotorPwm(0.0f, 0.0f, 0.0f, 0.0f);
    __asm volatile("cpsid i");
    for (int i = 0; i < 4; ++i) prev_enc_count[i] = enc_count[i];
    __asm volatile("cpsie i");

    vxTarget = 0.0f; vyTarget = 0.0f; wzTarget = 0.0f;
    vxCmd = 0.0f; vyCmd = 0.0f; wzCmd = 0.0f;
    headingLocked = false;
    headingIntegral = 0.0f;
    for (int i = 0; i < 4; ++i) {
        targetRPM[i] = 0.0f; actualRPM[i] = 0.0f;
        actualRPMFilt[i] = 0.0f; wheelIntegralPwm[i] = 0.0f;
        filtRef[i] = 0.0f; pwmOutput[i] = 0.0f;
    }
}

void resetOdometry(void) {
    __asm volatile("cpsid i");
    for (int i = 0; i < 4; ++i) { enc_count[i] = 0; prev_enc_count[i] = 0; }
    __asm volatile("cpsie i");

    odom_x_m = 0.0f; odom_y_m = 0.0f; odom_theta_rad = 0.0f;
    robot_vx_mps = 0.0f; robot_vy_mps = 0.0f; robot_wz_rad_s = 0.0f;
    captureIMUYawOffset();
}

// ==================== TOÁN HỌC ROBOT ====================
void mecanumIK(float vx, float vy, float omega) {
    float v_fl =  vx - vy + ROTATION_RADIUS_M * omega;
    float v_fr =  vx + vy - ROTATION_RADIUS_M * omega;
    float v_bl = -vx - vy - ROTATION_RADIUS_M * omega;
    float v_br = -vx + vy + ROTATION_RADIUS_M * omega;

    float factor = 60.0f / WHEEL_CIRCUMFERENCE;
    targetRPM[0] = v_fl * factor;
    targetRPM[1] = v_fr * factor;
    targetRPM[2] = v_bl * factor;
    targetRPM[3] = v_br * factor;

    float maxReq = 0;
    for (int i = 0; i < 4; i++) {
        if (fabsf(targetRPM[i]) > maxReq) maxReq = fabsf(targetRPM[i]);
    }
    if (maxReq > RPM_MAX) {
        float scale = RPM_MAX / maxReq;
        for (int i = 0; i < 4; i++) targetRPM[i] *= scale;
    }
}

float updateWheelController(float rpmRef, float rpmMeas, float *integralPwm, float *refFilt, float dt) {
    *refFilt += (rpmRef - *refFilt) * dt / TAU_P_FILT;
    *refFilt = fminf(fmaxf(*refFilt, -RPM_MAX), RPM_MAX);

    if (fabsf(*refFilt) < WHEEL_REF_ZERO_RPM && fabsf(rpmMeas) < WHEEL_MEAS_ZERO_RPM) {
        *integralPwm = 0.0f;
        *refFilt = 0.0f;
        return 0.0f;
    }

    float err = *refFilt - rpmMeas;
    float uFF = WHEEL_KV_PWM_PER_RPM * (*refFilt);
    if (fabsf(*refFilt) > WHEEL_REF_ZERO_RPM) {
        uFF += signNonZero(*refFilt) * WHEEL_KS_PWM;
    }

    float p = WHEEL_KP_PWM_PER_RPM * err;
    float uPre = uFF + p + *integralPwm;
    float uSat = fminf(fmaxf(uPre, -255.0f), 255.0f);

    bool allowIntegrate = (fabsf(uPre - uSat) < 0.5f) ||
                          ((uPre > uSat) && (err < 0.0f)) ||
                          ((uPre < uSat) && (err > 0.0f));

    if (allowIntegrate) {
        *integralPwm += WHEEL_KI_PWM_PER_RPM_S * err * dt;
        *integralPwm = fminf(fmaxf(*integralPwm, -WHEEL_I_LIMIT_PWM), WHEEL_I_LIMIT_PWM);
    }

    uPre = uFF + p + *integralPwm;
    return fminf(fmaxf(uPre, -255.0f), 255.0f);
}

void updateOdometry(float dt) {
    float v_fl = rpmToWheelMps(actualRPMFilt[0]);
    float v_fr = rpmToWheelMps(actualRPMFilt[1]);
    float v_bl = rpmToWheelMps(actualRPMFilt[2]);
    float v_br = rpmToWheelMps(actualRPMFilt[3]);

    float vx_local = ( v_fl + v_fr - v_bl - v_br) * 0.25f;
    float vy_local = (-v_fl + v_fr - v_bl + v_br) * 0.25f;
    float wz_encoder = ( v_fl - v_fr - v_bl + v_br) / (4.0f * ROTATION_RADIUS_M);
    float wz = imu_has_gyro ? imu_wz_rad_s : wz_encoder;

    if (imu_has_yaw) {
        odom_theta_rad = imu_yaw_rad;
    } else {
        odom_theta_rad = wrapAngle(odom_theta_rad + wz * dt);
    }

    float cos_th = cosf(odom_theta_rad);
    float sin_th = sinf(odom_theta_rad);

    odom_x_m += (vx_local * cos_th - vy_local * sin_th) * dt;
    odom_y_m += (vx_local * sin_th + vy_local * cos_th) * dt;
    robot_vx_mps = vx_local;
    robot_vy_mps = vy_local;
    robot_wz_rad_s = wz;
    robot_wz_enc_rad_s = wz_encoder;
}

float applyHeadingControl(float vxCmdLocal, float vyCmdLocal, float wzCmdLocal, float dt) {
    if (externalHeadingControl) {
        headingLocked = false;
        headingIntegral = 0.0f;
        return clampAbs(wzCmdLocal, 3.14f);
    }

    if (!imu_has_yaw || !imu_has_gyro) {
        headingLocked = false;
        headingIntegral = 0.0f;
        return wzCmdLocal;
    }

    bool hasLinear = (fabsf(vxCmdLocal) > LIN_MOVE_THRESHOLD) ||
                     (fabsf(vyCmdLocal) > LIN_MOVE_THRESHOLD);
    bool omegaNearZero = fabsf(wzCmdLocal) < OMEGA_CMD_DEADBAND;

    if (omegaNearZero && hasLinear) {
        if (!headingLocked) {
            headingTargetRad = imu_yaw_rad;
            headingLocked = true;
            headingIntegral = 0.0f;
        }
        float headingErr = wrapAngle(headingTargetRad - imu_yaw_rad);
        headingIntegral += KI_HEADING_HOLD * headingErr * dt;
        headingIntegral = clampAbs(headingIntegral, MAX_HEADING_INTEGRAL);

        float omegaFix = KP_HEADING_HOLD * headingErr + headingIntegral - KD_HEADING_HOLD * imu_wz_rad_s;
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

// ==================== SERIAL COMMANDS ====================
static bool parseThreeFloats(const char *str, float *a, float *b, float *c) {
    char *end;
    *a = strtof(str, &end); if (end == str) return false; str = end;
    *b = strtof(str, &end); if (end == str) return false; str = end;
    *c = strtof(str, &end); if (end == str) return false;
    return true;
}

static bool parseFourFloats(const char *str, float *a, float *b, float *c, float *d) {
    char *end;
    *a = strtof(str, &end); if (end == str) return false; str = end;
    *b = strtof(str, &end); if (end == str) return false; str = end;
    *c = strtof(str, &end); if (end == str) return false; str = end;
    *d = strtof(str, &end); if (end == str) return false;
    return true;
}

static bool parseTwoFloats(const char *str, float *a, float *b) {
    char *end;
    *a = strtof(str, &end); if (end == str) return false; str = end;
    *b = strtof(str, &end); if (end == str) return false;
    return true;
}

static bool parseOneFloat(const char *str, float *a) {
    char *end;
    *a = strtof(str, &end);
    return end != str;
}

void processCommand(const char* cmd) {
    if (cmd[0] == 'V' && cmd[1] == ' ') {
        float vx = 0, vy = 0, omega = 0;
        if (parseThreeFloats(cmd + 2, &vx, &vy, &omega)) {
            vxCmd = fminf(fmaxf(vx, -1.0f), 1.0f);
            vyCmd = fminf(fmaxf(vy, -1.0f), 1.0f);
            wzCmd = fminf(fmaxf(omega, -3.14f), 3.14f);
            vxTarget = vxCmd; vyTarget = vyCmd; wzTarget = wzCmd;
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
        serial_print("ODOM_RESET\r\n");
    }
    else if (cmd[0] == 'P' && cmd[1] == ' ') {
        float p0=0, p1=0, p2=0, p3=0;
        if (parseFourFloats(cmd + 2, &p0, &p1, &p2, &p3)) {
            calibrationPwm[0] = fminf(fmaxf(p0, -255.0f), 255.0f);
            calibrationPwm[1] = fminf(fmaxf(p1, -255.0f), 255.0f);
            calibrationPwm[2] = fminf(fmaxf(p2, -255.0f), 255.0f);
            calibrationPwm[3] = fminf(fmaxf(p3, -255.0f), 255.0f);
            if (p0 == 0 && p1 == 0 && p2 == 0 && p3 == 0) {
                calibrationMode = false;
                hardStopAll();
                for (int i = 0; i < 4; i++) calibrationPwm[i] = 0;
            } else {
                calibrationMode = true;
                for (int i = 0; i < 4; i++) wheelIntegralPwm[i] = 0.0f;
            }
            lastCmdTime = millis();
        }
    }
    else if (cmd[0] == 'Q') {
        char s0[16], s1[16], s2[16], s3[16];
        sprintf(print_buffer, "QR %s %s %s %s\r\n", 
            f2s(actualRPMFilt[0], 2, s0), f2s(actualRPMFilt[1], 2, s1),
            f2s(actualRPMFilt[2], 2, s2), f2s(actualRPMFilt[3], 2, s3));
        serial_print(print_buffer);
    }
    else if (strncmp(cmd, "KV ", 3) == 0) {
        float new_kv=0, new_ks=0;
        if (parseTwoFloats(cmd + 3, &new_kv, &new_ks) && new_kv > 0 && new_ks >= 0) {
            WHEEL_KV_PWM_PER_RPM = new_kv;
            WHEEL_KS_PWM = new_ks;
        }
    }
    else if (strncmp(cmd, "RPMMAX ", 7) == 0) {
        float new_rpm_max=0;
        if (parseOneFloat(cmd + 7, &new_rpm_max) && new_rpm_max > 10.0f && new_rpm_max < 5000.0f) {
            RPM_MAX = new_rpm_max;
        }
    }
    else if (strncmp(cmd, "RADIUS ", 7) == 0) {
        float new_r=0;
        if (parseOneFloat(cmd + 7, &new_r) && new_r > 0.01f && new_r < 1.0f) {
            ROTATION_RADIUS_M = new_r;
        }
    }
    else if (strncmp(cmd, "WHEEL ", 6) == 0) {
        float new_d=0;
        if (parseOneFloat(cmd + 6, &new_d) && new_d > 0.01f && new_d < 0.5f) {
            WHEEL_DIAMETER_M = new_d;
            WHEEL_CIRCUMFERENCE = PI * new_d;
        }
    }
    else if (strncmp(cmd, "IMU_OFFSET ", 11) == 0) {
        float target = strtof(cmd + 11, NULL);
        if (imu_has_yaw) {
            imu_yaw_offset_rad = imu_yaw_raw_rad - target;
            imu_yaw_rad = wrapAngle(imu_yaw_raw_rad - imu_yaw_offset_rad);
            headingLocked = false;
        }
    }
    else if (strncmp(cmd, "HEADCTRL ", 9) == 0) {
        long mode = strtol(cmd + 9, NULL, 10);
        externalHeadingControl = (mode != 0);
        headingLocked = false;
        headingIntegral = 0.0f;
    }
}

// ==================== MAIN ====================
int main(void) {
    *(volatile uint32_t *)0xE000ED88U |= (0xFU << 20U); // Enable FPU
    
    hardware_init();
    bno_hw_init();
    __asm volatile("cpsie i");

    memset(&imu, 0, sizeof(imu));
    memset(&est, 0, sizeof(est));

    BNO08x_EstimatorConfig est_cfg = {
        .yaw_sign = 1.0f,
        .gyro_tau_s = 0.025f,
        .correction_gain = 0.10f,
    };
    bno_estimator_config(&est_cfg);

    imu_connected = (bno_init() != 0U);
    if (imu_connected) {
        bno_enable(BNO_RPT_GAMEROT, 10000U);
        bno_enable(BNO_RPT_GYRO, 10000U);
        bno_zero_yaw(&est);
        serial_print("[IMU] San sang!\r\n");
    } else {
        serial_print("[IMU] Khong the ket noi BNO08x. Su dung Odometry thuan tuy.\r\n");
    }

    hardStopAll();
    resetOdometry();
    lastCmdTime = millis();
    serial_print("MECANUM_READY\r\n");

    uint32_t last_measure_us = micros();
    uint32_t last_imu_poll_ms = millis();

    while (1) {
        // --- Đọc Serial ---
        uint8_t c;
        if (serial_receive(&c)) {
            if (c == '\n' || c == '\r') {
                if (serialIdx > 0) {
                    serialBuf[serialIdx] = '\0';
                    processCommand(serialBuf);
                    serialIdx = 0;
                }
            } else {
                if (serialIdx < sizeof(serialBuf) - 1) {
                    serialBuf[serialIdx++] = (char)c;
                } else {
                    serialIdx = 0;
                }
            }
        }

        // --- Đọc BNO08x (Polling) ---
        if (imu_connected && (uint32_t)(millis() - last_imu_poll_ms) >= 5U) {
            last_imu_poll_ms = millis();
            bno_update(&imu);
        }

        // --- Vòng PID 100Hz (10ms) ---
        uint32_t now_us = micros();
        uint32_t elapsed_us = now_us - last_measure_us;
        if (elapsed_us >= (CTRL_DT_MS * 1000U)) {
            float dt = (float)elapsed_us * 1.0e-6f;
            if (dt <= 0.0f || dt > 0.1f) dt = (float)(CTRL_DT_MS * 1000U) * 1.0e-6f;
            last_measure_us = now_us;

            // 1. Đo lường encoder
            long encNow[4];
            __asm volatile("cpsid i");
            for (int i = 0; i < 4; i++) encNow[i] = enc_count[i];
            __asm volatile("cpsie i");

            float ticks_to_rpm = 60.0f / (TICKS_PER_REV * dt);
            float alpha = dt / (TAU_WHEEL_MEAS + dt);
            for (int i = 0; i < 4; i++) {
                long delta = encNow[i] - prev_enc_count[i];
                if (labs(delta) > 10000) delta = 0;
                actualRPM[i] = delta * ticks_to_rpm;
                actualRPMFilt[i] += alpha * (actualRPM[i] - actualRPMFilt[i]);
                prev_enc_count[i] = encNow[i];
            }

            // 2. Cập nhật IMU State
            if (imu_connected) {
                bno_estimator_update(&imu, dt, &est);
                imu_has_yaw = true;
                imu_has_gyro = true;
                imu_yaw_raw_rad = est.yaw_raw_rad;
                
                if (imu_pending_offset) {
                    imu_yaw_offset_rad = imu_yaw_raw_rad;
                    imu_pending_offset = false;
                    serial_print("[IMU] Chup offset yaw\r\n");
                }
                
                imu_yaw_rad = wrapAngle(imu_yaw_raw_rad - imu_yaw_offset_rad);
                imu_wz_rad_s = est.gyro_z_filt_rad_s;
            }

            // 3. Tính toán Odometry & Timeout
            updateOdometry(dt);

            if (millis() - lastCmdTime > CMD_TIMEOUT_MS) {
                vxTarget = 0.0f; vyTarget = 0.0f; wzTarget = 0.0f;
            }

            vxCmd = slewToward(vxCmd, vxTarget, CMD_SLEW_VX_MPS_S * dt);
            vyCmd = slewToward(vyCmd, vyTarget, CMD_SLEW_VY_MPS_S * dt);
            wzCmd = slewToward(wzCmd, wzTarget, CMD_SLEW_WZ_RAD_S2 * dt);

            // 4. IK và PID
            float wzApplied = applyHeadingControl(vxCmd, vyCmd, wzCmd, dt);
            mecanumIK(vxCmd, vyCmd, wzApplied);

            for (int i = 0; i < 4; i++) {
                pwmOutput[i] = updateWheelController(targetRPM[i], actualRPMFilt[i], &wheelIntegralPwm[i], &filtRef[i], dt);
            }

            bool nearStopCmd = (fabsf(targetRPM[0]) + fabsf(targetRPM[1]) + fabsf(targetRPM[2]) + fabsf(targetRPM[3])) < 20.0f;
            bool nearStopMeas = true;
            for (int i = 0; i < 4; ++i) {
                if (fabsf(actualRPMFilt[i]) > WHEEL_MEAS_ZERO_RPM) {
                    nearStopMeas = false;
                    break;
                }
            }

            if (calibrationMode) {
                applyMotorPwm(calibrationPwm[0], calibrationPwm[1], calibrationPwm[2], calibrationPwm[3]);
            } else if (nearStopCmd && nearStopMeas) {
                applyMotorPwm(0.0f, 0.0f, 0.0f, 0.0f);
                for (int i = 0; i < 4; ++i) wheelIntegralPwm[i] = 0.0f;
            } else {
                applyMotorPwm(pwmOutput[0], pwmOutput[1], pwmOutput[2], pwmOutput[3]);
            }

            // 5. Gửi Telemetry (Feedback)
            static uint8_t fbCount = 0;
            if (++fbCount >= 10) {
                fbCount = 0;
                char s_vx[16], s_vy[16], s_wz[16], s_x[16], s_y[16], s_th[16], s_wz_enc[16];
                sprintf(print_buffer, "T %s %s %s %s %s %s %d %s\r\n",
                    f2s(robot_vx_mps, 3, s_vx),
                    f2s(robot_vy_mps, 3, s_vy),
                    f2s(robot_wz_rad_s, 3, s_wz),
                    f2s(odom_x_m, 3, s_x),
                    f2s(odom_y_m, 3, s_y),
                    f2s(odom_theta_rad * 180.0f / PI, 2, s_th),
                    imu_connected ? 1 : 0,
                    f2s(robot_wz_enc_rad_s, 4, s_wz_enc));
                serial_print(print_buffer);
            }
        }
    }
}
