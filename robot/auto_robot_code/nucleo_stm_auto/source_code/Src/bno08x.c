#include "bno08x.h"
#include <string.h>
#include <math.h>

#ifndef PI
#define PI 3.141592653589793f
#endif

#ifndef USE_HAL_DRIVER

/* Bare-metal STM32F446 I2C1 driver for BNO08x on PB8/PB9. */
#define RCC_BASE_BM      0x40023800UL
#define GPIOB_BASE_BM    0x40020400UL
#define I2C1_BASE_BM     0x40005400UL

#define RCC_AHB1ENR_BM   (*(volatile uint32_t *)(RCC_BASE_BM  + 0x30U))
#define RCC_APB1ENR_BM   (*(volatile uint32_t *)(RCC_BASE_BM  + 0x40U))

#define GPIOB_MODER_BM   (*(volatile uint32_t *)(GPIOB_BASE_BM + 0x00U))
#define GPIOB_OTYPER_BM  (*(volatile uint32_t *)(GPIOB_BASE_BM + 0x04U))
#define GPIOB_OSPEEDR_BM (*(volatile uint32_t *)(GPIOB_BASE_BM + 0x08U))
#define GPIOB_PUPDR_BM   (*(volatile uint32_t *)(GPIOB_BASE_BM + 0x0CU))
#define GPIOB_AFRH_BM    (*(volatile uint32_t *)(GPIOB_BASE_BM + 0x24U))
#define GPIOB_BSRR_BM    (*(volatile uint32_t *)(GPIOB_BASE_BM + 0x18U))

#define I2C1_CR1_BM      (*(volatile uint32_t *)(I2C1_BASE_BM + 0x00U))
#define I2C1_CR2_BM      (*(volatile uint32_t *)(I2C1_BASE_BM + 0x04U))
#define I2C1_DR_BM       (*(volatile uint32_t *)(I2C1_BASE_BM + 0x10U))
#define I2C1_SR1_BM      (*(volatile uint32_t *)(I2C1_BASE_BM + 0x14U))
#define I2C1_SR2_BM      (*(volatile uint32_t *)(I2C1_BASE_BM + 0x18U))
#define I2C1_CCR_BM      (*(volatile uint32_t *)(I2C1_BASE_BM + 0x1CU))
#define I2C1_TRISE_BM    (*(volatile uint32_t *)(I2C1_BASE_BM + 0x20U))

#define I2C_BM_SB        (1U << 0)
#define I2C_BM_ADDR      (1U << 1)
#define I2C_BM_BTF       (1U << 2)
#define I2C_BM_RXNE      (1U << 6)
#define I2C_BM_TXE       (1U << 7)
#define I2C_BM_AF        (1U << 10)
#define I2C_BM_PE        (1U << 0)
#define I2C_BM_START     (1U << 8)
#define I2C_BM_STOP      (1U << 9)
#define I2C_BM_ACK       (1U << 10)
#define I2C_BM_SWRST     (1U << 15)

static uint8_t g_bm_addr = 0x4AU;
static uint8_t g_bm_seq[6];
static uint8_t g_bm_rx[BNO_MAX_PKT + 4U];

static BNO08x_EstimatorConfig g_est_cfg = { 1.0f, 0.025f, 0.10f };
static float g_yaw_offset_rad = 0.0f;
static uint8_t g_zero_pending = 1U;

static void bno_delay_ms(volatile uint32_t ms) {
    for (; ms > 0U; ms--) {
        volatile uint32_t t = 8000U;
        while (t-- > 0U) {}
    }
}

void bno_hw_init(void) {
    RCC_AHB1ENR_BM |= (1U << 1);
    RCC_APB1ENR_BM |= (1U << 21);

    GPIOB_MODER_BM   &= ~(0xFU << 16U);
    GPIOB_MODER_BM   |=  (0x5U << 16U);
    GPIOB_OTYPER_BM  |=  (3U   <<  8U);
    GPIOB_OSPEEDR_BM |=  (0xFU << 16U);
    GPIOB_PUPDR_BM   &= ~(0xFU << 16U);
    GPIOB_PUPDR_BM   |=  (0x5U << 16U);

    for (int i = 0; i < 9; i++) {
        GPIOB_BSRR_BM = (1U << 24U);
        bno_delay_ms(1);
        GPIOB_BSRR_BM = (1U << 8U);
        bno_delay_ms(1);
    }
    GPIOB_BSRR_BM = (1U << 25U);
    bno_delay_ms(1);
    GPIOB_BSRR_BM = (1U << 8U);
    bno_delay_ms(1);
    GPIOB_BSRR_BM = (1U << 9U);
    bno_delay_ms(2);

    GPIOB_MODER_BM &= ~(0xFU << 16U);
    GPIOB_MODER_BM |=  (0xAU << 16U);
    GPIOB_AFRH_BM  &= ~(0xFFU);
    GPIOB_AFRH_BM  |=  (0x44U);

    I2C1_CR1_BM |= I2C_BM_SWRST;
    I2C1_CR1_BM &= ~I2C_BM_SWRST;
    I2C1_CR2_BM = 16U;
    I2C1_CCR_BM = 80U;
    I2C1_TRISE_BM = 17U;
    I2C1_CR1_BM = I2C_BM_PE;
}

static int bm_i2c_wait(uint32_t flag) {
    uint32_t to = 80000U;
    while (!(I2C1_SR1_BM & flag)) {
        if (--to == 0U) return -1;
    }
    return 0;
}

static void bm_i2c_recover(void) {
    I2C1_CR1_BM |= I2C_BM_STOP;
    volatile uint32_t t = 8000U;
    while (t-- > 0U) {}
    I2C1_SR1_BM &= ~I2C_BM_AF;
    I2C1_CR1_BM |= I2C_BM_SWRST;
    I2C1_CR1_BM &= ~I2C_BM_SWRST;
    I2C1_CR2_BM = 16U;
    I2C1_CCR_BM = 80U;
    I2C1_TRISE_BM = 17U;
    I2C1_CR1_BM = I2C_BM_PE;
}

static int bm_i2c_write(uint8_t addr7, const uint8_t *data, uint16_t len) {
    I2C1_CR1_BM |= I2C_BM_START;
    if (bm_i2c_wait(I2C_BM_SB) < 0) {
        bm_i2c_recover();
        return -1;
    }
    I2C1_DR_BM = (uint32_t)(addr7 << 1U);
    if (bm_i2c_wait(I2C_BM_ADDR) < 0) {
        I2C1_SR1_BM &= ~I2C_BM_AF;
        I2C1_CR1_BM |= I2C_BM_STOP;
        return -1;
    }
    (void)I2C1_SR2_BM;
    for (uint16_t i = 0U; i < len; i++) {
        if (bm_i2c_wait(I2C_BM_TXE) < 0) {
            I2C1_CR1_BM |= I2C_BM_STOP;
            return -1;
        }
        I2C1_DR_BM = data[i];
    }
    if (bm_i2c_wait(I2C_BM_BTF) < 0) {
        I2C1_CR1_BM |= I2C_BM_STOP;
        return -1;
    }
    I2C1_CR1_BM |= I2C_BM_STOP;
    return 0;
}

static int bm_i2c_read(uint8_t addr7, uint8_t *data, uint16_t len) {
    if (len == 0U) return 0;
    I2C1_CR1_BM |= I2C_BM_ACK | I2C_BM_START;
    if (bm_i2c_wait(I2C_BM_SB) < 0) {
        bm_i2c_recover();
        return -1;
    }
    I2C1_DR_BM = (uint32_t)((addr7 << 1U) | 1U);
    if (bm_i2c_wait(I2C_BM_ADDR) < 0) {
        I2C1_SR1_BM &= ~I2C_BM_AF;
        I2C1_CR1_BM |= I2C_BM_STOP;
        return -1;
    }

    if (len == 1U) {
        I2C1_CR1_BM &= ~I2C_BM_ACK;
        (void)I2C1_SR2_BM;
        I2C1_CR1_BM |= I2C_BM_STOP;
        if (bm_i2c_wait(I2C_BM_RXNE) < 0) {
            bm_i2c_recover();
            return -1;
        }
        data[0] = (uint8_t)I2C1_DR_BM;
        return 0;
    }

    (void)I2C1_SR2_BM;
    for (uint16_t i = 0U; i < len; i++) {
        if (i == (uint16_t)(len - 1U)) {
            I2C1_CR1_BM &= ~I2C_BM_ACK;
            I2C1_CR1_BM |= I2C_BM_STOP;
        }
        if (bm_i2c_wait(I2C_BM_RXNE) < 0) {
            bm_i2c_recover();
            return -1;
        }
        data[i] = (uint8_t)I2C1_DR_BM;
    }
    return 0;
}

static uint16_t shtp_recv(void) {
    uint8_t hdr[4];
    if (bm_i2c_read(g_bm_addr, hdr, 4U) < 0) return 0U;

    uint16_t pkt_len = (uint16_t)hdr[0] | ((uint16_t)(hdr[1] & 0x7FU) << 8U);
    if (pkt_len < 4U) return 0U;
    if (pkt_len > (uint16_t)sizeof(g_bm_rx)) pkt_len = (uint16_t)sizeof(g_bm_rx);

    if (bm_i2c_read(g_bm_addr, g_bm_rx, pkt_len) < 0) return 0U;
    return (uint16_t)(pkt_len - 4U);
}

static uint8_t bno_report_len(uint8_t rid) {
    switch (rid) {
    case BNO_RPT_ACCEL:   return 10U;
    case BNO_RPT_GYRO:    return 10U;
    case BNO_RPT_ROTVEC:  return 14U;
    case BNO_RPT_GAMEROT: return 12U;
    default:              return 0U;
    }
}

static float bno_wrap_rad(float r) {
    while (r >  PI) r -= 2.0f * PI;
    while (r < -PI) r += 2.0f * PI;
    return r;
}

static void bno_quat_to_euler(float qi, float qj, float qk, float qr,
                              float *roll_rad, float *pitch_rad, float *yaw_rad) {
    float qn = sqrtf(qi * qi + qj * qj + qk * qk + qr * qr);
    if (qn > 0.001f) {
        qi /= qn;
        qj /= qn;
        qk /= qn;
        qr /= qn;
    }

    float sinr = 2.0f * (qr * qi + qj * qk);
    float cosr = 1.0f - 2.0f * (qi * qi + qj * qj);
    float sinp = 2.0f * (qr * qj - qk * qi);
    float siny = 2.0f * (qr * qk + qi * qj);
    float cosy = 1.0f - 2.0f * (qj * qj + qk * qk);

    if (roll_rad) *roll_rad = atan2f(sinr, cosr);
    if (pitch_rad) {
        if (sinp > 1.0f) *pitch_rad = PI * 0.5f;
        else if (sinp < -1.0f) *pitch_rad = -PI * 0.5f;
        else *pitch_rad = asinf(sinp);
    }
    if (yaw_rad) *yaw_rad = atan2f(siny, cosy);
}

static int16_t rd_i16(const uint8_t *p) {
    return (int16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8U));
}

static void bno_parse_report(const uint8_t *p, uint16_t avail, BNO08x_Data *d) {
    if (avail < 10U) return;

    uint8_t rid = p[0];
    const uint8_t *dt = &p[4];

    switch (rid) {
    case BNO_RPT_ACCEL:
        d->raw_ax = rd_i16(&dt[0]);
        d->raw_ay = rd_i16(&dt[2]);
        d->raw_az = rd_i16(&dt[4]);
        d->ax = (float)d->raw_ax / (float)(1U << BNO_ACCEL_Q);
        d->ay = (float)d->raw_ay / (float)(1U << BNO_ACCEL_Q);
        d->az = (float)d->raw_az / (float)(1U << BNO_ACCEL_Q);
        d->new_accel = 1U;
        break;

    case BNO_RPT_GYRO:
        d->raw_gx = rd_i16(&dt[0]);
        d->raw_gy = rd_i16(&dt[2]);
        d->raw_gz = rd_i16(&dt[4]);
        d->gx = (float)d->raw_gx / (float)(1U << BNO_GYRO_Q);
        d->gy = (float)d->raw_gy / (float)(1U << BNO_GYRO_Q);
        d->gz = (float)d->raw_gz / (float)(1U << BNO_GYRO_Q);
        d->new_gyro = 1U;
        break;

    case BNO_RPT_GAMEROT:
    case BNO_RPT_ROTVEC: {
        if (rid == BNO_RPT_GAMEROT && avail < 12U) return;
        if (rid == BNO_RPT_ROTVEC && avail < 14U) return;

        d->raw_qi = rd_i16(&dt[0]);
        d->raw_qj = rd_i16(&dt[2]);
        d->raw_qk = rd_i16(&dt[4]);
        d->raw_qr = rd_i16(&dt[6]);

        d->qi = (float)d->raw_qi / (float)(1U << BNO_QUAT_Q);
        d->qj = (float)d->raw_qj / (float)(1U << BNO_QUAT_Q);
        d->qk = (float)d->raw_qk / (float)(1U << BNO_QUAT_Q);
        d->qr = (float)d->raw_qr / (float)(1U << BNO_QUAT_Q);
        d->qacc = (rid == BNO_RPT_ROTVEC) ? ((float)rd_i16(&dt[8]) / (float)(1U << BNO_ACC_Q)) : 0.0f;

        float qn = sqrtf(d->qi * d->qi + d->qj * d->qj + d->qk * d->qk + d->qr * d->qr);
        if (qn > 0.001f) {
            d->qi /= qn;
            d->qj /= qn;
            d->qk /= qn;
            d->qr /= qn;
        }

        float roll, pitch, yaw;
        bno_quat_to_euler(d->qi, d->qj, d->qk, d->qr, &roll, &pitch, &yaw);
        d->roll = roll * 180.0f / PI;
        d->pitch = pitch * 180.0f / PI;
        d->yaw = yaw * 180.0f / PI;
        d->new_rotvec = 1U;
        break;
    }

    default:
        break;
    }
}

uint8_t bno_init(void) {
    memset(g_bm_seq, 0, sizeof(g_bm_seq));
    bno_delay_ms(600U);

    uint8_t tmp[4];
    if (bm_i2c_read(0x4AU, tmp, 4U) == 0) {
        g_bm_addr = 0x4AU;
    } else {
        bno_delay_ms(20U);
        g_bm_addr = (bm_i2c_read(0x4BU, tmp, 4U) == 0) ? 0x4BU : 0x00U;
    }
    if (g_bm_addr == 0x00U) return 0x00U;

    (void)shtp_recv();
    bno_delay_ms(50U);
    return g_bm_addr;
}

void bno_enable(uint8_t rptID, uint32_t intervalUs) {
    uint8_t tx[21] = {
        21, 0, 2, 0, 0xFD, rptID, 0, 0, 0,
        (uint8_t)(intervalUs & 0xFFU),
        (uint8_t)((intervalUs >> 8) & 0xFFU),
        (uint8_t)((intervalUs >> 16) & 0xFFU),
        (uint8_t)((intervalUs >> 24) & 0xFFU),
        0, 0, 0, 0, 0, 0, 0, 0
    };
    tx[3] = g_bm_seq[2]++;
    (void)bm_i2c_write(g_bm_addr, tx, sizeof(tx));
    bno_delay_ms(50U);
}

int bno_update(BNO08x_Data *d) {
    if (!d) return 0;
    d->new_rotvec = 0U;
    d->new_gyro = 0U;
    d->new_accel = 0U;

    uint16_t pay_len = shtp_recv();
    if (pay_len == 0U) return 0;
    if (g_bm_rx[2] != 3U) return 0;

    d->pkt_len = (pay_len <= (uint16_t)sizeof(d->pkt)) ? pay_len : (uint16_t)sizeof(d->pkt);
    memcpy(d->pkt, &g_bm_rx[4], d->pkt_len);

    const uint8_t *p = &g_bm_rx[4];
    uint16_t off = 0U;
    while (off < pay_len) {
        uint8_t rid = p[off];
        if (rid == 0xFBU || rid == 0xFAU) {
            off += 5U;
            continue;
        }

        uint8_t rlen = bno_report_len(rid);
        if (rlen == 0U || (uint16_t)(off + rlen) > pay_len) break;

        bno_parse_report(&p[off], (uint16_t)(pay_len - off), d);
        off += rlen;
    }

    return (d->new_accel || d->new_gyro || d->new_rotvec) ? 1 : 0;
}

void bno_estimator_config(const BNO08x_EstimatorConfig *cfg) {
    if (!cfg) return;
    g_est_cfg = *cfg;
    if (g_est_cfg.yaw_sign == 0.0f) g_est_cfg.yaw_sign = 1.0f;
    if (g_est_cfg.gyro_tau_s < 0.0f) g_est_cfg.gyro_tau_s = 0.0f;
    if (g_est_cfg.correction_gain < 0.0f) g_est_cfg.correction_gain = 0.0f;
    if (g_est_cfg.correction_gain > 1.0f) g_est_cfg.correction_gain = 1.0f;
}

void bno_zero_yaw(BNO08x_Estimator *est) {
    g_yaw_offset_rad = 0.0f;
    g_zero_pending = 1U;
    if (est) {
        est->yaw_bno_rad = 0.0f;
        est->yaw_est_rad = 0.0f;
        est->yaw_zeroed = 0U;
    }
}

void bno_estimator_update(const BNO08x_Data *d, float dt_s, BNO08x_Estimator *est) {
    if (!d || !est) return;
    if (dt_s <= 0.0f || dt_s > 0.5f) dt_s = 0.01f;

    if (d->new_rotvec) {
        est->yaw_raw_rad = g_est_cfg.yaw_sign * (d->yaw * PI / 180.0f);
        est->roll_rad = d->roll * PI / 180.0f;
        est->pitch_rad = d->pitch * PI / 180.0f;
        est->has_yaw = 1U;
        if (g_zero_pending) {
            g_yaw_offset_rad = est->yaw_raw_rad;
            est->yaw_est_rad = 0.0f;
            est->yaw_zeroed = 1U;
            g_zero_pending = 0U;
        }
        est->yaw_bno_rad = bno_wrap_rad(est->yaw_raw_rad - g_yaw_offset_rad);
    }

    if (d->new_gyro) {
        est->gyro_x_raw_rad_s = d->gx;
        est->gyro_y_raw_rad_s = d->gy;
        est->gyro_z_raw_rad_s = g_est_cfg.yaw_sign * d->gz;
        est->has_gyro = 1U;
    }

    if (!est->yaw_zeroed) return;

    float alpha = (g_est_cfg.gyro_tau_s <= 0.0f) ? 1.0f : (dt_s / (g_est_cfg.gyro_tau_s + dt_s));
    est->gyro_x_filt_rad_s += alpha * (est->gyro_x_raw_rad_s - est->gyro_x_filt_rad_s);
    est->gyro_y_filt_rad_s += alpha * (est->gyro_y_raw_rad_s - est->gyro_y_filt_rad_s);
    est->gyro_z_filt_rad_s += alpha * (est->gyro_z_raw_rad_s - est->gyro_z_filt_rad_s);
    est->yaw_est_rad = bno_wrap_rad(est->yaw_est_rad + est->gyro_z_filt_rad_s * dt_s);
    if (est->has_yaw) {
        est->yaw_est_rad = bno_wrap_rad(est->yaw_est_rad +
            g_est_cfg.correction_gain * bno_wrap_rad(est->yaw_bno_rad - est->yaw_est_rad));
    }
}

uint16_t bno_get_last_packet(const BNO08x_Data *d, const uint8_t **payload) {
    if (!d || !payload) return 0U;
    *payload = d->pkt;
    return d->pkt_len;
}

uint8_t BNO_Init(void) {
    return bno_init();
}

void BNO_EnableReport(uint8_t rptID, uint32_t intervalUs) {
    bno_enable(rptID, intervalUs);
}

int BNO_Update(BNO08x_Data *d) {
    return bno_update(d);
}

#endif /* !USE_HAL_DRIVER */
