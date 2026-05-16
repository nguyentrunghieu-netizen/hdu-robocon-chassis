/**
 ******************************************************************************
 * @file    bno08x.h
 * @brief   Driver cho BNO08x IMU - tuong thich STM32 HAL (I2C)
 *
 * Cach su dung:
 *   1. Tao project STM32CubeIDE, bat I2C trong CubeMX
 *   2. Them bno08x.h (Inc/) va bno08x.c (Src/) vao project
 *   3. Trong main.c:
 *        #include "bno08x.h"
 *        BNO08x_t imu;
 *        BNO08x_Init(&imu, &hi2c1, BNO08X_ADDR_DEFAULT, 100);
 *        BNO08x_EnableReport(&imu, BNO08X_REPT_ROTATION_VECTOR, 10000);
 *        while(1) {
 *            BNO08x_Update(&imu);
 *            float roll, pitch, yaw;
 *            BNO08x_QuatToEulerDeg(&imu, &roll, &pitch, &yaw);
 *        }
 *
 * Ket noi phan cung (I2C):
 *   BNO08x VCC  -> 3.3V
 *   BNO08x GND  -> GND
 *   BNO08x SDA  -> PB7 (I2C1_SDA) hoac theo cau hinh CubeMX
 *   BNO08x SCL  -> PB6 (I2C1_SCL) hoac theo cau hinh CubeMX
 *   BNO08x ADR  -> GND  (dia chi 0x4A), hoac 3.3V (0x4B)
 *   BNO08x INT  -> tuy chon (doc khi co du lieu san sang)
 *   BNO08x RST  -> tuy chon (reset phan cung)
 ******************************************************************************
 */

#ifndef INC_BNO08X_H_
#define INC_BNO08X_H_

/* Chi compile khi dung STM32Cube HAL project.
 * Neu la bare-metal project, khong can file nay. */
#ifdef USE_HAL_DRIVER

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------
 * Doi voi du an STM32Cube, include nay duoc them tu dong.
 * Neu dung thu vien ngoai du an Cube, doi thanh:
 *   #include "stm32f4xx_hal.h"   (F4)
 *   #include "stm32f1xx_hal.h"   (F1)
 *   v.v.
 * ------------------------------------------------------------ */
#include "stm32f4xx_hal.h"
#include <stdint.h>
#include <stdbool.h>

/* ============================================================
 *  Dia chi I2C (8-bit, da dich cho HAL)
 * ============================================================ */
#define BNO08X_ADDR_DEFAULT    (0x4AU << 1U)  /* Chan ADR = GND */
#define BNO08X_ADDR_ALTERNATE  (0x4BU << 1U)  /* Chan ADR = VCC */

/* ============================================================
 *  Feature Report IDs (loai cam bien can kich hoat)
 * ============================================================ */
#define BNO08X_REPT_ACCELEROMETER         0x01U  /* Gia toc (m/s^2)          */
#define BNO08X_REPT_GYROSCOPE             0x02U  /* Toc do goc (rad/s)       */
#define BNO08X_REPT_MAGNETIC_FIELD        0x03U  /* Tu truong (uT)           */
#define BNO08X_REPT_LINEAR_ACCELERATION   0x04U  /* Gia toc tuyen tinh       */
#define BNO08X_REPT_ROTATION_VECTOR       0x05U  /* Quaternion 9DOF          */
#define BNO08X_REPT_GRAVITY               0x06U  /* Trong luc (m/s^2)        */
#define BNO08X_REPT_GAME_ROTATION_VECTOR  0x08U  /* Quaternion 6DOF (no mag) */
#define BNO08X_REPT_STEP_COUNTER          0x11U  /* Dem buoc chan            */
#define BNO08X_REPT_STABILITY_CLASS       0x13U  /* Phan loai trang thai     */
#define BNO08X_REPT_GYRO_ROTATION_VECTOR  0x14U  /* Gyro-integrated RV      */

/* ============================================================
 *  Cau truc du lieu
 * ============================================================ */

/** Vector 3D (Gia toc, Gyro, Tu truong, Trong luc) */
typedef struct {
    float x;
    float y;
    float z;
} BNO08x_Vec3;

/** Quaternion (Rotation Vector) */
typedef struct {
    float i;
    float j;
    float k;
    float real;
    float accuracy;   /* rad - chi co trong ROTATION_VECTOR (9DOF) */
} BNO08x_Quat;

/** Tap hop tat ca du lieu cam bien */
typedef struct {
    BNO08x_Vec3 accel;      /* Gia toc toan phan (m/s^2) */
    BNO08x_Vec3 gyro;       /* Toc do goc (rad/s)        */
    BNO08x_Vec3 mag;        /* Tu truong (uT)            */
    BNO08x_Vec3 linAccel;   /* Gia toc tuyen tinh (m/s^2)*/
    BNO08x_Vec3 gravity;    /* Vector trong luc (m/s^2)  */
    BNO08x_Quat rotVec;     /* Quaternion 9DOF           */
    BNO08x_Quat gameRot;    /* Quaternion 6DOF           */

    uint8_t accelStatus;    /* 0-3: do chinh xac calibration */
    uint8_t gyroStatus;
    uint8_t magStatus;
    uint8_t rotVecStatus;
} BNO08x_Data;

/* ============================================================
 *  Handle thiet bi
 * ============================================================ */
#define BNO08X_MAX_PKT_SIZE  128U   /* Byte toi da 1 goi tin SHTP */

typedef struct {
    I2C_HandleTypeDef *hi2c;           /* Con tro I2C HAL handle        */
    uint16_t           devAddr;        /* Dia chi 8-bit (da dich)       */
    uint32_t           timeout;        /* Timeout I2C (ms)              */
    uint8_t            seqNum[6];      /* So thu tu SHTP theo kenh      */
    uint8_t            rxBuf[BNO08X_MAX_PKT_SIZE];
    uint8_t            txBuf[BNO08X_MAX_PKT_SIZE];
    BNO08x_Data        data;           /* Du lieu cam bien moi nhat     */
    uint8_t            lastRptID;      /* ID report cuoi cung nhan duoc */
} BNO08x_t;

/* ============================================================
 *  Ham API
 * ============================================================ */

/**
 * @brief  Khoi tao BNO08x, kiem tra ket noi I2C.
 * @param  dev       Con tro den handle BNO08x_t
 * @param  hi2c      Con tro I2C HAL handle (vd: &hi2c1)
 * @param  addr      Dia chi I2C (BNO08X_ADDR_DEFAULT hoac ALTERNATE)
 * @param  timeoutMs Timeout cho tung giao dich I2C (ms), khuyen nghi 100
 * @retval HAL_OK neu thanh cong
 */
HAL_StatusTypeDef BNO08x_Init(BNO08x_t *dev, I2C_HandleTypeDef *hi2c,
                               uint16_t addr, uint32_t timeoutMs);

/**
 * @brief  Soft-reset BNO08x qua lenh SHTP.
 */
HAL_StatusTypeDef BNO08x_Reset(BNO08x_t *dev);

/**
 * @brief  Kich hoat mot loai cam bien voi chu ky lay mau mong muon.
 * @param  reportID   ID loai cam bien (BNO08X_REPT_*)
 * @param  intervalUs Chu ky lay mau tinh bang microseconds
 *                    (vd: 10000 = 10ms = 100Hz)
 */
HAL_StatusTypeDef BNO08x_EnableReport(BNO08x_t *dev, uint8_t reportID,
                                       uint32_t intervalUs);

/**
 * @brief  Doc 1 goi tin tu BNO08x va cap nhat dev->data.
 *         Goi trong vong lap chinh hoac khi co ngat INT.
 * @retval HAL_OK    : Nhan goi tin thanh cong (co the co hoac khong co du lieu moi)
 *         HAL_TIMEOUT: Chua co du lieu san sang
 *         HAL_ERROR  : Loi I2C
 */
HAL_StatusTypeDef BNO08x_Update(BNO08x_t *dev);

/* ---- Ham lay du lieu (goi sau BNO08x_Update) ---- */

void BNO08x_GetAccel(const BNO08x_t *dev, float *x, float *y, float *z);
void BNO08x_GetGyro(const BNO08x_t *dev,  float *x, float *y, float *z);
void BNO08x_GetMag(const BNO08x_t *dev,   float *x, float *y, float *z);
void BNO08x_GetLinearAccel(const BNO08x_t *dev, float *x, float *y, float *z);

void BNO08x_GetRotationQuat(const BNO08x_t *dev,
                             float *i, float *j, float *k,
                             float *real, float *accuracy);

void BNO08x_GetGameRotQuat(const BNO08x_t *dev,
                            float *i, float *j, float *k, float *real);

/**
 * @brief  Chuyen quaternion rotation vector thanh goc Euler (don vi: do).
 *         Phai kich hoat BNO08X_REPT_ROTATION_VECTOR truoc khi dung.
 * @param  roll   Goc lan (X-axis), -180 den +180 do
 * @param  pitch  Goc nghieng (Y-axis), -90 den +90 do
 * @param  yaw    Goc la huong (Z-axis), -180 den +180 do
 */
void BNO08x_QuatToEulerDeg(const BNO08x_t *dev,
                            float *roll, float *pitch, float *yaw);

#ifdef __cplusplus
}
#endif


#else   /* ---- BARE-METAL: không cần HAL ---- */

#include <stdint.h>

/* ============================================================
 *  Cấu hình Q-point  (SH-2 Reference Manual, Table 2-1)
 * ============================================================ */
#define BNO_ACCEL_Q    8U    /* 1 LSB = 1/256   m/s²        */
#define BNO_GYRO_Q     9U    /* 1 LSB = 1/512   rad/s       */
#define BNO_QUAT_Q    14U    /* 1 LSB = 1/16384 (unitless)  */
#define BNO_ACC_Q     12U    /* accuracy field  → rad       */
#define BNO_MAX_PKT  128U    /* SHTP rx/tx buffer (bytes)   */

/* ============================================================
 *  Report IDs
 * ============================================================ */
#define BNO_RPT_ACCEL   0x01U
#define BNO_RPT_GYRO    0x02U
#define BNO_RPT_ROTVEC  0x05U
#define BNO_RPT_GAMEROT 0x08U
#define BNO_STATUS_INVALID 0xFFU

/* ============================================================
 *  Cấu trúc dữ liệu sensor  (điền bởi bno_update)
 * ============================================================ */
typedef struct {
    /* Giá trị vật lý đã chuyển đổi */
    float ax,  ay,  az;         /* Accelerometer   (m/s²)   */
    float gx,  gy,  gz;         /* Gyroscope       (rad/s)  */
    float qi,  qj,  qk,  qr;   /* Quaternion (chuẩn hoá)  */
    float qacc;                  /* Độ chính xác    (rad)    */
    float roll, pitch, yaw;      /* Euler           (deg)    */

    /* int16 gốc để kiểm tra Q-point */
    int16_t raw_ax, raw_ay, raw_az;
    int16_t raw_gx, raw_gy, raw_gz;
    int16_t raw_qi, raw_qj, raw_qk, raw_qr;

    /* Cờ dữ liệu mới — main.c xoá sau khi xử lý */
    uint8_t new_accel;
    uint8_t new_gyro;
    uint8_t new_rotvec;

    /* Payload thô SHTP packet cuối (để in [HEX]) */
    uint8_t  pkt[BNO_MAX_PKT];
    uint16_t pkt_len;
} BNO08x_Data;

typedef struct {
    float yaw_sign;        /* +1 hoac -1 de doi chieu yaw/gyro Z */
    float gyro_tau_s;      /* hang so loc thong thap gyro Z */
    float correction_gain; /* buoc keo yaw_est ve yaw_bno moi chu ky */
} BNO08x_EstimatorConfig;

typedef struct {
    float yaw_raw_rad;
    float yaw_bno_rad;
    float yaw_est_rad;
    float gyro_x_raw_rad_s;
    float gyro_y_raw_rad_s;
    float gyro_z_raw_rad_s;
    float gyro_x_filt_rad_s;
    float gyro_y_filt_rad_s;
    float gyro_z_filt_rad_s;
    float roll_rad;
    float pitch_rad;

    uint8_t has_yaw;
    uint8_t has_gyro;
    uint8_t yaw_zeroed;
} BNO08x_Estimator;

/* ============================================================
 *  API bare-metal
 * ============================================================ */

/** Khởi tạo I2C1 (PB8=SCL/D15, PB9=SDA/D14, 100 kHz). Gọi trước bno_init(). */
void    bno_hw_init(void);

/** Boot BNO08x: quét 0x4A/0x4B, nhận advertisement. Trả về địa chỉ 7-bit, hoặc 0x00. */
uint8_t bno_init(void);

/** Kích hoạt cảm biến với chu kỳ lấy mẫu.
 *  rptID : BNO_RPT_ACCEL / GYRO / ROTVEC / GAMEROT
 *  intervalUs : chu kỳ (µs) — vd 50000 = 20 Hz */
void    bno_enable(uint8_t rptID, uint32_t intervalUs);

/** Đọc 1 SHTP packet, cập nhật *d (bao gồm TẤT CẢ report trong batch).
 *  Lưu payload vào d->pkt / d->pkt_len để main.c in [HEX].
 *  Trả về 1 nếu nhận được packet cảm biến hợp lệ, 0 nếu không. */
int     bno_update(BNO08x_Data *d);

/** Cau hinh bo loc yaw/gyro nam trong thu vien. Goi 1 lan sau bno_init(). */
void    bno_estimator_config(const BNO08x_EstimatorConfig *cfg);

/** Dat yaw hien tai thanh 0 deg. Neu chua co quaternion, thu vien se zero khi co mau dau. */
void    bno_zero_yaw(BNO08x_Estimator *est);

/** Cap nhat yaw_raw/yaw_bno/yaw_est/gyroZ tu data moi nhat cua bno_update(). */
void    bno_estimator_update(const BNO08x_Data *d, float dt_s, BNO08x_Estimator *est);

/** Lay payload SHTP cuoi de gui [HEX] len serial. */
uint16_t bno_get_last_packet(const BNO08x_Data *d, const uint8_t **payload);

/* API ten cu de giu tuong thich voi main.c/cu cac vi du dang goi BNO_* */
uint8_t BNO_Init(void);
void    BNO_EnableReport(uint8_t rptID, uint32_t intervalUs);
int     BNO_Update(BNO08x_Data *d);

#endif /* USE_HAL_DRIVER */
#endif /* INC_BNO08X_H_ */

