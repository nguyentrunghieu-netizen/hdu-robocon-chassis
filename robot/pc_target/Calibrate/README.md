# Robot Calibration Mode - Huong dan tich hop

## File trong package nay

```
calibration/
├── calibration.py              # Module chinh (logic calibration thuan)
├── calibrate_cli.py            # Skeleton CLI runner
├── arduino_patch.txt           # Patch them vao mecanum_base.ino
├── serial_manager_patch.py     # Patch + huong dan tich hop Python
├── calibration_panel.html      # Web UI panel
├── robot_config.json           # File luu ket qua (auto sinh ra)
└── README.md                   # File nay
```

## Tong quan: chuyen gi xay ra khi calibrate

### 1. Motor Kv/Ks (~3 phut, can NHAC robot len)
- Cap tung muc PWM (60, 80, 100, ..., 220) cho tung banh.
- Doi 1.5s steady-state, do RPM 5 lan -> trung binh.
- Linear regression: `PWM = Ks + Kv * RPM`.
- Lam cho 4 banh × 2 chieu = 8 cap (Kv, Ks).
- Ghi vao `robot_config.json`.

### 2. Wheel Diameter (~30s, robot tren san)
- Robot di thang `vx = 0.15 m/s` trong ~9s (~1m).
- Encoder tich phan -> bao da di X met (theo wheel_diameter cu).
- LiDAR/SLAM bao thuc su di duoc Y met (tu pose dau-cuoi).
- Ratio = Y / X. Wheel diameter moi = wheel_diameter cu × ratio.
- Neu user nhap manual (do bang thuoc), uu tien manual.

### 3. Rotation Radius (~30s, robot tren san)
- Robot xoay tai cho `omega = 0.6 rad/s` trong ~11s (~1 vong).
- Encoder tich phan wz -> bao da xoay X rad (theo rotation_radius cu).
- IMU integrate yaw co xu ly wrap-around -> Y rad.
- Ratio = X / Y. Rotation radius moi = radius cu × ratio.
- Neu user nhap manual (do goc bang la ban), uu tien manual.

## Tich hop tung buoc

### Buoc A: Patch Arduino (mecanum_base.ino)

Mo file `arduino_patch.txt`, copy/paste cac doan code vao dung vi tri trong 
`mecanum_base.ino`. Cu the:

1. **Them bien global** (gan dau file, sau cac const khac):
   ```cpp
   bool calibrationMode = false;
   float calibrationPwm[4] = {0, 0, 0, 0};
   ```

2. **Them helper `parseFourFloats`** (gan ham `parseThreeFloats`).

3. **Them 2 case `P` va `Q`** trong `processCommand()`.

4. **Sua block cuoi `loop()`** de respect `calibrationMode`.

Sau do **upload code Arduino moi**.

### Buoc B: Patch Python (camera_lidar_web.py)

1. Copy file `calibration.py` vao cung thu muc voi `camera_lidar_web.py`.

2. Trong `class SerialManager`, them 3 method moi 
   (xem `serial_manager_patch.py`): `send_raw_pwm`, `reset_odometry`, 
   `query_raw_rpm`. Trong `__init__` them:
   ```python
   self._qr_response = None
   self._qr_event = threading.Event()
   ```

3. Trong `read_feedback()`, **NGAY SAU dong `line = ...`** them xu ly 'QR':
   ```python
   if line.startswith('QR '):
       import re
       m = re.match(r'QR\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+'
                    r'(-?\d+\.?\d*)\s+(-?\d+\.?\d*)', line.strip())
       if m:
           self._qr_response = [float(m.group(i)) for i in range(1, 5)]
           self._qr_event.set()
       continue
   ```

4. Trong `main()` (sau khi tao `app_state`, **TRUOC khi khoi tao serial/lidar**):
   ```python
   from calibration import Calibrator
   pre_cal = Calibrator(None, app_state, config_path='robot_config.json')
   if pre_cal.load_config():
       # Override constants
       global WHEEL_DIAMETER_M, ROTATION_RADIUS_M
       WHEEL_DIAMETER_M = pre_cal.config.wheel_diameter_m
       ROTATION_RADIUS_M = pre_cal.config.rotation_radius_m
       print(f'[CFG] wheel_diameter = {WHEEL_DIAMETER_M*1000:.2f}mm')
       print(f'[CFG] rotation_radius = {ROTATION_RADIUS_M*1000:.2f}mm')
   ```

5. Them CLI flag `--calibrate` (xem `calibrate_cli.py`).

6. Them Flask endpoints `/api/calibration/*` 
   (xem `INTEGRATION_NOTES` trong `serial_manager_patch.py`).

### Buoc C: Tich hop Web UI

Mo file HTML cua web UI hien tai (giao dien chinh), tim 1 vi tri thich hop 
(vd cuoi panel SLAM, hoac tab moi). Copy toan bo block trong 
`calibration_panel.html` vao do.

Refresh trang web -> se thay panel "🛠️ Robot Calibration".

## Cach dung

### Tu Web UI

1. Mo trang web cua robot.
2. Cuon den panel "Robot Calibration".
3. Bam **Motor Kv/Ks** -> nhac robot len -> bam **OK**. Doi ~3 phut.
4. Dat robot xuong san phang.
5. (Optional) Do khoang cach 1m bang thuoc, danh dau diem dau-cuoi tren san. 
   Sau khi robot dung, do khoang cach thuc te giua 2 diem -> nhap vao 
   "Distance (m)".
6. Bam **Wheel Diameter**.
7. Tuong tu cho **Rotation Radius** (do goc xoay bang la ban tren than robot, 
   nhap vao "Rotation (deg)").
8. Khi xong, **restart Python program** de load tham so moi.

### Tu CLI

```bash
# Full calibration
python3 camera_lidar_web.py --calibrate all

# Chi calibrate motor (robot phai duoc nhac len)
python3 camera_lidar_web.py --calibrate motors

# Calibrate wheel voi manual measurement
python3 camera_lidar_web.py --calibrate wheel --manual-distance 1.022

# Calibrate rotation voi 1 vong = 360 deg
python3 camera_lidar_web.py --calibrate rotation --manual-rotation-deg 360.5

# Xem config hien tai
python3 calibrate_cli.py --show-config

# Reset config ve default
python3 calibrate_cli.py --reset-config
```

## Lua chon thiet ke quan trong

### 1. Tai sao calibrate Motor TRUOC khi calibrate Wheel/Rotation?

Neu PID motor sai (Kv/Ks lech), khi ban yeu cau `vx = 0.15 m/s`, robot 
co the chi di duoc 0.10 m/s thuc te. Khi do, encoder bao van la 0.15 m/s 
(vi `targetRPM` la dau vao PID), nhung LiDAR do thuc te la 0.10 m/s. 
Ban se nham la wheel_diameter sai trong khi nguyen nhan that la PID kem.

Calibrate motor truoc -> dam bao banh xe quay dung toc do duoc yeu cau 
-> sau do moi do duoc wheel_diameter chinh xac.

### 2. Tai sao trong Wheel Calibration KHONG dung `(x1-x0, y1-y0)` true?

Robot di thang co the bi lech sang trai/phai (do banh khong deu, ma sat khong 
deu). `math.hypot(x1-x0, y1-y0)` = khoang cach Euclidean = van dung. Khoang 
cach nay la **path length** xap xi neu robot di tuong doi thang.

Neu muon chinh xac hon, co the dung `path length` = tich phan |v| dt tu LiDAR 
pose theo thoi gian. Hien tai code dung Euclidean cho don gian.

### 3. Tai sao IMU yaw integration phai xu ly wrap-around?

BNO085 tra ve yaw trong khoang [-pi, pi]. Khi robot xoay 1 vong, yaw nhay tu 
+pi xuong -pi. Neu khong xu ly, encoder bao 6.28 rad ma IMU "bao" 0 rad -> 
ratio = inf -> sai.

Code da xu ly bang cach tinh `delta = cur_yaw - prev_yaw`, wrap delta vao 
[-pi, pi], sau do tich luy `total_yaw_change += delta`.

### 4. Tai sao tach `wheel_0/wheel_1/wheel_2/wheel_3 × fwd/rev` rieng?

DC motor co the co Kv/Ks khac nhau giua chieu xuoi va nguoc (do brush, 
backlash). Neu trung binh hoa thi PID feedforward se sai trong 1 chieu. 
Luu rieng -> Arduino lookup table -> chinh xac hon.

Hien tai code Arduino chua co lookup table 8 cap. **Improvement de xuat**:
- Trong Arduino, doc 8 cap tu file (qua command moi `K wheel_idx dir kv ks`).
- Hoac don gian: lay trung binh 8 cap, chia se cho 4 banh × 2 chieu.
- Hien tai Calibrator chi luu vao file. Phia Python co the tu su dung 
  cac gia tri nay (vd send command `K` moi khi connect lai Arduino).

## Limit / TODO trong tuong lai

1. **Auto-tune PID**: hien tai chi calibrate feedforward (Kv/Ks). Co the them 
   step-response test de tu chinh Kp/Ki/Kd.

2. **Mecanum slip detection**: trong rotation calibration, neu encoder 
   bao 1.5 vong ma IMU chi do 0.8 vong -> banh truot nhieu. Co the canh 
   bao "san qua tron" hoac "tai trong qua nang".

3. **Per-wheel kinematic calibration**: hien gia su 4 banh hoan toan giong 
   nhau. Neu 1 banh hoi to/be hon -> robot drift. Co the calibrate tung banh 
   bang cach quay lan luot tung banh va so wheel_speed voi imu_wz.

4. **Asymmetry detection**: trong Wheel Calibration, ngoai khoang cach, 
   con co the do `dy / dx` (lech sang ngang). Neu lech > 5%, canh bao 
   "robot bi keo lech" -> goi y kiem tra banh xe hoac Kv/Ks bat doi xung.

5. **Save command vao Arduino**: hien tai sau calibrate, ban phai 
   re-flash Arduino voi gia tri Kv/Ks moi (hoac Python phai gui xuong 
   moi lan boot). Tot nhat la them EEPROM save tren Arduino voi command 
   `EEPROM_SAVE` va `EEPROM_LOAD`.
