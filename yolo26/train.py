# from ultralytics import YOLO

# # Load model có sẵn (n = nano = nhỏ nhất, train nhanh nhất)
# model = YOLO("yolo26n.pt")

# # Bắt đầu train
# model.train(
#     data="data.yaml",   # file config bước 3
#     epochs=100,         # số vòng train
#     imgsz=640,          # kích thước ảnh
#     batch=16,           # số ảnh/batch (giảm xuống nếu thiếu RAM/GPU)
#     device='cpu'            # 0 = dùng GPU, "cpu" = dùng CPU
# )
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,           # số ảnh/batch (giảm xuống nếu thiếu RAM/GPU)
    device='cpu',       # 0 = dùng GPU, "cpu" = dùng CPU
    # ===== Augmentation tăng cường =====
    degrees=15,        # xoay ngẫu nhiên ±15°
    translate=0.2,     # dịch chuyển nhiều hơn
    scale=0.5,         # zoom in/out
    fliplr=0.5,        # lật ngang 50%
    flipud=0.0,        # KHÔNG lật dọc (vd: ảnh xe thì không lật ngược)
    mosaic=1.0,        # luôn ghép 4 ảnh
    mixup=0.1,         # 10% xác suất trộn 2 ảnh
    hsv_h=0.02,
    hsv_s=0.7,
    hsv_v=0.4,
)