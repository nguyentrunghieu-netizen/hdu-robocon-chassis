from ultralytics import YOLO

# Load model đã train
model = YOLO("runs/detect/train/weights/best.pt")

# Detect với ngưỡng siêu thấp
results = model("KHAN3330.jpg", conf=0.1, save=True)

# In ra số lượng box phát hiện được
for r in results:
    print(f"Số box phát hiện: {len(r.boxes)}")
    if len(r.boxes) > 0:
        print(f"Confidence cao nhất: {r.boxes.conf.max().item():.3f}")
        print(f"Class các box: {r.boxes.cls.tolist()}")