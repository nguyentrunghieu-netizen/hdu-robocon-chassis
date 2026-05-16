"""
detector.py
===========
FaceDetector: phat hien khuon mat dung YOLO.
"""
from config import YOLO_MODEL, YOLO_CONF, YOLO_IMGSZ, YOLO_CLASSES, MIN_BBOX_H
from ultralytics import YOLO


class FaceDetector:
    def __init__(self, model_path=YOLO_MODEL, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, classes=YOLO_CLASSES):
        print(f"[YOLO] Model: {model_path} | conf={conf} | imgsz={imgsz}")
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.classes = classes

    def detect(self, frame):
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=self.classes,
            verbose=False,
        )

        if not results:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        best_idx = int(areas.argmax())

        x1, y1, x2, y2 = xyxy[best_idx]
        box_w = x2 - x1
        box_h = y2 - y1
        if box_h < MIN_BBOX_H:
            return None

        center_x = x1 + box_w / 2.0
        center_y = y1 + box_h / 2.0
        return (
            float(center_x),
            float(center_y),
            float(box_h),
            float(box_w),
            (float(x1), float(y1), float(x2), float(y2)),
        )
