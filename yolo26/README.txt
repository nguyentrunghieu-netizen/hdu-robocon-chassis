# YOLO Dataset

Cau truc:
- images/train/  - anh de train (8 anh)
- images/val/    - anh de validate (3 anh)
- labels/train/  - file label tuong ung (.txt)
- labels/val/    - file label tuong ung (.txt)
- data.yaml      - file config cho training

## Cach dung:

1. Giai nen file zip nay
2. Mo data.yaml va sua dong "path:" thanh duong dan tuyet doi
3. Cai dat ultralytics:
   pip install ultralytics

4. Train voi code Python:

   from ultralytics import YOLO
   model = YOLO("yolo26n.pt")
   model.train(data="data.yaml", epochs=100, imgsz=640)

## Classes:
0: pa

## Format label YOLO (moi dong = 1 box):
class_id x_center y_center width height

Tat ca toa do duoc chuan hoa ve [0, 1] (chia cho kich thuoc anh).
