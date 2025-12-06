from ultralytics import YOLO
import os

# Tạo file config
yaml_content = f"""
path: {os.path.abspath('../datasets')}
train: images/train
val: images/val
names:
  0: balloon
"""
with open("manga_det.yaml", "w") as f:
    f.write(yaml_content)

# TRƯỚC KHI CHẠY: Đảm bảo folder 'datasets/labels' đang chứa dữ liệu của 'labels_det'
# (Bạn cần đổi tên folder labels_det -> labels)

model = YOLO('yolov8n.pt') # Detection model

results = model.train(
    data='manga_det.yaml',
    epochs=50,
    imgsz=640,
    project='../runs/pipeline2_det',
    name='train_det',
    device=0
)