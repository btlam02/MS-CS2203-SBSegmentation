from ultralytics import YOLO
import os

# Tạo file config động
yaml_content = f"""
path: {os.path.abspath('../datasets')}
train: images/train
val: images/val
names:
  0: balloon
"""
with open("manga_seg.yaml", "w") as f:
    f.write(yaml_content)


model = YOLO('yolov8n-seg.pt') # n: nano, s: small, m: medium


# ... existing code ...
results = model.train(
    data='manga_seg.yaml',
    epochs=50,
    imgsz=640,
    project='../runs/pipeline1_yolo',
    name='train_seg_augmented_11_n',
    device=0,
    box=7.5,
    cls=0.5,
    
    # --- TÙY CHỈNH AUGMENTATION ---
    mosaic=1.0,      # (0.0 - 1.0) Bật Mosaic 100% (Rất tốt cho bóng thoại nhỏ)
    mixup=0.1,       # (0.0 - 1.0) Trộn ảnh nhẹ (giúp bền vững với nhiễu)
    degrees=10.0,    # (+/- deg) Xoay ảnh nhẹ +/- 10 độ (Manga có thể bị nghiêng khi scan)
    translate=0.1,   # (+/- fraction) Dịch chuyển ảnh
    scale=0.5,       # (+/- gain) Co giãn ảnh (quan trọng vì bóng thoại to nhỏ khác nhau)
    shear=0.0,       # Làm méo ảnh (ít dùng cho Manga vì khung tranh thường vuông vức)
    perspective=0.0, # Phối cảnh (ít dùng)
    flipud=0.0,      # Lật dọc (Không nên dùng vì chữ sẽ bị ngược/lộn xộn)
    fliplr=0.5,      # Lật ngang (Có thể dùng, nhưng cẩn thận với thứ tự đọc phải-trái của Manga)
    hsv_h=0.015,     # Chỉnh màu Hue
    hsv_s=0.0,       # Chỉnh độ bão hòa (Manga đen trắng -> set về 0)
    hsv_v=0.4,       # Chỉnh độ sáng
    
    # --- CHIẾN LƯỢC HUẤN LUYỆN ---
    close_mosaic=10  # Tắt Mosaic ở 10 epoch cuối
)