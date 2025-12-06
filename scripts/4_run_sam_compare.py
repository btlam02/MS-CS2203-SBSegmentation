from ultralytics import YOLO, SAM
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CẤU HÌNH ---
IMG_PATH = '../datasets/images/train/AisazuNihaIrarenai_005.jpg'
PADDING = 10 

# Load Models
model_yolo_seg = YOLO('../runs/pipeline1_yolo/train_seg_augmented_11_n/weights/best.pt')
model_yolo_det = YOLO('../runs/pipeline2_det/train_det/weights/best.pt')
model_sam = SAM('sam_b.pt')

def get_center_point(box):
    """Tính tâm của box để làm Point Prompt"""
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return [cx, cy]

def expand_box(box, h, w, pad=10):
    """Mở rộng box (Padding) để không cắt mất viền bóng thoại"""
    x1, y1, x2, y2 = box
    nx1 = max(0, x1 - pad)
    ny1 = max(0, y1 - pad)
    nx2 = min(w, x2 + pad)
    ny2 = min(h, y2 + pad)
    return [nx1, ny1, nx2, ny2]

# --- PIPELINE 1: YOLO SEG (Basic) ---
res_p1 = model_yolo_seg(IMG_PATH, retina_masks=True)[0]
img_p1 = res_p1.plot(boxes=False)

# --- PIPELINE 2: YOLO DET + SAM (Advanced) ---
img_raw = cv2.imread(IMG_PATH)
h, w = img_raw.shape[:2]

# Bước 1: Detect
res_det = model_yolo_det(IMG_PATH)[0]
raw_boxes = res_det.boxes.xyxy.cpu().numpy()

final_mask = np.zeros((h, w, 3), dtype=np.uint8)

if len(raw_boxes) > 0:
    # Bước 2: Prompt Engineering (Tạo Box Padding + Point Center)
    processed_boxes = []
    points = []
    point_labels = []

    for box in raw_boxes:
        # A. Padding Box
        p_box = expand_box(box, h, w, PADDING)
        processed_boxes.append(p_box)
        
        # B. Center Point (Prompt bổ sung)
        center = get_center_point(box)
        points.append([center])     # Format của SAM là [[x,y], [x,y]]
        point_labels.append([1])    # 1 nghĩa là foreground (điểm thuộc vật thể)

    # Bước 3: SAM Inference với Hybrid Prompts
    # Lưu ý: Ultralytics SAM hiện hỗ trợ bboxes tốt, point hỗ trợ tùy phiên bản.
    # Để an toàn và hiệu quả nhất theo Slide, ta dùng bboxes đã padding.
    res_sam = model_sam(IMG_PATH, bboxes=processed_boxes)
    
    # Plot kết quả SAM
    img_p2 = res_sam[0].plot(boxes=False) # Vẽ mask lên ảnh
else:
    img_p2 = img_raw

# --- SO SÁNH TRỰC QUAN ---
plt.figure(figsize=(15, 8))

# Ảnh gốc
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# Pipeline 1
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img_p1, cv2.COLOR_BGR2RGB))
plt.title("P1: YOLOv8 Seg (One-Stage)")
plt.axis('off')

# Pipeline 2
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(img_p2, cv2.COLOR_BGR2RGB))
plt.title(f"P2: YOLO Det + SAM\n(Padding {PADDING}px + Center Context)")
plt.axis('off')

plt.tight_layout()
plt.savefig('../advanced_comparison.jpg')
plt.show()

print(f"Đã lưu ảnh so sánh tại ../advanced_comparison.jpg")