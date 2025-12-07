import os
import cv2
import numpy as np
from ultralytics import SAM
from tqdm import tqdm
import glob

# --- CẤU HÌNH ---
DATASET_ROOT = '../datasets'
# Danh sách các tập cần xử lý. Code sẽ tự động bỏ qua nếu folder không tồn tại.
SPLITS_TO_PROCESS = ['train', 'val', 'test'] 

# Model SAM
SAM_MODEL = 'mobile_sam.pt' 

def create_yolo_polygon_label(mask, class_id, img_h, img_w):
    """Chuyển Binary Mask thành YOLO Polygon format"""
    # 1. Tìm Contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return None
    
    # Lấy contour lớn nhất
    cnt = max(contours, key=cv2.contourArea)
    
    # 2. Làm mượt contour
    epsilon = 0.005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    if len(approx) < 3: return None 
    
    # 3. Chuẩn hóa về [0, 1]
    points = approx.flatten().astype(float)
    normalized_points = []
    
    for i in range(0, len(points), 2):
        x = points[i] / img_w
        y = points[i+1] / img_h
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        normalized_points.extend([x, y])
        
    poly_str = f"{class_id} {' '.join(map('{:.6f}'.format, normalized_points))}"
    return poly_str

def process_split(model, split_name):
    """Hàm xử lý cho từng tập dữ liệu (train/val/test)"""
    
    # Định nghĩa đường dẫn động dựa trên split_name
    img_dir = os.path.join(DATASET_ROOT, 'images', split_name)
    label_det_dir = os.path.join(DATASET_ROOT, 'labels_det', split_name)
    output_dir = os.path.join(DATASET_ROOT, 'labels_seg_pseudo', split_name)
    
    # Kiểm tra tồn tại
    if not os.path.exists(label_det_dir):
        print(f"Bỏ qua tập '{split_name}': Không tìm thấy folder nhãn tại {label_det_dir}")
        return

    print(f"\nĐang xử lý tập: {split_name}...")
    os.makedirs(output_dir, exist_ok=True)
    
    label_files = glob.glob(os.path.join(label_det_dir, '*.txt'))
    print(f"Tìm thấy {len(label_files)} files.")
    
    if len(label_files) == 0: return

    for label_path in tqdm(label_files, desc=f"Processing {split_name}"):
        filename = os.path.basename(label_path)
        # Hỗ trợ cả jpg và png, ưu tiên jpg trước
        img_name_base = filename.rsplit('.', 1)[0]
        
        possible_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        img_path = None
        for ext in possible_exts:
            temp_path = os.path.join(img_dir, img_name_base + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        
        if img_path is None: continue # Không tìm thấy ảnh tương ứng
            
        # 1. Đọc ảnh & Box
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        boxes = []
        class_ids = []
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 5: continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = parts[1:]
                
                # YOLO -> XYXY
                x1 = (cx - bw/2) * w
                y1 = (cy - bh/2) * h
                x2 = (cx + bw/2) * w
                y2 = (cy + bh/2) * h
                
                boxes.append([x1, y1, x2, y2])
                class_ids.append(cls_id)
        
        if not boxes: continue
        
        # 2. Run SAM Inference
        results = model(img_path, bboxes=boxes, verbose=False) 
        new_lines = []
        
        # 3. Process Results
        for result in results:
            if result.masks is None: continue
            
            masks_np = result.masks.data.cpu().numpy().astype(np.uint8)
            
            for i, mask in enumerate(masks_np):
                if i >= len(class_ids): break
                
                if mask.shape[:2] != (h, w):
                     mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                poly_str = create_yolo_polygon_label(mask, class_ids[i], h, w)
                if poly_str:
                    new_lines.append(poly_str)
        
        # 4. Save Label
        if new_lines:
            save_path = os.path.join(output_dir, filename)
            with open(save_path, 'w') as f:
                f.write('\n'.join(new_lines))

def main():
    print(f"Khởi động quy trình tạo nhãn giả (Pseudo-Labeling)...")
    print(f"Model: {SAM_MODEL}")
    print(f"Root: {DATASET_ROOT}")
    
    try:
        model = SAM(SAM_MODEL)
    except Exception as e:
        print(f"Lỗi load model SAM: {e}")
        return

    # Duyệt qua từng split (train, val, test)
    for split in SPLITS_TO_PROCESS:
        process_split(model, split)

    print(f"\nHoàn tất! Nhãn mới đã được lưu tại: {DATASET_ROOT}/labels_seg_pseudo")
    print("Hãy cập nhật file 'manga_seg.yaml' để trỏ đến folder này cho cả 'train' và 'val'.")

if __name__ == "__main__":
    main()