import os
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO, SAM
from tqdm import tqdm
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# --- CẤU HÌNH HỆ THỐNG ---
DATASET_DIR = '../datasets'
SPLIT = 'val'
IMG_DIR = os.path.join(DATASET_DIR, 'images', SPLIT)
LABEL_DIR = os.path.join(DATASET_DIR, 'labels_seg', SPLIT) # Dùng nhãn Segmentation gốc

# Định nghĩa các Pipeline cần test
# Bạn có thể thêm bao nhiêu model tùy thích vào đây
MODELS_TO_TEST = [
    {
        'name': 'P1_YOLO_Nano',
        'type': 'yolo_seg',
        'path': '../runs/pipeline1_yolo/train_seg_augmented/weights/best.pt'
    },
    # {
    #     'name': 'P1_YOLO_Small',
    #     'type': 'yolo_seg', 
    #     'path': 'yolov8s-seg.pt' 
    # },
    {
        'name': 'P2_YOLO_Det_Nano + SAM',
        'type': 'pipeline_2',
        'det_path': '../runs/pipeline2_det/train_det/weights/best.pt',
        'sam_path': 'sam_b.pt'
    }
]

def load_gt_polygons(img_name, h, w):
    """Đọc Ground Truth dưới dạng danh sách các Polygon riêng biệt"""
    txt_name = img_name.rsplit('.', 1)[0] + '.txt'
    txt_path = os.path.join(LABEL_DIR, txt_name)
    polygons = []
    
    if not os.path.exists(txt_path): return []
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = list(map(float, line.strip().split()))
            # Format YOLO Seg: class x1 y1 x2 y2 ...
            coords = np.array(parts[1:]).reshape(-1, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            if len(coords) >= 3:
                polygons.append(Polygon(coords))
    return polygons

def mask_to_polygon(mask):
    """Chuyển Binary Mask thành Polygon (Shapely)"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50 and len(cnt) >= 3:
            pts = cnt.squeeze().tolist()
            if len(pts) >= 3:
                polys.append(Polygon(pts))
    # Trả về polygon lớn nhất (giả sử 1 mask = 1 bóng thoại)
    if not polys: return None
    return max(polys, key=lambda x: x.area)

def calculate_instance_iou(gt_polys, pred_masks, h, w):
    """
    Tính IoU theo cơ chế Matching (Instance Segmentation Evaluation)
    Thay vì gộp tất cả mask lại, ta tìm cặp (GT, Pred) khớp nhất.
    """
    if not gt_polys: return 0.0, 0.0
    
    # Chuyển đổi list pred_masks thành list polygons
    pred_polys = []
    for m in pred_masks:
        # Resize về kích thước gốc nếu cần
        if m.shape[:2] != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Quan trọng: Threshold chuẩn để không bị lỗi float->int
        # Mask của YOLO/SAM thường là float 0..1 hoặc logit. 
        # Cần > 0.5 để ra binary.
        bin_mask = (m > 0.5).astype(np.uint8)
        
        poly = mask_to_polygon(bin_mask)
        if poly and poly.is_valid:
            pred_polys.append(poly)
            
    if not pred_polys: return 0.0, 0.0

    # Tính IoU Matrix
    iou_matrix = np.zeros((len(gt_polys), len(pred_polys)))
    for i, gt in enumerate(gt_polys):
        for j, pred in enumerate(pred_polys):
            if not gt.is_valid or not pred.is_valid: continue
            intersect = gt.intersection(pred).area
            union = gt.union(pred).area
            if union > 0:
                iou_matrix[i, j] = intersect / union
    
    # Matching đơn giản: Lấy max IoU cho mỗi GT
    # (Cách này chưa phải tối ưu nhất như Hungarian Matching nhưng đủ tốt để đánh giá)
    matched_ious = []
    for i in range(len(gt_polys)):
        if iou_matrix.shape[1] > 0:
            max_iou = np.max(iou_matrix[i])
            matched_ious.append(max_iou)
        else:
            matched_ious.append(0)
            
    mean_iou = np.mean(matched_ious)
    
    # Precision tại IoU 0.5 (Metric phổ biến)
    matches_05 = sum(1 for iou in matched_ious if iou >= 0.5)
    precision_05 = matches_05 / len(gt_polys)
    
    return mean_iou, precision_05

def run_benchmark():
    print("🚀 BẮT ĐẦU BENCHMARK HỆ THỐNG ĐA MÔ HÌNH...")
    
    # 1. Load tất cả model
    loaded_models = []
    for cfg in MODELS_TO_TEST:
        print(f"Loading {cfg['name']}...")
        try:
            if cfg['type'] == 'yolo_seg':
                model = YOLO(cfg['path'])
                loaded_models.append({'cfg': cfg, 'model': model})
            elif cfg['type'] == 'pipeline_2':
                det_model = YOLO(cfg['det_path'])
                sam_model = SAM(cfg['sam_path'])
                loaded_models.append({'cfg': cfg, 'det': det_model, 'sam': sam_model})
        except Exception as e:
            print(f"Lỗi load model {cfg['name']}: {e}")

    img_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    # img_files = img_files[:20] # Uncomment để test nhanh trên 20 ảnh đầu
    
    results = []

    print(f"Đang xử lý {len(img_files)} ảnh...")
    for img_file in tqdm(img_files):
        img_path = os.path.join(IMG_DIR, img_file)
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        # Load GT
        gt_polys = load_gt_polygons(img_file, h, w)
        if not gt_polys: continue

        for item in loaded_models:
            cfg = item['cfg']
            pred_masks = []
            
            # --- RUN INFERENCE ---
            if cfg['type'] == 'yolo_seg':
                res = item['model'](img_path, verbose=False, retina_masks=True)[0]
                if res.masks is not None:
                    pred_masks = res.masks.data.cpu().numpy() # Float array
                    
            elif cfg['type'] == 'pipeline_2':
                # Step 1: Detect
                res_det = item['det'](img_path, verbose=False)[0]
                boxes = res_det.boxes.xyxy.cpu().numpy()
                
                # Step 2: SAM
                if len(boxes) > 0:
                    # Mở rộng box nhẹ (padding) để SAM bắt tốt hơn
                    padding = 5
                    padded_boxes = []
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        padded_boxes.append([
                            max(0, x1-padding), max(0, y1-padding),
                            min(w, x2+padding), min(h, y2+padding)
                        ])
                    
                    res_sam = item['sam'](img_path, bboxes=padded_boxes, verbose=False)
                    # SAM trả về list results
                    for r in res_sam:
                        if r.masks is not None:
                            # r.masks.data shape thường là (N, 1, H, W) hoặc (N, H, W)
                            masks_np = r.masks.data.cpu().numpy()
                            for m in masks_np:
                                pred_masks.append(m.squeeze())

            # --- CALCULATE METRICS ---
            mean_iou, p_05 = calculate_instance_iou(gt_polys, pred_masks, h, w)
            
            results.append({
                'Image': img_file,
                'Model': cfg['name'],
                'mIoU': mean_iou,
                'Precision@0.5': p_05
            })

    # --- TỔNG HỢP KẾT QUẢ ---
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("KẾT QUẢ CUỐI CÙNG")
    print("="*50)
    
    summary = df.groupby('Model')[['mIoU', 'Precision@0.5']].mean()
    print(summary)
    
    # Lưu file CSV
    df.to_csv('benchmark_details.csv', index=False)
    summary.to_csv('benchmark_summary.csv')
    
    # Vẽ biểu đồ so sánh
    try:
        summary.plot(kind='bar', figsize=(10, 6))
        plt.title('So sánh Hiệu năng các Pipeline')
        plt.ylabel('Score')
        plt.ylim(0, 1.0)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig('benchmark_chart.png')
        print("Đã lưu biểu đồ so sánh tại benchmark_chart.png")
    except: pass

if __name__ == "__main__":
    run_benchmark()