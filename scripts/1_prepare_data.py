import pandas as pd
import numpy as np
import os
import cv2
import shutil
import ast
from pycocotools import mask as mask_utils
from tqdm import tqdm
from sklearn.model_selection import train_test_split

CSV_PATH = '../raw_data/annotations.csv'
RAW_IMG_ROOT = '../raw_data/images' 
OUTPUT_DIR = '../datasets'

def setup_dirs():
    for split in ['train', 'val']:
        os.makedirs(f'{OUTPUT_DIR}/images/{split}', exist_ok=True)
        os.makedirs(f'{OUTPUT_DIR}/labels_seg/{split}', exist_ok=True) # Pipeline 1
        os.makedirs(f'{OUTPUT_DIR}/labels_det/{split}', exist_ok=True) # Pipeline 2

def decode_rle_and_get_polygon(rle_counts, rle_size, img_h, img_w):
    try:
        # Decode RLE thành Mask
        rle_obj = {'counts': rle_counts.encode('utf-8'), 'size': rle_size}
        mask = mask_utils.decode(rle_obj)
        
        # Tìm Contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 50: continue 
            # Simplify contour để giảm điểm thừa
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3: continue
            
            # Normalize Polygon [0, 1]
            points = approx.flatten()
            norm_points = []
            for i in range(0, len(points), 2):
                x = min(max(points[i] / img_w, 0), 1)
                y = min(max(points[i+1] / img_h, 0), 1)
                norm_points.extend([x, y])
            polygons.append(norm_points)
        return polygons
    except:
        return []

def main():
    setup_dirs()
    print("Đang đọc CSV...")
    df = pd.read_csv(CSV_PATH)
    
    unique_files = df['file_name'].unique()
    train_files, val_files = train_test_split(unique_files, test_size=0.15, random_state=42)
    
    print("Bắt đầu xử lý dữ liệu...")
    for file_name, group in tqdm(df.groupby('file_name')):
        # 1. Xác định file ảnh gốc và đích
        src_path = os.path.join(RAW_IMG_ROOT, file_name)
        if not os.path.exists(src_path): continue
        
        split = 'train' if file_name in train_files else 'val'
        flat_name = file_name.replace('/', '_') # Boku/002.jpg -> Boku_002.jpg
        dst_img_path = os.path.join(OUTPUT_DIR, 'images', split, flat_name)
        
        # Copy ảnh (chỉ copy 1 lần)
        if not os.path.exists(dst_img_path):
            shutil.copy2(src_path, dst_img_path)
        
        # Đọc ảnh để lấy kích thước thật
        img = cv2.imread(src_path)
        if img is None: continue
        h_img, w_img = img.shape[:2]
        
        seg_lines = [] # Cho Pipeline 1
        det_lines = [] # Cho Pipeline 2
        
        for _, row in group.iterrows():
            # A. Xử lý Segmentation (Polygon)
            try:
                rle_counts = row['segmentation_counts']
                # Xử lý segmentation_size (có thể là string "[h, w]")
                rle_size_raw = row['segmentation_size']
                if isinstance(rle_size_raw, str):
                    rle_size = ast.literal_eval(rle_size_raw)
                else:
                    rle_size = rle_size_raw # Nếu pandas đã tự convert
                    
                polys = decode_rle_and_get_polygon(rle_counts, rle_size, h_img, w_img)
                for poly in polys:
                    # Format: 0 x1 y1 x2 y2 ...
                    seg_lines.append(f"0 {' '.join(map('{:.6f}'.format, poly))}")
            except Exception as e: pass

            # B. Xử lý Detection (BBox)
            try:
                # Bbox trong CSV thường là [x_min, y_min, w, h]
                bbox_raw = row['bbox']
                if isinstance(bbox_raw, str): bbox = ast.literal_eval(bbox_raw)
                else: bbox = bbox_raw
                
                x, y, w, h = bbox
                # Convert sang YOLO format: x_center, y_center, w_norm, h_norm
                xc = (x + w/2) / w_img
                yc = (y + h/2) / h_img
                wn = w / w_img
                hn = h / h_img
                
                det_lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            except Exception as e: pass
            
        # Ghi file nhãn Seg
        if seg_lines:
            with open(f'{OUTPUT_DIR}/labels_seg/{split}/{flat_name.replace(".jpg", ".txt")}', 'w') as f:
                f.write('\n'.join(seg_lines))
                
        # Ghi file nhãn Det
        if det_lines:
            with open(f'{OUTPUT_DIR}/labels_det/{split}/{flat_name.replace(".jpg", ".txt")}', 'w') as f:
                f.write('\n'.join(det_lines))

if __name__ == "__main__":
    main()