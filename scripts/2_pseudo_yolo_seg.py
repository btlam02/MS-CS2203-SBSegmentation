from ultralytics import YOLO
import os
import shutil

# --- CẤU HÌNH ---
DATASET_DIR = '../datasets'
PSEUDO_LABEL_DIR = os.path.join(DATASET_DIR, 'labels_seg')
TARGET_LABEL_DIR = os.path.join(DATASET_DIR, 'labels')

def setup_labels_for_yolo():
    """
    YOLOv8 bắt buộc folder nhãn phải tên là 'labels' và nằm cùng cấp với 'images'.
    Hàm này sẽ tự động thay thế folder 'labels' hiện tại bằng nhãn giả từ SAM.
    """
    # 1. Kiểm tra xem đã chạy step 0 tạo nhãn giả chưa
    if not os.path.exists(PSEUDO_LABEL_DIR):
        print(f"LỖI: Không tìm thấy folder '{PSEUDO_LABEL_DIR}'. Hãy chạy file '0_auto_label_with_sam.py' trước!")
        return False

    # 2. Backup folder 'labels' cũ (thường là nhãn detection) nếu có
    if os.path.exists(TARGET_LABEL_DIR):
        # Kiểm tra xem đây có phải là nhãn Det cũ không hay đã là Seg rồi
        # Backup an toàn ra folder riêng
        backup_dir = os.path.join(DATASET_DIR, 'labels_backup_before_seg')
        if not os.path.exists(backup_dir):
            print(f"Đang backup folder 'labels' hiện tại sang '{backup_dir}'...")
            os.rename(TARGET_LABEL_DIR, backup_dir)
        else:
            # Nếu đã có backup rồi thì xóa folder labels hiện tại đi để chép mới
            shutil.rmtree(TARGET_LABEL_DIR)

    # 3. Copy nhãn giả (Segmentation) vào vị trí chuẩn 'labels'
    print(f"Đang copy nhãn Segmentation vào vị trí chuẩn: {TARGET_LABEL_DIR}...")
    shutil.copytree(PSEUDO_LABEL_DIR, TARGET_LABEL_DIR)
    return True

# --- MAIN ---
if setup_labels_for_yolo():
    # Tạo file config (LƯU Ý: train/val trỏ vào IMAGES, không phải labels)
    yaml_content = f"""
    path: {os.path.abspath(DATASET_DIR)}
    train: images/train
    val: images/val
    names:
      0: balloon
    """
    with open("manga_seg.yaml", "w") as f:
        f.write(yaml_content)

    print("Bắt đầu huấn luyện YOLO Segmentation...")
    
    # Load model pre-trained
    model = YOLO('yolov8n-seg.pt') 
    
    results = model.train(
        data='manga_seg.yaml',
        epochs=50,
        imgsz=640,
        project='../runs/pipeline1_yolo',
        name='train_seg_pseudo',
        device=0,
      
        box=7.5,        # Tăng trọng số loss cho Box
        cls=2.5,       # Tăng trọng số loss cho Mask
        close_mosaic=10 # Tắt Mosaic augmentation ở 10 epoch cuối để ổn định mask
    )
else:
    print("Dừng chương trình do thiếu dữ liệu nhãn.")