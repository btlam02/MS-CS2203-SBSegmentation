import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO, SAM
from PIL import Image
import tempfile
import os
import time

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Manga Bubble Segmentation",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TÙY CHỈNH ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #FF4B4B; text-align: center;}
    .sub-header {font-size: 1.5rem; font-weight: 600; color: #333; margin-bottom: 20px;}
    .stButton>button {width: 100%; border-radius: 8px; height: 50px; font-weight: bold;}
    .reportview-container .main .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# --- HÀM TẢI MODEL (CACHED) ---
@st.cache_resource
def load_model(model_path, task='seg'):
    """Load model và cache để không phải load lại mỗi lần reload trang"""
    try:
        if task == 'sam':
            return SAM(model_path)
        else:
            return YOLO(model_path)
    except Exception as e:
        st.error(f"Lỗi không tìm thấy model tại {model_path}. Vui lòng kiểm tra thư mục 'models'.")
        return None

# --- XỬ LÝ HÌNH ẢNH ---
def process_pipeline_1(image, model, conf, iou):
    """Xử lý Pipeline 1: YOLOv8 Segmentation"""
    start_time = time.time()
    results = model(image, conf=conf, iou=iou, retina_masks=True)
    end_time = time.time()
    
    # Vẽ kết quả lên ảnh
    res_plotted = results[0].plot()
    return res_plotted, (end_time - start_time) * 1000

def process_pipeline_2(image, det_model, sam_model, conf, padding=10):
    """Xử lý Pipeline 2: YOLO Detection + SAM"""
    start_time = time.time()
    h, w = image.shape[:2]
    
    # Bước 1: Detect Box
    det_results = det_model(image, conf=conf)
    boxes = det_results[0].boxes.xyxy.cpu().numpy()
    
    if len(boxes) == 0:
        return image, (time.time() - start_time) * 1000, "Không tìm thấy bóng thoại nào."

    # Bước 2: Prompt Engineering (Padding)
    processed_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        # Mở rộng box
        nx1 = max(0, x1 - padding)
        ny1 = max(0, y1 - padding)
        nx2 = min(w, x2 + padding)
        ny2 = min(h, y2 + padding)
        processed_boxes.append([nx1, ny1, nx2, ny2])

    # Bước 3: SAM Inference
    # Lưu ý: Ultralytics SAM hỗ trợ nhận list bboxes
    sam_results = sam_model(image, bboxes=processed_boxes)
    
    # Vẽ kết quả (Lấy kết quả đầu tiên vì ta xử lý 1 ảnh)
    # SAM trả về list results, ta plot đè lên ảnh gốc
    res_plotted = sam_results[0].plot(boxes=False) # Chỉ vẽ mask, không vẽ box của SAM
    
    # Để đẹp hơn, ta vẽ thêm box detection từ YOLO lên
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(res_plotted, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    end_time = time.time()
    return res_plotted, (end_time - start_time) * 1000, f"Phát hiện {len(boxes)} bóng thoại."

# --- GIAO DIỆN CHÍNH ---
def main():
    st.markdown('<div class="main-header">Manga Speech Bubble Segmentation Demo</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Hệ thống phát hiện và phân đoạn bóng thoại truyện tranh sử dụng YOLOv8 & SAM</p>', unsafe_allow_html=True)
    st.markdown("---")

    # --- SIDEBAR: Cấu hình ---
    with st.sidebar:
        st.header("⚙️ Cấu Hình Hệ Thống")
        
        # Chọn Pipeline
        pipeline_option = st.radio(
            "Chọn Pipeline xử lý:",
            ("Pipeline 1: YOLOv8-Seg ", "Pipeline 2: YOLO Det & SAM")
        )
        
        st.markdown("---")
        
        selected_model = None
        det_model = None
        sam_model = None
        
        if "Pipeline 1" in pipeline_option:
            st.subheader("Tùy chọn Pipeline 1")
            model_size = st.selectbox("Chọn kích thước model:", ["Nano", "Small", "Medium"])
            
            # Đường dẫn model (Thay bằng đường dẫn thực tế của bạn)
            model_path_map = {
                "Nano": "models/seg-yolov8n.pt", 
                "Small": "models/seg-yolov8s.pt",
                "Medium": "models/seg-yolov8m.pt"
            }
            selected_model = load_model(model_path_map[model_size])
            
        else:
            st.subheader("🔧 Tùy chọn Pipeline 2")
            st.markdown("**1. Detection Model (YOLOv8)**")
            # Pipeline 2 thường cố định detection model là Nano hoặc Small cho nhanh
            det_path = "models/det-yolov8n.pt" # Hoặc đường dẫn model det custom của bạn
            det_model = load_model(det_path, task='det')
            
            st.markdown("**2. Segmentation Model (SAM)**")
            sam_type = st.selectbox("Chọn phiên bản SAM:", ["MobileSAM (Nhanh)", "SAM Base (Chính xác)"])
            
            sam_path_map = {
                "MobileSAM (Nhanh)": "models/mobile_sam.pt",
                "SAM Base (Chính xác)": "models/sam_b.pt" 
            }
            sam_model = load_model(sam_path_map[sam_type], task='sam')
            
            padding = st.slider("Box Padding (px):", 0, 20, 10, help="Mở rộng vùng box để SAM bắt biên tốt hơn")

        # Tham số chung
        st.markdown("---")
        st.subheader("🎚️ Tham số suy luận")
        conf_thres = st.slider("Confidence Threshold:", 0.0, 1.0, 0.25)
        iou_thres = st.slider("IoU Threshold:", 0.0, 1.0, 0.45)

    # --- MAIN CONTENT: Upload & Display ---
    col1, col2 = st.columns([1, 1])
    
    uploaded_file = st.file_uploader("Tải lên trang truyện tranh (JPG, PNG)", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        # Đọc ảnh
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_opencv = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
        
        with col1:
            st.subheader("Ảnh Gốc")
            st.image(img_rgb, use_column_width=True)

        # Nút Chạy
        if st.sidebar.button("HẠY PHÂN ĐOẠN"):
            with col2:
                st.subheader("Kết Quả Phân Đoạn")
                result_placeholder = st.empty()
                status_text = st.info("Đang xử lý... Vui lòng đợi.")
                
                try:
                    res_img = None
                    exec_time = 0
                    msg = ""
                    
                    if "Pipeline 1" in pipeline_option:
                        if selected_model:
                            res_img, exec_time = process_pipeline_1(img_opencv, selected_model, conf_thres, iou_thres)
                    else:
                        if det_model and sam_model:
                            res_img, exec_time, msg = process_pipeline_2(img_opencv, det_model, sam_model, conf_thres, padding)
                            if msg: st.success(msg)

                    if res_img is not None:
                        # Convert lại sang RGB để hiển thị đúng màu
                        res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                        result_placeholder.image(res_img_rgb, use_column_width=True)
                        status_text.success(f"Hoàn thành trong {exec_time:.2f} ms")
                    else:
                        status_text.error("Lỗi: Không thể xử lý ảnh.")
                        
                except Exception as e:
                    status_text.error(f"Đã xảy ra lỗi: {str(e)}")
                    st.exception(e)
    else:
        st.info("Vui lòng tải ảnh lên để bắt đầu demo.")

if __name__ == "__main__":
    main()