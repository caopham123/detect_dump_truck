import os
import cv2
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
IMAGE_DIR = "./raw_images"       # Thư mục chứa ảnh gốc (.jpg, .png) cần cắt xe ben
OUTPUT_DIR = "./cropped_trucks"   # Thư mục đầu ra để lưu các ảnh xe ben đã cắt
CONFIDENCE_THRESHOLD = 0.5       # Độ tin cậy tối thiểu để nhận diện xe ben
MODEL_PATH = "yolov8n.pt"        # Mô hình YOLOv8n pretrained để phát hiện đối tượng

def setup_directories():
    """Tạo các thư mục nếu chưa tồn tại."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Đã chuẩn bị thư mục:")
    print(f"  - Thư mục ảnh gốc: {os.path.abspath(IMAGE_DIR)}")
    print(f"  - Thư mục ảnh cắt xe: {os.path.abspath(OUTPUT_DIR)}")

def main():
    setup_directories()
    
    # Kiểm tra xem có ảnh nào trong thư mục raw_images chưa
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = [
        p for p in Path(IMAGE_DIR).glob("**/*") 
        if p.suffix.lower() in image_extensions
    ]
    
    if not image_paths:
        print("\n[WARNING] Không tìm thấy ảnh nào trong thư mục './raw_images'!")
        print(">> HƯỚNG DẪN:")
        print(f"1. Hãy copy các ảnh raw chứa xe ben vào thư mục: {IMAGE_DIR}")
        print("2. Chạy lại script này để tiến hành cắt ảnh tự động.")
        return

    print(f"\n[INFO] Đang tải mô hình phát hiện đối tượng: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Lớp xe tải trong bộ dữ liệu COCO là 7 ('truck'), xe ô tô là 2 ('car'), xe buýt là 5 ('bus')
    # Chúng ta chủ yếu tập trung vào lớp 7 ('truck') và lớp 2 ('car' - đối với một số xe tải nhỏ bị nhận nhầm)
    TARGET_CLASSES = [7, 2] 
    
    success_count = 0
    print(f"\n[INFO] Bắt đầu quét và cắt ảnh đối tượng từ {len(image_paths)} tệp tin...")
    
    for idx, img_path in enumerate(image_paths):
        # Đọc ảnh
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[ERR] Không thể đọc ảnh: {img_path.name}")
            continue
            
        # Chạy inference
        results = model.predict(frame, verbose=False)[0]
        
        crop_idx = 0
        # Duyệt qua các bounding box phát hiện được
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Chỉ lọc xe ben/xe tải (truck) hoặc ô tô lớn có độ tự tin cao
            if class_id in TARGET_CLASSES and conf >= CONFIDENCE_THRESHOLD:
                # Lấy tọa độ x1, y1, x2, y2
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Tránh tọa độ âm hoặc vượt quá kích thước ảnh
                h, w, _ = frame.shape
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Cắt vùng xe ben
                cropped_img = frame[y1:y2, x1:x2]
                
                # Bỏ qua nếu ảnh cắt quá nhỏ (tránh nhiễu xa camera)
                if cropped_img.shape[0] < 50 or cropped_img.shape[1] < 50:
                    continue
                    
                # Tạo tên tệp tin độc nhất cho ảnh cắt
                base_name = img_path.stem
                output_name = f"{base_name}_truck_{crop_idx}.jpg"
                output_path = os.path.join(OUTPUT_DIR, output_name)
                
                # Lưu ảnh cắt
                cv2.imwrite(output_path, cropped_img)
                crop_idx += 1
                success_count += 1
                
        if (idx + 1) % 10 == 0 or (idx + 1) == len(image_paths):
            print(f"  -> Đã xử lý: {idx + 1}/{len(image_paths)} ảnh. Đã cắt được {success_count} xe ben.")
            
    print("\n" + "="*50)
    print(f"[HOÀN THÀNH] Đã hoàn tất cắt tự động!")
    print(f"Tổng số ảnh xe ben được cắt: {success_count} ảnh.")
    print(f"Vị trí lưu ảnh: {os.path.abspath(OUTPUT_DIR)}")
    print("="*50)
    print(">> BƯỚC TIẾP THEO:")
    print("1. Vào thư mục './cropped_trucks' kiểm tra các ảnh xe ben đã cắt.")
    print("2. Tạo 2 thư mục 'lowered' và 'raised' để phân loại ảnh vào.")
    print("3. Duyệt nhanh qua các ảnh cắt và kéo thả vào đúng thư mục trạng thái tương ứng.")
    print("4. Sử dụng tập thư mục đó để huấn luyện mô hình phân loại YOLOv8-cls!")

if __name__ == "__main__":
    main()
