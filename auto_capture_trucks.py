import cv2
import os
from datetime import datetime
from ultralytics import YOLO
from dotenv import load_dotenv

# Tải các biến môi trường từ file .env
load_dotenv()

def auto_capture_trucks(stream_source, output_dir, skip_frames=10, conf_thresh=0.5):
  """
  Đọc stream từ Camera/Video, dùng YOLO để bắt xe tải và tự động lưu ảnh.
  
  Tham số:
  - stream_source: Link RTSP camera hoặc đường dẫn file video mp4 (Dùng 0 để test Webcam).
  - output_dir: Thư mục lưu ảnh.
  - skip_frames: Bỏ qua số lượng frame nhất định (vd 15) để tránh chụp liên tục 30 ảnh/giây gây trùng lặp.
  - conf_thresh: Độ tin cậy tối thiểu để xác nhận đó là xe tải.
  """
  
  # Tạo thư mục chứa ảnh nếu chưa có
  os.makedirs(output_dir, exist_ok=True)
  
  # Tải mô hình YOLOv8 pre-trained (Model này đã học sẵn 80 class COCO, có class xe tải)
  model = YOLO('model_loader/yolov8n.pt')
  print(model.info())
  
  # Trong bộ dữ liệu COCO: 'truck' có class ID = 7
  TRUCK_CLASS_ID = [7] 
  
  print(f"Connecting to stream: {stream_source}")
  cap = cv2.VideoCapture(stream_source)
  
  if not cap.isOpened():
    print("Lỗi: Không thể kết nối tới Camera/Video.")
    return

  frame_count = 0
  saved_count = 0
  
  while True:
    ret, frame = cap.read()
    if not ret:
      print("Đã hết video stream hoặc mất kết nối.")
      break
      
    frame_count += 1
    
    if frame_count % skip_frames != 0:
      continue
      
    # Đưa ảnh qua mô hình YOLO để nhận diện xe tải
    results = model.predict(frame, classes=TRUCK_CLASS_ID, conf=conf_thresh, verbose=False)
    boxes = results[0].boxes
    
    # Nếu phát hiện có xe tải (độ dài mảng boxes > 0)
    if len(boxes) > 0:
      # Optional: Lọc thêm điều kiện diện tích Box để tránh xe ở quá xa
      # Lấy toạ độ xe đầu tiên
      x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy()
      box_area = (x2 - x1) * (y2 - y1)
      frame_area = frame.shape[0] * frame.shape[1]
      
      # Chỉ lưu nếu xe chiếm ít nhất 10% diện tích khung hình (tránh nhiễu xe xa lấp ló)
      if box_area / frame_area > 0.01:
        # Lưu frame gốc (KHÔNG có box vẽ đè lên) để mang đi gán nhãn Pose
        datestamp = datetime.now().strftime('%Y-%m-%d')
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"truck_{timestamp}_{frame_count}.jpg"

        # Tạo folder ngày
        save_dir = os.path.join(output_dir, datestamp)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        cv2.imwrite(save_path, frame)
        saved_count += 1
        print(f"[+] Đã lưu {filename} vào {save_dir} (Tổng: {saved_count} ảnh)")
        
        # Vẽ box lên frame để xem trực quan trên màn hình
        frame = results[0].plot()

    # Hiển thị luồng stream kèm Box nhận diện (nếu có)
    cv2.imshow("Auto Data Capture", frame)
    
    # Bấm phím 'q' để thoát an toàn
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
      
  cap.release()
  cv2.destroyAllWindows()
  print(f"\n[INFO] Hoàn thành! Đã thu thập tự động được {saved_count} ảnh xe tải.")

if __name__ == "__main__":
  # Lấy thông tin nhạy cảm từ file .env. Nếu không khai báo trong .env, mặc định sẽ dùng "0" (Webcam)
  STREAM_SOURCE = os.getenv("RTSP_STREAM_URL", "0")
  
  # Ảnh sẽ được lưu vào mục dataset_raw bên trong thư mục detect_dump_truck
  OUTPUT_DIRECTORY = "dataset_raw"
  
  # Khởi chạy tool
  auto_capture_trucks(STREAM_SOURCE, OUTPUT_DIRECTORY, skip_frames=10, conf_thresh=0.5)
