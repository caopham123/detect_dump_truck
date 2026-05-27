import cv2
import os
from datetime import datetime, date
from ultralytics import YOLO
from dotenv import load_dotenv

# Tải các biến môi trường từ file .env
load_dotenv()

def auto_capture_trucks(stream_source, skip_frames=15, conf_thresh=0.5):
  """
  Đọc stream từ Camera/Video, dùng YOLO để bắt xe tải và tự động lưu ảnh.
  
  Tham số:
  - stream_source: Link RTSP camera hoặc đường dẫn file video mp4 (Dùng 0 để test Webcam).
  - skip_frames: Bỏ qua số lượng frame nhất định (vd 15) để tránh chụp liên tục 30 ảnh/giây gây trùng lặp.
  - conf_thresh: Độ tin cậy tối thiểu để xác nhận đó là xe tải.
  """
  date_now = date.today().strftime("%Y-%m-%d")
  output_dir = f"training/dataset_raw/{date_now}"
  
  # Tạo thư mục chứa ảnh nếu chưa có
  os.makedirs(output_dir, exist_ok=True)
  frame_list = [file for file in os.listdir(output_dir) if file.endswith(".jpg")]
  curr_frame_count = len(frame_list)
  print(f"curr_frame_count: {curr_frame_count}")
  
  # Tải mô hình YOLOv8 pre-trained (Model này đã học sẵn 80 class COCO, có class xe tải)
  # model = YOLO('model_loader/yolov8n.pt')
  model = YOLO('training/model_loader/yolov8s.pt')
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
        filename = f"truck_{timestamp}_{curr_frame_count + frame_count}.jpg"

        cv2.imwrite(os.path.join(output_dir, filename), frame)
        saved_count += 1
        print(f"[+] Đã lưu {filename} vào {output_dir} (Tổng: {saved_count} ảnh)")
        
        # Vẽ box lên frame để xem trực quan trên màn hình
        frame = results[0].plot()

    # Hiển thị luồng stream kèm Box nhận diện (nếu có)
    cv2.imshow("Auto Data Capture", cv2.resize(frame, (960, 600)))
    
    # Bấm phím 'q' để thoát an toàn
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
      
  cap.release()
  cv2.destroyAllWindows()
  print(f"\n[INFO] Hoàn thành! Đã thu thập tự động được {saved_count} ảnh xe tải.")

if __name__ == "__main__":
  # video_paths = "training/dataset/test/video07-05-23.mp4"
  # Lấy thông tin nhạy cảm từ file .env. Nếu không khai báo trong .env, mặc định sẽ dùng "0" (Webcam)
  STREAM_SOURCE = os.getenv("RTSP_STREAM_URL", "0")
  
  # Khởi chạy tool
  auto_capture_trucks(STREAM_SOURCE, skip_frames=10, conf_thresh=0.5)
  # auto_capture_trucks(video_paths, OUTPUT_DIRECTORY, skip_frames=10, conf_thresh=0.5)
