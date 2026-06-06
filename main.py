# main.py
import cv2
from ultralytics import YOLO
from config.settings import MODEL_PATH, CAMERAS, CONF_THRESHOLD
from src.inference import Inference
from utils.logger import logger

def main():
  logger.info("=" * 50)
  logger.info("==== Khởi động hệ thống phát hiện xe chưa nâng thùng ====")
  logger.info("Đang nạp mô hình YOLOv8 vào bộ nhớ...")
  
  # Nạp mô hình một lần để phân phối cho các tiến trình con
  model = YOLO(MODEL_PATH)
  
  pipelines = []
  
  # Duyệt trực tiếp qua cấu trúc danh sách Camera mới trong tệp settings
  if not CAMERAS:
    logger.error("Không tìm thấy luồng camera nào được cấu hình trong tệp settings.")
    return

  for cam_config in CAMERAS:
    cam_id = cam_config.get("camera_id", "Unknown")
    logger.info(f"Đang khởi tạo luồng xử lý: {cam_id}")
    pipeline = Inference(cam_config, model, conf=CONF_THRESHOLD)
    pipelines.append(pipeline)
      
  logger.info("Hệ thống đã sẵn sàng xử lý. Nhấn phím 'ESC' tại cửa sổ để dừng chương trình.")
  
  try:
    while True:
      # Chạy bước xử lý tuần tự cho từng camera
      for pipeline in pipelines:
        pipeline.process_frame()
          
      # Kiểm tra ngắt từ bàn phím
      if cv2.waitKey(1) == 27:
        break
  except KeyboardInterrupt:
    logger.info("Đang tắt hệ thống nhận diện...")
  finally:
    # Giải phóng tài nguyên camera an toàn
    for pipeline in pipelines:
      pipeline.close()
    cv2.destroyAllWindows()
    logger.info("Hệ thống đã dừng an toàn.")

if __name__ == "__main__":
  main()
  # print("Model path:", MODEL_PATH)