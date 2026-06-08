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
  
  pipelines = []
  
  # Visit all of cameras in CAMERAS list
  if not CAMERAS:
    logger.error("Không tìm thấy luồng camera nào được cấu hình trong tệp settings.")
    return

  # Load separate model for each camera pipeline
  for cam_config in CAMERAS:
    cam_id = cam_config.get("camera_id", "Unknown")
    logger.info(f"Đang khởi tạo luồng xử lý: {cam_id}")
    cam_model = YOLO(MODEL_PATH)
    pipeline = Inference(cam_config, cam_model, conf=CONF_THRESHOLD)
    pipelines.append(pipeline)
      
  logger.info("Hệ thống đã sẵn sàng xử lý. Nhấn phím 'ESC' tại cửa sổ để dừng chương trình.")
  
  try:
    while pipelines:
      # Duyệt qua bản sao của list (pipelines[:]) để có thể xoá phần tử an toàn
      for pipeline in pipelines[:]:
        status = pipeline.process_frame()
        
        # Nếu hàm trả về False (Mất kết nối hoặc người dùng bấm X tắt cửa sổ)
        if status is False:
          pipeline.close()
          pipelines.remove(pipeline)
          logger.info(f"Đã dừng luồng camera: {pipeline.camera_id}. Các luồng khác vẫn tiếp tục.")
          
      # Kiểm tra ngắt từ bàn phím (Bấm ESC để thoát TOÀN BỘ)
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