# main.py
import torch

# Fix for PyTorch 2.6 weights_only=True default causing UnpicklingError in Ultralytics 8.2.0
_original_load = torch.load
def _patched_load(*args, **kwargs):
  if 'weights_only' not in kwargs:
    kwargs['weights_only'] = False
  return _original_load(*args, **kwargs)
torch.load = _patched_load

import cv2
from ultralytics import YOLO
from config.settings import MODEL_PATH, CAMERAS, CONF_THRESHOLD
from src.inference import Inference, SkipCameraError
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

  # Start web server
  from web_app import start_web_app
  start_web_app(pipelines)
  logger.info("Khởi chạy máy chủ Web tại http://localhost:5000")
  
  # Load separate model for each camera pipeline
  for cam_config in CAMERAS:
    cam_id = cam_config.get("camera_id", "Unknown")
    logger.info(f"Đang khởi tạo luồng xử lý: {cam_id}")
    cam_model = YOLO(MODEL_PATH)
    try:
      pipeline = Inference(cam_config, cam_model, conf=CONF_THRESHOLD)
      pipelines.append(pipeline)
    except SkipCameraError:
      logger.warning(f"Camera {cam_id} - Bỏ qua camera do không thể kết nối lần đầu. Tiếp tục các camera khác.")
      continue
  logger.info("Hệ thống đã sẵn sàng xử lý. Nhấn phím 'ESC' tại cửa sổ để dừng chương trình.")
  
  try:
    while pipelines:
      for pipeline in pipelines[:]:
        status = pipeline.process_frame()
        # When return False (Camera lost connection or user closed window)
        if status is False:
          pipeline.close()
          pipelines.remove(pipeline)
          logger.info(f"Đã dừng luồng camera: {pipeline.camera_id}. Các luồng khác vẫn tiếp tục.")
          
      # Press ESC to exit ALL
      if cv2.waitKey(1) == 27:
        break
  except KeyboardInterrupt:
    logger.info("... Đang tắt hệ thống nhận diện ...")
  finally:
    # Releasing camera resources safely
    for pipeline in pipelines:
      pipeline.close()
    cv2.destroyAllWindows()
    logger.info("====== Hệ thống đã dừng an toàn ======")

if __name__ == "__main__":
  main()