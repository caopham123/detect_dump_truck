from ultralytics import YOLO
from dotenv import load_dotenv
import os, cv2
from datetime import datetime, date

load_dotenv()
MODEL_PATH = os.getenv('MODEL_PATH')
model_detect = YOLO(MODEL_PATH)
print(f"model_names: {model_detect.names}")
# date_now = date.today().strftime("%Y-%m-%d")
date_now = "2026-05-26"
print(f"date_now: {date_now}")

def test_image(img_path):
  img = cv2.imread(img_path)
  if img is None:
    print(f"Error: Could not read image at {img_path}")
    return

  results = model_detect.track(img, persist=True, conf=0.5)
  
  for r in results:
    for box in r.boxes:
      # 1. Trích xuất tọa độ xyxy
      b = box.xyxy[0].cpu().numpy() 
      x1, y1, x2, y2 = map(int, b)
      
      # 2. Xác định class_id, tên class và track_id
      class_id = int(box.cls[0].item())
      class_name = model_detect.names[class_id]
      conf = box.conf[0].item()
      track_id = int(box.id[0].item()) if box.id is not None else None
      
      # In kết quả ra console
      track_str = f" | Track ID: {track_id}" if track_id is not None else ""
      print(f"Detected: Track_ID: {track_id}_ {class_name} (ID: {class_id}){track_str} | Conf: {conf:.2f} | BBox: [{x1}, {y1}, {x2}, {y2}]")
      
      # 3. Vẽ kết quả lên ảnh
      cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
      label = f"{class_name} {conf:.2f}"
      if track_id is not None:
        label = f"ID:{track_id}" + " | " + label
      cv2.putText(img, label, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)


  ## 4. Luu ket qua trong file output
  basename = os.path.basename(img_path)
  filename = basename.split(".jpg")[0]
  os.makedirs(f"training/dataset/test/{date_now}_output_v7", exist_ok=True)
  cv2.imwrite(f"training/dataset/test/{date_now}_output_v7/{filename}_detected.jpg", img)
  print(f"Saved detected image to training/dataset/test/output/{date_now}/{filename}_detected.jpg")
  # cv2.imshow('Detected Trucks', cv2.resize(img, (960, 600)))
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

def test_video_frames(video_path, SKIP_FRAMES=5, CONF=0.5):
  cap = cv2.VideoCapture(video_path)
  # Check video is opened
  if not cap.isOpened():
    print(f"Cannot open video: {video_path}")
    exit()

  frame_count = 0
  while True:
    ret, frame = cap.read()
    if not ret:
      print()
      break
    
    frame_count += 1
    if frame_count % SKIP_FRAMES != 0:
      continue
    
    frame_resized = cv2.resize(frame, (960, 600))
    results = model_detect.track(frame_resized, persist=True, conf=CONF)
    is_detected = False

    for r in results:
      for box in r.boxes:
        is_detected = True
        # 1. Trích xuất tọa độ xyxy
        b = box.xyxy[0].cpu().numpy() 
        x1, y1, x2, y2 = map(int, b)
        
        # 2. Xác định class_id, tên class và track_id
        class_id = int(box.cls[0].item())
        class_name = model_detect.names[class_id]
        conf = box.conf[0].item()
        track_id = int(box.id[0].item()) if box.id is not None else None
        
        # In kết quả ra console
        track_str = f" | Track ID: {track_id}" if track_id is not None else ""
        print(f"Detected: Track_{track_id} | {class_name} (ID: {class_id}) | Conf: {conf:.2f} | BBox: [{x1}, {y1}, {x2}, {y2}]")
        
        # 3. Vẽ kết quả lên ảnh
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {conf:.2f}"
        if track_id is not None:
          label = f"ID:{track_id}" + " | " + label
        cv2.putText(frame_resized, label, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    # 4. Luu ket qua trong file output
    if is_detected:
      basename = os.path.basename(video_path)
      filename = basename.split(".mp4")[0]
      os.makedirs(f"training/dataset/test/{date_now}_output_v7", exist_ok=True)
      cv2.imwrite(f"training/dataset/test/{date_now}_output_v7/{frame_count:03d}_{filename}_detected.jpg", frame_resized)
      print(f"Saved detected image to training/dataset/test/{date_now}_output_v5/{frame_count:03d}_{filename}_detected.jpg")

    # # 5. Hiển thị video
    # cv2.imshow('Tracking Video', frame_resized)
    # # Nhấn 'q' để thoát sớm
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #   break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  # RTSP_STREAM = os.getenv('RTSP_STREAM_URL')
  # test_video_frames(RTSP_STREAM, 5, 0.6)

  # test_image('training/dataset/test/pic1.jpg')
  # test_image('training/dataset/test/pic2.jpg')
  # test_image('training/dataset/test/pic3.jpg')
  # test_image('training/dataset/test/pic4.jpg')
  # test_image('training/dataset/test/pic5.jpg')
  # test_image('training/dataset/test/pic6.jpg')
  # test_image('training/dataset/test/pic7.jpg')
  # test_image('training/dataset/test/pic8.jpg')
  # test_image('training/dataset/test/pic9.jpg')
  # test_image('training/dataset/test/pic10.jpg')
  # test_image('training/dataset/test/pic11.jpg')

  # test_video_frames("training/dataset/test/video05-05-23.mp4", 10, 0.5)
  # test_video_frames("training/dataset/test/video04-05-23.mp4", 10, 0.5)
  test_video_frames("training/dataset/test/video03.mp4", 15, 0.5)
  test_video_frames("training/dataset/test/video02.mp4", 15, 0.5)
  test_video_frames("training/dataset/test/video01.mp4", 15, 0.5)

  
