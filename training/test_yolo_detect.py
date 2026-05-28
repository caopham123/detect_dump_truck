from ultralytics import YOLO
from dotenv import load_dotenv
import os, cv2
from datetime import datetime, date
from collections import deque, Counter, defaultdict

load_dotenv()
MODEL_PATH = os.getenv('MODEL_PATH')
RTSP_STREAM = os.getenv('RTSP_STREAM_URL')
CONF_THRESHOLD = float(os.getenv('CONF_THRESHOLD', 0.5))
SKIP_FRAMES = int(os.getenv('SKIP_FRAMES', 15))
VOTE_COUNT = int(os.getenv('MAJORITY_VOTE', 5))

# Setup color and label for classes
CLASS_COLORS = {
  0: (0, 0, 255),    # truck_raised  → ĐỎ (BGR)
  1: (0, 200, 0),    # truck_lowered → XANH (BGR)
}
CLASS_LABELS = {
  0: "CHUA HA",
  1: "DA HA",
}
# Load model
model_detect = YOLO(MODEL_PATH)
print(f"model_names: {model_detect.names}")

date_now = date.today().strftime("%Y-%m-%d")
# date_now = "2026-05-26"
print(f"date_now: {date_now}")

def test_image(img_path):
  img = cv2.imread(img_path)
  if img is None:
    print(f"Error: Could not read image at {img_path}")
    return
  img_resized = cv2.resize(img, (960, 600))       # resize current frame and process
  results = model_detect.track(img_resized, persist=True, conf=CONF_THRESHOLD, verbose=True)
  is_detected = False

  # Draw ROI line
  cv2.rectangle(img_resized, (400, 100), (600, 300), (110, 255, 20), 1)
  
  for box in results[0].boxes:
    is_detected = True
    # 1. Trích xuất tọa độ xyxy
    xyxy = box.xyxy[0].tolist()
    xywh = box.xywh[0].tolist()
    x1, y1, x2, y2 = map(int, xyxy)
    x_center, y_center, _, _ = map(int, xywh)
    # 2. Xác định class_id, tên class và track_id
    class_id = int(box.cls[0].item())
    conf = box.conf[0].item()
    track_id = int(box.id[0].item()) if box.id is not None else None
    # In kết quả ra console
    track_str = f" | Track ID: {track_id}" if track_id is not None else "No tracking"
    print(f"Detected: {track_str} | {CLASS_LABELS[class_id]} (ID: {class_id}) | Conf: {conf:.0%} | BBox: [{x1}, {y1}, {x2}, {y2}]")
    
    # Check if the object is in ROI
    if 400 <= x_center <= 600 and 100 <= y_center <= 300:
      print(f"Truck detected in ROI: Track_ID {track_id} | {CLASS_LABELS[class_id]} | Conf: {conf:.0%} | BBox: [{x1}, {y1}, {x2}, {y2}]")

    # 3. Vẽ kết quả lên ảnh
    color = CLASS_COLORS[class_id]
    cv2.rectangle(img_resized, (x1, y1), (x2, y2), color, 2)
    label = f"{CLASS_LABELS[class_id]} {conf:.0%}"
    if track_id is not None:
      label = f"ID:{track_id}" + " | " + label
    cv2.putText(img_resized, label, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


  ## 4. Luu ket qua trong file output
  # basename = os.path.basename(img_path)
  # filename = basename.split(".jpg")[0]
  # os.makedirs(f"training/dataset/test/{date_now}_output_v7", exist_ok=True)
  # cv2.imwrite(f"training/dataset/test/{date_now}_output_v7/{filename}_detected.jpg", img)
  # print(f"Saved detected image to training/dataset/test/output/{date_now}/{filename}_detected.jpg")
  
  # # 5. Hien thi hinh anh
  cv2.imshow('Detected Trucks', img_resized)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def test_video_frames(video_path, SKIP_FRAMES=10, CONF=0.7):
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print(f"Cannot open video: {video_path}")
    return
  frame_count = 0
  os.makedirs(f"training/dataset/test/{date_now}_output_v7", exist_ok=True)

  # Track ID voting history: key is track_id, value is number of frames
  track_votes = defaultdict(lambda: deque(maxlen=VOTE_COUNT))

  # Define the ROI boundaries
  ROI_X_MIN, ROI_X_MAX, ROI_Y_MIN, ROI_Y_MAX = 150, 350, 100, 300
  
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    frame_count += 1
    if frame_count % SKIP_FRAMES != 0:
      continue
    frame_resized = cv2.resize(frame, (960, 600))       # resize current frame and process
    cv2.rectangle(frame_resized, (ROI_X_MIN, ROI_Y_MIN), (ROI_X_MAX, ROI_Y_MAX), (81, 152, 232), 2)

    results = model_detect.track(frame_resized, persist=True, conf=CONF, verbose=False)
    is_detected = False

    if results is not None and len(results[0].boxes) > 0:
      is_detected = True
      print(f"---------- Frame {frame_count} found truck:")

      for box in results[0].boxes:
        # 1. Coordinates extractions
        xyxy = box.xyxy[0].tolist()
        xywh = box.xywh[0].tolist()
        x1, y1, x2, y2 = map(int, xyxy)
        x_center, y_center, _, _ = map(int, xywh)

        # 2. Extract class id, class name and track id
        class_id = int(box.cls[0].item())
        conf = box.conf[0].item()
        track_id = int(box.id[0].item()) if box.id is not None else None
        
        if track_id is None: continue
        
        # 3. Draw bounding box
        color = CLASS_COLORS.get(class_id, (0, 0, 0)) # Default to black if class_id not found
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 1)

        label = f"{CLASS_LABELS.get(class_id, 'UNKNOWN')} {conf:.0%}"
        if track_id is not None:
          label = f"Track: {track_id} | " + label
        else: label = "No Track | " + label
        cv2.putText(frame_resized, label, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        print(f"Detected: {label} | BBox: [{x1}, {y1}, {x2}, {y2}]")
        
        # 4. Check if center of box is in ROI
        is_in_roi = (ROI_X_MIN <= x_center <= ROI_X_MAX and ROI_Y_MIN <= y_center <= ROI_Y_MAX)
        
        if is_in_roi and track_id is not None:
          print(f"Truck detected in ROI: Track_ID {track_id}")
          # Store track_id and class_id in trackvotes
          track_votes[track_id].append(class_id)
          # Evaluate the 3/5 voting for each track
          if len(track_votes[track_id]) == VOTE_COUNT:
            class_raised_count = track_votes[track_id].count(0)     # Count number of frames with 0: truck_raised
            if class_raised_count >= int(VOTE_COUNT * 0.75):
              alert_warning(track_id, frame_count, x1, y1, x2, y2, conf, color)
              write_log(f"Frame: {frame_count:03d} | Truck ID {track_id} detected in ROI | "
                        f"Class {CLASS_LABELS.get(0, 'TRUCK_RAISED')}: {class_raised_count}/{VOTE_COUNT}")
              

    # #  Luu ket qua trong file output
    # if is_detected:
    #   basename = os.path.basename(video_path)
    #   filename = basename.split(".mp4")[0]
    #   os.makedirs(f"training/dataset/test/{date_now}_output_v7", exist_ok=True)
    #   cv2.imwrite(f"training/dataset/test/{date_now}_output_v7/{frame_count:03d}_{filename}_detected.jpg", frame_resized)
    #   print(f"Saved detected image to training/dataset/test/{date_now}_output_v5/{frame_count:03d}_{filename}_detected.jpg")
    
    # #  Hiển thị video
    cv2.imshow('Tracking Video', frame_resized)
    if cv2.waitKey(1) == 27:
      break
  cap.release()
  cv2.destroyAllWindows()

def write_log(msg):
  with open(f"training/dataset/test/{date_now}_output_v7/log_{date_now}.txt", "a", encoding="utf-8") as f:
    f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " | " + msg + "\n")

def alert_warning(track_id, frame_count, x1, y1, x2, y2, conf, color):
  print(f"WARNING!!! Track ID {track_id} detected in ROI - Truck is raising")

if __name__ == '__main__':
  test_video_frames(RTSP_STREAM,5, 0.6)

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

  # test_video_frames("training/dataset/test/video05-05-23.mp4", 10, 0.7)
  # test_video_frames("training/dataset/test/video04-05-23.mp4", 10, 0.5)
  # test_video_frames("training/dataset/test/video03.mp4", 15, 0.5)
  # test_video_frames("training/dataset/test/video02.mp4", 15, 0.5)
  # test_video_frames("training/dataset/test/video01.mp4", 15, 0.5)
