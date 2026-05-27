import cv2, os
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD"))
RTSP_STREAM_URL = os.getenv("RTSP_STREAM_URL")

# 1. Load the trained model
print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

def detect_and_draw(image_path, output_path):
  
  # 2. Read the input image
  img = cv2.imread(image_path)
  if img is None:
    print(f"Error: Could not read image {image_path}")
    return
  
  img_basename = os.path.basename(image_path)
  os.makedirs(output_path, exist_ok=True)

  # 3. Run inference
  print(f"Running inference on {image_path}...")
  results = model.track(source=img, persist=True, conf=CONF_THRESHOLD)
  
  # 4. Process the results
  for result in results:
    # result.boxes contains bounding box info
    boxes = result.boxes
    # result.keypoints contains keypoints info (x, y, conf)
    keypoints = result.keypoints
    # Check if tracking IDs are available
    if boxes.id is not None:
      track_ids = boxes.id.int().cpu().tolist()
    else:
      track_ids = []
    
    if keypoints is not None and len(keypoints) > 0:
      # Iterate over each detected truck in the image
      for i in range(len(boxes)):
        # Extract box coordinates: [x1, y1, x2, y2]
        box = boxes[i].xyxy[0].cpu().numpy()
        # Extract keypoints: [[x, y, conf], [x, y, conf], [x, y, conf]]
        kpts = keypoints[i].data[0].cpu().numpy()
        track_id = track_ids[i] if i < len(track_ids) else -1
        
        # Draw the bounding box (Blue color)
        x1, y1, x2, y2 = map(int, box[:4])
        label_text = f"Truck_id: {track_id}" if track_id != -1 else "dump_truck"
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Colors for points: Orange (V0), Green (V1), Yellow (V2) in BGR format
        colors = [(0, 165, 255), (0, 255, 0), (0, 255, 255)]
        labels = ["V0 (Cabin)", "V1 (Hinge)", "V2 (Tail)"]
        
        valid_points = []
        
        # Draw the 3 keypoints
        for j, kpt in enumerate(kpts):
          if len(kpt) >= 2:
            x, y = int(kpt[0]), int(kpt[1])
            conf = kpt[2] if len(kpt) == 3 else 1.0

            # Only draw if confidence is reasonably high (> 0.5)
            if conf > CONF_THRESHOLD:
              color = colors[j % len(colors)]
              label = labels[j % len(labels)]

              # Draw point
              cv2.circle(img, (x, y), 6, color, -1)
              # Draw label
              cv2.putText(img, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

              valid_points.append((x, y))
              
        # Optional: Draw lines connecting the skeleton if all 3 points are valid
        if len(valid_points) == 3:
          cv2.line(img, valid_points[0], valid_points[1], (255, 0, 255), 2) # Cabin to Hinge (Purple)
          cv2.line(img, valid_points[1], valid_points[2], (255, 0, 255), 2) # Hinge to Tail (Purple)

  # 5. Save and display the result
  cv2.imshow("YOLO-Pose Detection", cv2.resize(img, (1000, 750)))
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  cv2.imwrite(os.path.join(output_path, img_basename), img)
  print(f"Result saved to {output_path}")
  
def detect_and_video(video_path, output_path, skip_frame=10):
  # 2. Read the input video
  video = cv2.VideoCapture(video_path)
  if video is None:
    print(f"Error: Could not play video {video_path}")
    return
  
  os.makedirs(output_path, exist_ok=True)
  frame_count = 0
  while True:
    ret, frame = video.read()
    if not ret:
      print(f"Error: Could not play video {video_path}")
      break

    frame_count += 1
    if frame_count % skip_frame != 0:
      continue

    # 3. Run inference
    # print(f"Running inference on {video_path}...")
    results = model.track(source=frame, persist=True, conf=CONF_THRESHOLD)
  
    # 4. Process the results
    for result in results:
      # result.boxes contains bounding box info
      boxes = result.boxes
      # result.keypoints contains keypoints info (x, y, conf)
      keypoints = result.keypoints
      # Check if tracking IDs are available
      if boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()
      else:
        track_ids = []
      
      if keypoints is not None and len(keypoints) > 0:
        # Iterate over each detected truck in the image
        for i in range(len(boxes)):
          # Extract box coordinates: [x1, y1, x2, y2]
          box = boxes[i].xyxy[0].cpu().numpy()
          # Extract keypoints: [[x, y, conf], [x, y, conf], [x, y, conf]]
          kpts = keypoints[i].data[0].cpu().numpy()
          track_id = track_ids[i] if i < len(track_ids) else -1
          
          # Draw the bounding box (Blue color)
          x1, y1, x2, y2 = map(int, box[:4])
          label_text = f"Truck_id: {track_id}" if track_id != -1 else "dump_truck"
          cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
          cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 4)
          
          # Colors for points: Orange (V0), Green (V1), Yellow (V2) in BGR format
          colors = [(0, 165, 255), (0, 255, 0), (0, 255, 255)]
          labels = ["V0 (Cabin)", "V1 (Hinge)", "V2 (Tail)"]
          
          valid_points = []
          
          # Draw the 3 keypoints
          for j, kpt in enumerate(kpts):
            if len(kpt) >= 2:
              x, y = int(kpt[0]), int(kpt[1])
              conf = kpt[2] if len(kpt) == 3 else 1.0

              # Only draw if confidence is reasonably high (> 0.5)
              if conf > CONF_THRESHOLD:
                color = colors[j % len(colors)]
                label = labels[j % len(labels)]

                # Draw point
                cv2.circle(frame, (x, y), 6, color, -1)
                # Draw label
                cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

                valid_points.append((x, y))
                
          # Optional: Draw lines connecting the skeleton if all 3 points are valid
          if len(valid_points) == 3:
            cv2.line(frame, valid_points[0], valid_points[1], (255, 0, 255), 2) # Cabin to Hinge (Purple)
            cv2.line(frame, valid_points[1], valid_points[2], (255, 0, 255), 2) # Hinge to Tail (Purple)
  
    # 5. Save and display the result
    cv2.imwrite(os.path.join(output_path, f"video03_frame_{frame_count}.jpg"), frame)
    print(f"Result saved to {output_path}")
  video.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  TEST_IMAGE = "training/dataset/test/pic10.jpg"
  TEST_VIDEO = "training/dataset/test/video03.mp4"
  OUTPUT_PATH = "training/dataset/test/output"
  detect_and_draw(TEST_IMAGE, OUTPUT_PATH)
  # detect_and_video(TEST_VIDEO, OUTPUT_PATH, 10)

