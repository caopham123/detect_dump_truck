import cv2
import numpy as np
from datetime import datetime
from config.settings import INFERENCE_RESIZE, CLASS_COLORS, CLASS_LABELS, DISPLAY_SKIP, PROCESS_SKIP, LOG_FILES, CONF_THRESHOLD
from src.tracker_voting import TrackerVoting
from src.camera_stream import CameraStream

class Inference:
  def __init__(self, cameras, model_instance):
    self.camera_id = cameras.get('camera_id', 'unknown')
    self.rtsp_url = cameras.get('rtsp_url', 'unknown')

    # Convert ROI from YAML config to a Numpy array of points for cv2.polylines
    roi_points = cameras.get('roi_polygon', [])
    self.roi_polygon = np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2)) if roi_points else None
    
    self.reader = CameraStream(self.camera_id, self.rtsp_url).start()  # Start the camera stream in a separate thread
    self.model = model_instance  # Load the YOLO model instance
    self.tracker_voting = TrackerVoting()  # Initialize the tracker voting system
    self.frame_count = 0  # Counter to keep track of the number of frames processed
    self.latest_tracks = []  # Store the latest tracks detected in the current frame
    
  def process_frame(self):
    ''' Read a frame from the camera stream, run inference, update tracker voting, 
    and return the processed frame with detections and alerts. '''
    ret, frame = self.reader.read()       # Get the latest frame written from the camera stream
    if ret is False or frame is None:
      print(f"[WARNING] {self.camera_id} - No frame received from camera stream.")
      return None
    self.frame_count += 1
    frame_resized = cv2.resize(frame, INFERENCE_RESIZE)
    
    ''' 1. Processevery PROCESS_SKIP frames and run inference to reduce load and focus on key moments. '''
    if self.frame_count % PROCESS_SKIP == 0:
      results = self.model.track(frame_resized, persist=True, conf=CONF_THRESHOLD, verbose=False)  # Run tracking on the resized frame
      self.latest_tracks = []
      available_track_ids = []
      
      if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
          x_center, y_center, _, _ = map(int, box.xywh[0].tolist())  # Get the center coordinates of the bounding box
          x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Get the bounding box coordinates
          
          class_id = int(box.cls)  # Get the class ID for the detected object
          conf = box.conf[0].item()  # Get the confidence score for the detected object
          track_id = int(box.id[0].item()) if box.id is not None else None  # Get the track ID for the detected object
          
          if track_id is None: continue
          available_track_ids.append(track_id)  # Add the track ID to the list of available track IDs
          
          ''' Check if the center of the bounding box is within the defined ROI polygon. '''
          in_roi = cv2.pointPolygonTest(self.roi_polygon, (x_center, y_center), False) >= 0 if self.roi_polygon is not None else False
          if in_roi: 
            self.tracker_voting.add_vote(track_id, class_id)  # Add the class prediction to the tracker voting system
            alert_triggered = self.tracker_voting.check_alert(track_id, self.camera_id)  # Check if an alert should be triggered for this track ID
          else: alert_triggered = False
          
          self.latest_tracks.append({   # Store the latest track information
            'track_id': track_id,
            'class_id': class_id,
            'conf': conf,
            'bbox': (x1, y1, x2, y2),
            'in_roi': in_roi,
            'alert_triggered': alert_triggered
          })
        
        # Remove track IDs that are no longer available on screen to prevent memory leak and reuse track IDs for new trucks
        self.tracker_voting.remove_unavailable_tracks(available_track_ids)
        
    ''' 2. Display the video every DISPLAY_SKIP frames with bounding boxes, labels, and alerts. '''
    if self.frame_count % DISPLAY_SKIP == 0:
      cv2.polylines(frame_resized, [self.roi_polygon], isClosed=True, color=(81, 152, 232), thickness=2) if self.roi_polygon is not None else None
      # Draw bounding boxes, labels and alerts from the latest tracks
      for track in self.latest_tracks:
        tx1, ty1, tx2, ty2 = track['bbox']
        color = CLASS_COLORS.get(track['class_id'], (0, 0, 0))  # Get the color for the class ID, default to black if not found
        class_name = CLASS_LABELS.get(track['class_id'], 'UNKNOWN')
        track_id = track['track_id']
        conf = track['conf']
        label = f"Track: {track_id} | {class_name} {conf:.0%}"
        
        if track['alert_triggered']:
          cv2.putText(frame_resized, "CANH BAO: CHUA NANG THUNG", (tx1, max(15, ty1 - 35)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
          
        cv2.rectangle(frame_resized, (tx1, ty1), (tx2, ty2), color, 2)
        cv2.putText(frame_resized, label, (tx1, max(10, ty1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
      cv2.imshow(f'Camera {self.camera_id} - Tracking', frame_resized)
  
  def close(self):
    ''' Stop the camera stream and release resources. '''
    self.reader.stop()
    cv2.destroyAllWindows()
          