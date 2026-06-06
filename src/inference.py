import cv2, os
import numpy as np
from datetime import datetime,date
from config.settings import DISPLAY_RESIZE, CLASS_COLORS, CLASS_LABELS, DISPLAY_SKIP, PROCESS_SKIP, CONF_THRESHOLD, SNAPSHOT_DIR, TRACK_BUFFER
from src.tracker_voting import TrackerVoting
from src.camera_stream import CameraStream
from utils.logger import logger
from ultralytics.trackers.basetrack import BaseTrack

class Inference:
  def __init__(self, cameras, model_instance, conf):
    self.camera_id = cameras.get('camera_id', 'unknown')
    self.rtsp_url = cameras.get('rtsp_url', 'unknown')
    self.confidence = conf

    # Create tracker config file to sync TRACK_BUFFER
    self.tracker_yaml_path = "config/custom_tracker.yaml"
    with open(self.tracker_yaml_path, "w", encoding="utf-8") as f:
      f.write("tracker_type: bytetrack\n")
      f.write("track_high_thresh: 0.5\n")
      f.write("track_low_thresh: 0.2\n")
      f.write("new_track_thresh: 0.5\n")
      f.write("match_thresh: 0.7\n")
      f.write("fuse_score: True\n")
      f.write(f"track_buffer: {TRACK_BUFFER}\n")

    # Convert ROI from YAML config to a Numpy array of points for cv2.polylines
    roi_points = cameras.get('roi_polygon', [])
    self.roi_polygon = np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2)) if roi_points else None
    
    self.reader = CameraStream(self.camera_id, self.rtsp_url).start()  # Start the camera stream in a separate thread
    self.model = model_instance  # Load the YOLO model instance
    self.tracker_voting = TrackerVoting()  # Initialize the tracker voting system
    self.frame_count = 0  # Counter to keep track of the number of frames processed
    self.latest_tracks = []  # Store the latest tracks detected in the current frame
    
    # Draw polygon on window
    self.window_name = f'Camera {self.camera_id} - Tracking'
    cv2.namedWindow(self.window_name)
    self.dragging_point_idx = -1
    cv2.setMouseCallback(self.window_name, self.mouse_callback)

    # Setup snapshot
    self.last_snapshot_time = 0
    self.snapshot_cooldown = 2 # seconds
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)   # Ensure the logs directory exists

    self.formatDate = ''
    self.formatTime = ''
    self.formatDateTime = ''

  def get_date_time(self):
    self.now = datetime.now()
    self.formatDate = self.now.strftime('%Y-%m-%d')
    self.formatTime = self.now.strftime('%H:%M:%S')
    self.formatDateTime = self.now.strftime('%Y-%m-%d %H:%M:%S')  

  def mouse_callback(self, event, x, y, flags, param):
    ''' Handle mouse events to drag and drop ROI polygon points '''
    if self.roi_polygon is None: return
        
    if event == cv2.EVENT_LBUTTONDOWN:
      # Find if clicked near any point
      for i, point in enumerate(self.roi_polygon):
        px, py = point[0]
        if abs(px - x) < 15 and abs(py - y) < 15:  # 15px threshold
          self.dragging_point_idx = i
          break
                
    elif event == cv2.EVENT_MOUSEMOVE:
      # Update point if dragging
      if self.dragging_point_idx != -1:
        self.roi_polygon[self.dragging_point_idx][0] = [x, y]
            
    elif event == cv2.EVENT_LBUTTONUP:
      # Stop dragging and print the new coordinates
      if self.dragging_point_idx != -1:
        logger.info(f"{self.camera_id} - New ROI Polygon points updated! Copy this to constants.yaml:")
        logger.info(f"roi_polygon: {self.roi_polygon.reshape(-1, 2).tolist()}")
        self.dragging_point_idx = -1
    
  def process_frame(self):
    ''' Read a frame from the camera stream, run inference, update tracker voting, 
    and return the processed frame with detections and alerts. '''
    ret, frame = self.reader.read()       # Get the latest frame written from the camera stream
    if ret is False or frame is None:
      logger.warning(f"{self.camera_id} - No frame received from camera stream.")
      return None
    self.frame_count += 1
    frame_resized = cv2.resize(frame, DISPLAY_RESIZE)
    
    ''' 1. Processevery PROCESS_SKIP frames and run inference to reduce load and focus on key moments. '''
    if self.frame_count % PROCESS_SKIP == 0:
      results = self.model.track(frame_resized, persist=True, conf=self.confidence, tracker=self.tracker_yaml_path, verbose=False)  # Run tracking on the resized frame
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
        
        # Reset YOLO's internal track ID counter when the screen is completely empty
        # This prevents the ID number from reaching 10000+ over long running times
        if not available_track_ids and len(self.tracker_voting.track_history) == 0:
          BaseTrack.reset_id()
        
    ''' 2. Display the video every DISPLAY_SKIP frames with bounding boxes, labels, and alerts. '''
    if self.frame_count % DISPLAY_SKIP == 0:
      if self.roi_polygon is not None:
        cv2.polylines(frame_resized, [self.roi_polygon], isClosed=True, color=(81, 152, 232), thickness=2)
        # Draw circles at polygon vertices for drag-and-drop visibility
        for point in self.roi_polygon:
          cv2.circle(frame_resized, tuple(point[0]), 5, (0, 0, 255), -1)
      
      # Draw bounding boxes, labels and alerts from the latest tracks
      for track in self.latest_tracks:
        tx1, ty1, tx2, ty2 = track['bbox']
        color = CLASS_COLORS.get(track['class_id'], (0, 0, 0))  # Get the color for the class ID, default to black if not found
        class_name = CLASS_LABELS.get(track['class_id'], 'UNKNOWN')
        track_id = track['track_id']
        conf = track['conf']
        label = f"Track: {track_id} | {class_name} {conf:.0%}"
        
        if track['alert_triggered']:
          cv2.rectangle(frame_resized, (tx1, ty1), (tx2, ty2), color, 2)
          cv2.putText(frame_resized, "CANH BAO: CHUA HA THUNG", (tx1, max(15, ty1 - 35)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
          # Fix filename colons and only save snapshot on alert
          image_folder = os.path.join(SNAPSHOT_DIR, date.today().strftime('%Y-%m-%d'))
          if not os.path.exists(image_folder): 
            os.makedirs(image_folder)
          snapshot_path = os.path.join(image_folder, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
          cv2.imwrite(snapshot_path, frame_resized)
          logger.info(f"Saved violation snapshot to {snapshot_path}")
          track['alert_triggered'] = False  # Tránh lưu ảnh 2 lần do DISPLAY_SKIP < PROCESS_SKIP

        # if track['class_id'] == 0:
        cv2.rectangle(frame_resized, (tx1, ty1), (tx2, ty2), color, 2)
        cv2.putText(frame_resized, label, (tx1, max(10, ty1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
      cv2.imshow(self.window_name, frame_resized)
  
  def close(self):
    ''' Stop the camera stream and release resources. '''
    self.reader.stop()
    cv2.destroyAllWindows()
          