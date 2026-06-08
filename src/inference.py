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

    # Create tracker config file to sync TRACK_BUFFER (riêng biệt cho từng camera)
    self.tracker_yaml_path = f"config/custom_tracker_{self.camera_id}.yaml"
    with open(self.tracker_yaml_path, "w", encoding="utf-8") as f:
      f.write("tracker_type: bytetrack\n")
      f.write("track_high_thresh: 0.5\n")
      f.write("track_low_thresh: 0.2\n")
      f.write("new_track_thresh: 0.5\n")
      f.write("match_thresh: 0.7\n")
      f.write("fuse_score: True\n")
      f.write(f"track_buffer: {TRACK_BUFFER}\n")

    # Convert ROIs from YAML config to Numpy arrays of points
    self.roi_polygons = []
    if 'roi_polygons' in cameras:
      for poly in cameras['roi_polygons']:
        self.roi_polygons.append(np.array(poly, dtype=np.int32).reshape((-1, 1, 2)))
    elif 'roi_polygon' in cameras: # Fallback to old format
      self.roi_polygons.append(np.array(cameras['roi_polygon'], dtype=np.int32).reshape((-1, 1, 2)))
    
    self.reader = CameraStream(self.camera_id, self.rtsp_url).start()  # Start the camera stream in a separate thread
    self.model = model_instance  # Load the YOLO model instance
    self.tracker_voting = TrackerVoting()  # Initialize the tracker voting system
    self.frame_count = 0  # Counter to keep track of the number of frames processed
    self.latest_tracks = []  # Store the latest tracks detected in the current frame
    
    # Draw polygon on window
    self.window_name = f'Camera {self.camera_id} - Counting'
    cv2.namedWindow(self.window_name)
    self.dragging_point = None # Stores (polygon_index, point_index)
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
    if not self.roi_polygons: return
        
    if event == cv2.EVENT_LBUTTONDOWN:
      # Find if clicked near any point in any polygon
      for poly_idx, poly in enumerate(self.roi_polygons):
        for pt_idx, point in enumerate(poly):
          px, py = point[0]
          if abs(px - x) < 15 and abs(py - y) < 15:  # 15px threshold
            self.dragging_point = (poly_idx, pt_idx)
            return
                
    elif event == cv2.EVENT_MOUSEMOVE:
      # Update point if dragging
      if self.dragging_point is not None:
        poly_idx, pt_idx = self.dragging_point
        self.roi_polygons[poly_idx][pt_idx][0] = [x, y]
            
    elif event == cv2.EVENT_LBUTTONUP:
      # Stop dragging and print the new coordinates
      if self.dragging_point is not None:
        logger.info(f"{self.camera_id} - New ROI Polygons updated! Copy this to constants.yaml:")
        logger.info("roi_polygons:")
        for poly in self.roi_polygons:
          logger.info(f"  - {poly.reshape(-1, 2).tolist()}")
        self.dragging_point = None
    
  def process_frame(self):
    ''' Read a frame from the camera stream, run inference, update tracker voting, 
    and return the processed frame with detections and alerts. '''
    ret, frame = self.reader.read()       # Get the latest frame written from the camera stream
    if ret is False or frame is None:
      logger.warning(f"{self.camera_id} - No frame received from camera stream.")
      return True     # lần đầu ko có frame vẫn tiếp tục
      
    # Kiểm tra nếu người dùng bấm dấu X để tắt cửa sổ của camera này
    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
      logger.info(f"{self.camera_id} - Cửa sổ đã bị đóng.")
      return False
    self.frame_count += 1
    frame_resized = cv2.resize(frame, DISPLAY_RESIZE)
    h, w, _ = frame_resized.shape
    
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
          
          ''' Check if the center of the bounding box is within ANY of the defined ROI polygons. '''
          in_roi = False
          if self.roi_polygons:
            for poly in self.roi_polygons:
              if cv2.pointPolygonTest(poly, (x_center, y_center), False) >= 0:
                in_roi = True
                break
          
          if in_roi: 
            available_track_ids.append(track_id)  # Only track voting for objects IN the ROI
            self.tracker_voting.add_vote(track_id, class_id)  # Add the class prediction to the tracker voting system
            alert_triggered = self.tracker_voting.check_alert(track_id, self.camera_id)  # Check if an alert should be triggered for this track ID
            dump_status = self.tracker_voting.update_dump_status(track_id)
          else: 
            alert_triggered = False
            dump_status = None
          
          self.latest_tracks.append({   # Store the latest track information
            'track_id': track_id,
            'class_id': class_id,
            'conf': conf,
            'bbox': (x1, y1, x2, y2),
            'in_roi': in_roi,
            'alert_triggered': alert_triggered,
            'dump_status': dump_status
          })
        
        # Remove track IDs that are no longer available on screen to prevent memory leak and reuse track IDs for new trucks
        self.tracker_voting.remove_unavailable_tracks(available_track_ids)
        
        # Reset YOLO's internal track ID counter when the screen is completely empty
        # This prevents the ID number from reaching 10000+ over long running times
        if not available_track_ids and len(self.tracker_voting.track_history) == 0:
          BaseTrack.reset_id()
        
    ''' 2. Display the video every DISPLAY_SKIP frames with bounding boxes, labels, and alerts. '''
    if self.frame_count % DISPLAY_SKIP == 0:
      if self.roi_polygons:
        for poly in self.roi_polygons:
          cv2.polylines(frame_resized, [poly], isClosed=True, color=(81, 152, 232), thickness=2)
          # Draw circles at polygon vertices for drag-and-drop visibility
          for point in poly:
            cv2.circle(frame_resized, tuple(point[0]), 5, (0, 0, 255), -1)
      
      # Show total dumps on screen
      total_dumps = self.tracker_voting.total_dumps
      cv2.putText(frame_resized, f"Total Dumps: {total_dumps}", (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

      # Draw bounding boxes, labels and alerts from the latest tracks
      for track in self.latest_tracks:
        tx1, ty1, tx2, ty2 = track['bbox']
        color = CLASS_COLORS.get(track['class_id'], (0, 0, 0))  # Get the color for the class ID, default to black if not found
        class_name = CLASS_LABELS.get(track['class_id'], 'UNKNOWN')
        track_id = track['track_id']
        conf = track['conf']
        label = f"Track: {track_id} | {class_name} {conf:.0%}"
        
        if track['alert_triggered']:
          # cv2.rectangle(frame_resized, (tx1, ty1), (tx2, ty2), color, 2)
          cv2.putText(frame_resized, "CANH BAO: CHUA HA THUNG", (tx1, max(15, ty1 - 35)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
          # Save snapshot after each detected object
          image_folder = os.path.join(SNAPSHOT_DIR, 'detections', date.today().strftime('%Y-%m-%d'))
          if not os.path.exists(image_folder): 
            os.makedirs(image_folder)
          snapshot_path = os.path.join(image_folder, f"{datetime.now().strftime('%H-%M-%S')}_{self.camera_id}_alert.jpg")
          cv2.imwrite(snapshot_path, frame_resized)
          logger.info(f"Saved violation snapshot to {snapshot_path}")
          track['alert_triggered'] = False  # Tránh lưu ảnh 2 lần do DISPLAY_SKIP < PROCESS_SKIP

        # 
        if track.get('dump_status') == 'started_dumping':
          image_folder = os.path.join(SNAPSHOT_DIR, 'dumps', date.today().strftime('%Y-%m-%d'))
          if not os.path.exists(image_folder): 
            os.makedirs(image_folder)
          snapshot_path = os.path.join(image_folder, f"{datetime.now().strftime('%H-%M-%S')}_{self.camera_id}_dumping.jpg")
          cv2.imwrite(snapshot_path, frame_resized)
          logger.info(f"Saved dumping snapshot for Track ID {track_id} to {snapshot_path}")
          track['dump_status'] = None # Tránh lưu nhiều lần

        cv2.rectangle(frame_resized, (tx1, ty1), (tx2, ty2), color, 2)
        cv2.putText(frame_resized, label, (tx1, max(10, ty1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
      cv2.imshow(self.window_name, frame_resized)
    return True
  
  def close(self):
    ''' Stop the camera stream and release resources. '''
    self.reader.stop()
    try:
      cv2.destroyWindow(self.window_name)
    except cv2.error:
      pass
          