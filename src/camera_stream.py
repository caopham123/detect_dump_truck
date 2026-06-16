import cv2, threading, time
from datetime import datetime

class CameraStream:
  def __init__(self, camera_id, rtsp_url):
    self.camera_id = camera_id
    self.rtsp_url = rtsp_url
    self.frame = None         # Store the latest frame read from the camera stream
    self.started = False      # Flag to control the thread running state
    self.read_lock = threading.Lock()   # Lock synchronous to allow 1 thread to read frames and another thread to write the latest frame without conflicts
    self.last_time_no_frame = time.time()  # Time when no frame was received
    self.is_healthy = False   # Set to True only after a successful initial probe
    self.cap = None           # Will be opened lazily after probe()

  def probe(self, timeout_sec=5):
    '''
    Try to read the first frame within `timeout_sec` seconds.
    Return True if successful, False if failed (URL error, connection lost...).
    Must be called before start() to avoid blocking the entire pipeline.
    '''
    if not self.rtsp_url:
      print(f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')} [WARN]  Camera {self.camera_id}: rtsp_url chưa được cấu hình, bỏ qua.")
      return False

    print(f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')} [INFO]  Camera {self.camera_id}: Đang kiểm tra kết nối ({self.rtsp_url})...")
    cap = cv2.VideoCapture(self.rtsp_url)
    deadline = time.time() + timeout_sec
    success = False
    while time.time() < deadline:
      ret, frame = cap.read()
      if ret and frame is not None:
        success = True
        break
      time.sleep(0.2)
    cap.release()

    if success:
      print(f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')} [INFO]  Camera {self.camera_id}: Kết nối thành công.")
      self.is_healthy = True
    else:
      print(f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')} [ERROR]  Camera {self.camera_id}: Không thể kết nối sau {timeout_sec}s, bỏ qua.")
    return success

  def start(self):
    ''' Start each camera stream in a separate thread. '''
    if self.started:
      print(f"Camera {self.camera_id} already started.")
      return self           # Avoid creating multiple threads for the same camera

    self.cap = cv2.VideoCapture(self.rtsp_url)  # Mở stream thật sự sau khi probe thành công
    self.started = True     # Set the stated flag to True before starting the thread
    self.thread = threading.Thread(target=self._update, args=(), daemon=True)   # Create a thread to read frames from camera
    self.thread.start()     # Run the thread
    return self             # Return CameraStream(arg1, arg2).start() to allow chaining.
  
  def _update(self):
    ''' Continuously read frames from the camera stream. '''
    ''' Use self.read_lock to synchronize access to the latest frame. '''
    while self.started:
      if self.cap is None:
        time.sleep(1)
        continue
      ret, frame = self.cap.read()
      if not ret or frame is None:
        if time.time() - self.last_time_no_frame > 60:
          print(f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')} [ERROR]  Camera {self.camera_id} stream ended or cannot be read.")
          self.last_time_no_frame = time.time()
          if self.cap.isOpened():
            self.cap.release()
          self.cap = cv2.VideoCapture(self.rtsp_url)         # Reopen the stream if it cannot read frames
          time.sleep(5)                       # Wait 5s before retrying to avoid busy loop
        continue
      with self.read_lock:
        self.frame = frame.copy()    # Update the latest frame read from the camera stream
      time.sleep(0.02)             # Sleep briefly to reduce CPU usage
    
  def read(self):
    ''' Get the latest frame read from the camera stream. 
    When _update() is running, this method is blocked. Avoid deadlock.'''
    with self.read_lock:
      if self.frame is not None:
        return True, self.frame.copy()   # Return a copy of the latest frame to avoid thread safety issues
      return False, None
    
  def stop(self):
    ''' Stop the camera stream and release resources. '''
    self.started = False
    # Cho phép background thread tự gọi release() để tránh lỗi thread safety của OpenCV
    if hasattr(self, 'thread') and self.thread.is_alive():
      self.thread.join()
    if self.cap is not None and self.cap.isOpened():
      self.cap.release()