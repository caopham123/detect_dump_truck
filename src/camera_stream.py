import cv2, threading, time
from datetime import datetime

class CameraStream:
  def __init__(self, camera_id, rtsp_url):
    self.camera_id = camera_id
    self.rtsp_url = rtsp_url
    self.cap = cv2.VideoCapture(rtsp_url)
    self.frame = None         # Store the latest frame read from the camera stream
    self.started = False      # Flag to control the thread running state
    self.read_lock = threading.Lock()   # Lock synchronous to allow 1 thread to read frames and another thread to write the latest frame without conflicts
    
  def start(self):
    ''' Start each camera stream in a separate thread. '''
    if self.started:
      print(f"Camera {self.camera_id} already started.")
      return self           # Avoid creating multiple threads for the same camera
    
    self.started = True     # Set the stated flag to True before starting the thread
    self.thread = threading.Thread(target=self._update, args=(), daemon=True)   # Create a thread to read frames from camere
    self.thread.start()     # Run the thread
    return self             # Return CameraStream(arg1, arg2).start() to allow chaining.
  
  def _update(self):
    ''' Continuously read frames from the camera stream. '''
    ''' Use self.read_lock to synchronize access to the latest frame. '''
    while self.started:
      ret, frame = self.cap.read()
      if not ret:
        print(f"[ERROR] {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')} Camera {self.camera_id} stream ended or cannot be read.")
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
    if self.cap.isOpened():
      self.cap.release()