from camera_stream import CameraStream
import cv2

stream = CameraStream("cam_01", "rtsp://admin:longson2016@192.168.10.149:554/Streaming/Channels/1902").start()

while True:
  ret, frame = stream.read()

  if ret:
    cv2.imshow("Camera", frame)
    # print(f"Frame shape: {frame.shape}")

  key = cv2.waitKey(1) & 0xFF

  if key == 27:  # ESC
    break

stream.stop()
cv2.destroyAllWindows()