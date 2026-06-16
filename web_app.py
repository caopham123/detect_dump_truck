import cv2
import numpy as np
import time
import uvicorn
import os
import threading
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from config.settings import LOG_DIR, CAMERAS
from datetime import datetime

app = FastAPI()

# Mount the static directory for CSS/JS
app.mount("/static_files", StaticFiles(directory="static"), name="static")

pipelines_ref = []

@app.get("/")
async def index():
  return FileResponse('static/index.html')

# Also serve CSS and JS directly if requested from root
@app.get("/style.css")
async def style():
  return FileResponse('static/style.css')

@app.get("/script.js")
async def script():
  return FileResponse('static/script.js')

@app.get("/api/cameras")
async def get_cameras():
  cameras = []
  for cam_config in CAMERAS:
    cam_id = cam_config.get("camera_id", "Unknown")
    pipeline = next((p for p in pipelines_ref if p.camera_id == cam_id), None)

    cameras.append({
      "id": cam_id,
      "status": 'active' if pipeline else 'offline',
      "dumps": (
        pipeline.tracker_voting.total_dumps if pipeline else 0
      )
    })
  return JSONResponse(content=cameras)

def _make_offline_frame():
  """Tạo JPEG placeholder khi camera offline."""
  img = np.zeros((360, 480, 3), dtype=np.uint8)
  # Draw background within grid
  for y in range(0, 360, 20):
    cv2.line(img, (0, y), (480, y), (20, 20, 20), 1)
  for x in range(0, 480, 20):
    cv2.line(img, (x, 0), (x, 360), (20, 20, 20), 1)
  # Icon
  cv2.rectangle(img, (200, 140), (280, 220), (60, 60, 60), 2)
  cv2.line(img, (200, 140), (280, 220), (80, 80, 80), 2)
  cv2.line(img, (280, 140), (200, 220), (80, 80, 80), 2)
  # Text
  cv2.putText(img, 'SIGNAL LOST', (145, 260),
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)
  cv2.putText(img, 'Camera Disconnected', (125, 290),
              cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)
  ret, buffer = cv2.imencode('.jpg', img)
  return buffer.tobytes() if ret else b''

def generate_frames(camera_id: str):
  offline_frame = _make_offline_frame()

  while True:
    pipeline = next((p for p in pipelines_ref if p.camera_id == camera_id), None)

    if pipeline and pipeline.latest_frame_processed is not None:
      ret, buffer = cv2.imencode('.jpg', pipeline.latest_frame_processed)
      if ret:
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)  # ~20fps
        continue

    # Camera offline or without frames, serve placeholder
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + offline_frame + b'\r\n')
    time.sleep(1.0)

@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
  return StreamingResponse(generate_frames(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/logs")
async def get_logs():
  current_date = datetime.now().strftime("%Y-%m-%d")
  log_file = os.path.join(LOG_DIR, f"{current_date}_log.txt")
  logs = []
  if os.path.exists(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
      # Read last 100 lines
      lines = f.readlines()
      logs = lines[-100:]
      # Reverse to show newest first
      logs.reverse()
  return JSONResponse(content={'logs': logs})

def start_web_app(pipelines):
  global pipelines_ref
  pipelines_ref = pipelines
  
  # Run uvicorn server in a separate daemon thread
  def run_server():
    # log_level="error" to avoid clogging the console with web access logs
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="error")
      
  t = threading.Thread(target=run_server)
  t.daemon = True
  t.start()
