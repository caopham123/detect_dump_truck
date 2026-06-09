import cv2
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from config.settings import LOG_DIR
from datetime import datetime
import os
import threading

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
  cameras = [{'id': p.camera_id, 'dumps': p.tracker_voting.total_dumps} for p in pipelines_ref]
  return JSONResponse(content=cameras)

def generate_frames(camera_id: str):
  pipeline = None
  for p in pipelines_ref:
    if p.camera_id == camera_id:
      pipeline = p
      break
      
  if not pipeline:
    return
      
  while True:
    if pipeline.latest_frame_processed is not None:
      ret, buffer = cv2.imencode('.jpg', pipeline.latest_frame_processed)
      if ret:
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    time.sleep(0.05) # ~20fps

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
