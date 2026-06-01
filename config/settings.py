from dotenv import load_dotenv
import os
import yaml
from pathlib import Path
from datetime import datetime

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent     # Go to folder detect_dump_truck
YAML_PATH = BASE_DIR / "config" / "constants.yaml"

# --------- Default settings (can be overridden by YAML config) --------
# Model and inference settings
MODEL_PATH = BASE_DIR / "model_loader" / "poc_v7_best.pt"
CONF_THRESHOLD = 0.5

# RTSP stream
DISPLAY_SKIP = 5
PROCESS_SKIP = 10
VOTE_COUNT = 5
INFERENCE_RESIZE = tuple(map(int, os.getenv('INFERENCE_RESIZE', '640,480').split(',')))
CAMERAS = []
LOG_FILES = BASE_DIR / "logs" / f"log_{datetime.now().strftime('%Y_%m_%d_log')}.txt"

# Setup color and label for classes
CLASS_COLORS = {
  0: (0, 0, 255),    # truck_raised  → ĐỎ (BGR)
  1: (0, 200, 0),    # truck_lowered → XANH (BGR)
}
CLASS_LABELS = {
  0: "CHUA HA",
  1: "DA HA",
}

# READ YAML CONFIG
if YAML_PATH.exists():
  try:
    with open(YAML_PATH, 'r') as f:
      content = yaml.safe_load(f)
      
      if 'model' in content:        # Read the 'model' key in YAML
        MODEL_PATH = content['model'].get('path', MODEL_PATH)     # If the 'path' key not exists, keep the existing MODEL_PATH from .env
        CONF_THRESHOLD = content['model'].get('conf_threshold', CONF_THRESHOLD)
      
      if 'processing' in content:   # Read the 'processing' key in YAML
        DISPLAY_SKIP = content['processing'].get('display_skip', DISPLAY_SKIP)
        PROCESS_SKIP = content['processing'].get('process_skip', PROCESS_SKIP)
      
      if 'voting' in content:      # Read the 'voting' key in YAML
        VOTE_COUNT = content['voting'].get('vote_count', VOTE_COUNT)
        VOTE_THRESHOLD = int(VOTE_COUNT * 0.75)  # Default threshold for majority vote (75% of VOTE_COUNT)
      
      if 'class_labels' in content:
        CLASS_LABELS = {int(k): v for k, v in content['class_labels'].items()}
      
      if 'class_colors' in content:
        CLASS_COLORS = {}
        for k, v in content['class_colors'].items():
          if isinstance(k, int) and len(v) == 3:
            CLASS_COLORS[int(k)] = tuple(reversed(v))  # Convert RGB to BGR
          else: CLASS_COLORS[int(k)] = (0, 255, 255)  # Default color if format is incorrect
          
      if 'cameras' in content:     # Read the 'cameras' list in YAML
        CAMERAS = content.get('cameras', [])
  except Exception as e:
    print(f"[WARNING] Error reading YAML config in {YAML_PATH}: {e}. Using defaults from .env and hardcoded values.")
else: print(f"[WARNING] YAML config file not found at {YAML_PATH}. Using defaults from .env and hardcoded values.")