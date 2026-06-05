import logging
import sys
from utils.daily_file_handle import DailyFileHandler

# Create a custom logger
logger = logging.getLogger("dump_truck_detector")
logger.setLevel(logging.INFO) # Record INFO and above (including ERROR)
logger.propagate = False  # Prevent log duplication when YOLO configures root logger

# 1. Console handler
c_handler = logging.StreamHandler(sys.stdout)
c_handler.setLevel(logging.INFO)

# 2. File handler (Rotates daily at midnight, keeps 30 days of logs)
f_handler = DailyFileHandler()
f_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
log_format = logging.Formatter("%(asctime)s | [%(levelname)s] | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
c_handler.setFormatter(log_format)
f_handler.setFormatter(log_format)

# Add handlers to the logger
# Check to avoid adding handlers multiple times if imported multiple times
if not logger.hasHandlers():
  logger.addHandler(c_handler)
  logger.addHandler(f_handler)
