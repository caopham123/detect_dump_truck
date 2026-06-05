import logging
import os
import glob
from config.settings import LOG_DIR
from datetime import datetime, timedelta


class DailyFileHandler(logging.FileHandler):
  def __init__(self, backup_count=30):
    self.log_dir = LOG_DIR
    self.backup_count = backup_count
    self.current_date = datetime.now().strftime("%Y-%m-%d")

    # Ensure log directory exists
    os.makedirs(self.log_dir, exist_ok=True)

    filename = self.log_dir / f"{self.current_date}_log.txt"

    super().__init__(
      filename=filename,
      mode="a",     # append, don't overwrite
      encoding="utf-8"
    )
    self._cleanup_old_logs()

  def emit(self, record):
    today = datetime.now().strftime("%Y-%m-%d")
    if today != self.current_date:
      self.current_date = today
      self.close()
      self.baseFilename = str(
        self.log_dir / f"{today}_log.txt"
      )
      self.stream = self._open()
      self._cleanup_old_logs() # Clean up when day changes
    super().emit(record)

  def _cleanup_old_logs(self):
    if self.backup_count <= 0: return
    
    log_pattern = str(self.log_dir / "*_log.txt")
    files = glob.glob(log_pattern)
    
    cutoff_date = datetime.now() - timedelta(days=self.backup_count)
    cutoff_str = cutoff_date.strftime("%Y-%m-%d")
    
    for file_path in files:
      filename = os.path.basename(file_path)
      # Extract YYYY-MM-DD from YYYY-MM-DD_log.txt
      date_str = filename.split("_log.txt")[0]
      
      # String comparison works perfectly for YYYY-MM-DD
      if date_str < cutoff_str:
        try:
          os.remove(file_path)
        except OSError:
          pass