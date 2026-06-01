''' Tracker with Voting '''

from collections import defaultdict, deque
import logging, os
from config.settings import VOTE_COUNT, VOTE_THRESHOLD, LOG_FILES

os.makedirs(LOG_FILES, exist_ok=True)   # Ensure the logs directory exists
logging.basicConfig(
  filename=LOG_FILES,     # Save logs to a file with timestamp in the name
  level=logging.INFO,     # Log INFO and above (including ERROR) to the file
  format="%(asctime)s | [%(levelname)s] | %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S"
)

class TrackerVoting:
  def __init__(self):
    ''' Initialize the tracker voting system. 
      track_history: A dictionary that maps track IDs to a deque of their recent class predictions (class_id).
      Example: {
        15: deque(["raised","raised","raised"]),
        22: deque(["lowered","lowered"])
      }
    '''
    self.track_history = defaultdict(lambda: deque(maxlen=VOTE_COUNT))   # Dictionary to store the history of class predictions
    self.alerted_ids = set()  # Set to track which IDs have already triggered an alert
  
  def add_vote(self, track_id, class_id):
    ''' Add a new class prediction for a given track ID. '''
    self.track_history[track_id].append(class_id)   # Add the new class prediction to the history of the track ID
    
  def check_alert(self, track_id):
    ''' Check if the track ID has enough votes to trigger an alert for "truck_raised". '''
    votes = self.track_history[track_id]  # Get the history of class predictions for the track ID
    if len(votes) < VOTE_COUNT:
      return None  # Not enough votes yet to make a decision
    
    truck_raised_count = votes.count(0)  # Count how many times the class "truck_raised" (class_id=0) was predicted
    if truck_raised_count >= VOTE_THRESHOLD and track_id not in self.alerted_ids:
      msg = f"ALERT: Track ID {track_id} | 'CHUA HA' | {truck_raised_count}/{VOTE_COUNT} votes."
      print(msg)
      logging.warning(msg)
      self.alerted_ids.add(track_id)
      return True
    return False
  
  def remove_unavailable_tracks(self, available_ids):
    ''' Remove track IDs that are no longer available on screen.
      available_ids: A set of track IDs that are currently detected on screen.
      unavailable_ids: A set of track IDs that are in track_history but no longer detected.
      Should Delete unvailable track IDs to prevent memory leak and reuse track IDs for new trucks,
      and remove track_id that was alerted.
    '''
    unavailable_ids = set(self.track_history.keys()) - set(available_ids)  # Find track IDs that are still not available
    for track_id in unavailable_ids:
      del self.track_history[track_id]  # Remove the history of the unavailable track ID
      if track_id in self.alerted_ids:
        self.alerted_ids.discard(track_id)  # Remove from alerted IDs if it was previously alerted