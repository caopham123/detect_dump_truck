import os
from datetime import datetime
from collections import defaultdict, deque
from config.settings import VOTE_COUNT, VOTE_THRESHOLD, LOG_DIR, TRACK_BUFFER
from utils.logger import logger

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
    self.missing_tracks = defaultdict(int)  # Count the number of missing frames for each track ID
    
  def add_vote(self, track_id, class_id):
    ''' Add a new class prediction for a given track ID. '''
    self.track_history[track_id].append(class_id)
    # Clear the missing counter if the track is detected
    if track_id in self.missing_tracks:
      self.missing_tracks[track_id] = 0


  def check_alert(self, track_id, camera_id):
    ''' Check if the track ID has enough votes to trigger an alert for "truck_raised". '''
    votes = self.track_history[track_id]  # Get the history of class predictions for the track ID
    if len(votes) < VOTE_COUNT:
      return False  # Not enough votes yet to make a decision
    
    truck_raised_count = votes.count(0)  # Count how many times the class "truck_raised" (class_id=0) was predicted
    if truck_raised_count >= VOTE_THRESHOLD and track_id not in self.alerted_ids:
      self.alerted_ids.add(track_id)  # Mark this track ID as having triggered an alert
      msg = f"ALERT: CAMERA {camera_id} | Track ID {track_id} | 'CHUA HA' | Rate: {truck_raised_count}/{VOTE_COUNT} votes."
      logger.warning(msg)
      self.alerted_ids.add(track_id)
      return True
    return False
  
  def remove_unavailable_tracks(self, available_ids):
    ''' Remove track IDs that are no longer available on screen.
      available_ids: A set of track IDs that are currently detected on screen.
      unavailable_ids: A set of track IDs that are in track_history but no longer detected.
    '''
    unavailable_ids = set(self.track_history.keys()) - set(available_ids)  # Find track IDs that are still not available
    for track_id in unavailable_ids:
      self.missing_tracks[track_id] += 1  # Increment the missing counter if the track is not detected
      if self.missing_tracks[track_id] >= TRACK_BUFFER:  # If the track has been missing for too long
        del self.track_history[track_id]  # Remove the history of the unavailable track ID
        if track_id in self.alerted_ids:
          self.alerted_ids.discard(track_id)  # Remove from alerted IDs if it was previously alerted