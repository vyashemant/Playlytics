from typing import List, Dict, Any, Optional
import math
from app.utils.logger import get_logger
from app.config.settings import settings

logger = get_logger(__name__)

class BallTracker:
    def __init__(self):
        self.max_missed_frames = settings.BALL_MAX_MISSED_FRAMES
        self.active_trajectory = []
        self.best_trajectory = []
        self.current_missed_frames = 0
        self.track_id_counter = 1
        self.current_track_id = None
        
        self.tracking_status = "no_reliable_ball_detected"
        
    def update(self, detections: List[Dict[str, Any]], frame_number: int):
        """Update tracker with new detections."""
        if not detections:
            if self.active_trajectory:
                self.current_missed_frames += 1
                if self.current_missed_frames > self.max_missed_frames:
                    self._terminate_trajectory()
            return

        # If multiple detections, pick the highest confidence one.
        # In a real system, we'd use distance to last known position to prevent jumping.
        
        # Simple distance heuristic if we have a previous point
        if self.active_trajectory:
            last_pos = self.active_trajectory[-1]["center"]
            def score(d):
                dist = math.sqrt((d["center"][0] - last_pos[0])**2 + (d["center"][1] - last_pos[1])**2)
                # Maximize confidence, minimize distance
                return d["confidence"] - (dist / 1000.0) 
            best_detection = max(detections, key=score)
        else:
            best_detection = max(detections, key=lambda d: d['confidence'])
        
        if not self.active_trajectory:
            self.current_track_id = self.track_id_counter
            self.track_id_counter += 1
            self.current_missed_frames = 0
            
        best_detection['track_id'] = self.current_track_id
        self.active_trajectory.append(best_detection)
        self.current_missed_frames = 0
        self.tracking_status = "tracking"
        
    def _terminate_trajectory(self):
        """Terminate the current trajectory if max gap is exceeded."""
        if len(self.active_trajectory) > len(self.best_trajectory):
            self.best_trajectory = list(self.active_trajectory)
            
        self.active_trajectory = []
        self.current_track_id = None
        self.current_missed_frames = 0
        
    def finalize(self):
        """Finalize tracking at the end of the video."""
        self._terminate_trajectory()
        if not self.best_trajectory:
            self.tracking_status = "no_detection"
        else:
            frames_observed = len(self.best_trajectory)
            if frames_observed < 10:
                self.tracking_status = "insufficient_trajectory"
            else:
                first = self.best_trajectory[0]["center"]
                last = self.best_trajectory[-1]["center"]
                displacement = math.sqrt((last[0]-first[0])**2 + (last[1]-first[1])**2)
                
                total_distance = 0.0
                for i in range(1, frames_observed):
                    p1 = self.best_trajectory[i-1]["center"]
                    p2 = self.best_trajectory[i]["center"]
                    total_distance += math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                
                avg_confidence = sum(d["confidence"] for d in self.best_trajectory) / frames_observed
                
                if displacement < 50.0 or total_distance < 100.0 or avg_confidence < 0.2:
                    self.tracking_status = "unreliable_candidate"
                else:
                    self.tracking_status = "reliable_trajectory"
        
    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Return the best found trajectory."""
        return self.best_trajectory
