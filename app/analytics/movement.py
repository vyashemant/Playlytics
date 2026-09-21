import math
from typing import List, Dict, Any, Iterable
from app.config.settings import settings
from app.detection.tracker import PlayerTrack

class MovementAnalyzer:
    def __init__(self, fps: float, width: int, height: int):
        self.fps = fps
        self.width = width
        self.height = height
        self.threshold = settings.MOVEMENT_THRESHOLD_PIXELS
        
    def analyze(self, tracks: Iterable[PlayerTrack]) -> Dict[str, Any]:
        """Analyze all tracks and return structured movement analysis."""
        players_data = []
        
        for track in tracks:
            # We skip tracks that have 0 frames.
            if not track.frames_data:
                continue
                
            metrics, timeline = self._analyze_track(track)
            
            players_data.append({
                "track_id": track.track_id,
                "metrics": metrics,
                "timeline": timeline
            })
            
        return {
            "video": {
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
                # Assuming timeline starts at 0, calculate duration based on last seen frame
                "duration_seconds": round(max((t.last_seen_frame for t in tracks), default=0) / max(self.fps, 1), 2)
            },
            "players": players_data
        }
        
    def _analyze_track(self, track: PlayerTrack):
        frames = track.frames_data
        if len(frames) == 0:
            return self._empty_metrics(), []
            
        first_frame = frames[0]
        last_frame = frames[-1]
        
        # Calculate Displacement
        first_x, first_y = first_frame["center"]
        last_x, last_y = last_frame["center"]
        displacement = math.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)
        
        total_distance = 0.0
        moving_frames = 0
        stationary_frames = 1 # First frame counts as stationary
        
        velocities = []
        accelerations = []
        
        timeline = []
        
        # Initial timeline entry for the first frame
        timeline.append({
            "frame_number": first_frame["frame_number"],
            "timestamp": first_frame["timestamp"],
            "position": first_frame["center"],
            "distance_from_previous_pixels": 0.0,
            "velocity_pixels_per_second": 0.0,
            "direction_degrees": 0.0,
            "state": "stationary"
        })
        
        prev_velocity = 0.0
        
        for i in range(1, len(frames)):
            curr = frames[i]
            prev = frames[i-1]
            
            curr_x, curr_y = curr["center"]
            prev_x, prev_y = prev["center"]
            
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            
            dist = math.sqrt(dx**2 + dy**2)
            total_distance += dist
            
            time_diff = curr["timestamp"] - prev["timestamp"]
            
            # Prevent division by zero
            if time_diff > 0:
                velocity = dist / time_diff
            else:
                velocity = 0.0
                
            velocities.append(velocity)
            
            if i == 1:
                acceleration = 0.0
            else:
                if time_diff > 0:
                    acceleration = (velocity - prev_velocity) / time_diff
                else:
                    acceleration = 0.0
            accelerations.append(acceleration)
            
            direction = math.degrees(math.atan2(dy, dx))
            
            state = "moving" if dist >= self.threshold else "stationary"
            if state == "moving":
                moving_frames += 1
            else:
                stationary_frames += 1
                
            timeline.append({
                "frame_number": curr["frame_number"],
                "timestamp": curr["timestamp"],
                "position": curr["center"],
                "distance_from_previous_pixels": round(dist, 2),
                "velocity_pixels_per_second": round(velocity, 2),
                "direction_degrees": round(direction, 2),
                "state": state
            })
            
            prev_velocity = velocity
            
        total_observations = moving_frames + stationary_frames
        if total_observations > 0:
            moving_pct = (moving_frames / total_observations) * 100.0
            stationary_pct = (stationary_frames / total_observations) * 100.0
        else:
            moving_pct = 0.0
            stationary_pct = 0.0
            
        avg_velocity = sum(velocities) / len(velocities) if velocities else 0.0
        max_velocity = max(velocities) if velocities else 0.0
        
        avg_accel = sum(accelerations) / len(accelerations) if accelerations else 0.0
        max_accel = max(accelerations) if accelerations else 0.0
        
        metrics = {
            "total_distance_pixels": round(total_distance, 2),
            "displacement_pixels": round(displacement, 2),
            "average_velocity_pixels_per_second": round(avg_velocity, 2),
            "maximum_velocity_pixels_per_second": round(max_velocity, 2),
            "average_acceleration_pixels_per_second_squared": round(avg_accel, 2),
            "maximum_acceleration_pixels_per_second_squared": round(max_accel, 2),
            "moving_frames": moving_frames,
            "stationary_frames": stationary_frames,
            "movement_percentage": round(moving_pct, 2),
            "stationary_percentage": round(stationary_pct, 2)
        }
        
        return metrics, timeline

    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            "total_distance_pixels": 0.0,
            "displacement_pixels": 0.0,
            "average_velocity_pixels_per_second": 0.0,
            "maximum_velocity_pixels_per_second": 0.0,
            "average_acceleration_pixels_per_second_squared": 0.0,
            "maximum_acceleration_pixels_per_second_squared": 0.0,
            "moving_frames": 0,
            "stationary_frames": 0,
            "movement_percentage": 0.0,
            "stationary_percentage": 0.0
        }
