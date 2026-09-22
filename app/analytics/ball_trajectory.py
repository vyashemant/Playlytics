from typing import List, Dict, Any
import math
from app.config.settings import settings

class BallTrajectoryAnalyzer:
    def __init__(self):
        self.smoothing_enabled = settings.BALL_SMOOTHING_ENABLED
        self.alpha = settings.BALL_SMOOTHING_ALPHA

    def analyze(self, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trajectory:
            return self._empty_trajectory()

        points = []
        total_distance = 0.0
        velocities = []
        accelerations = []
        
        smoothed_pos = trajectory[0]["center"]
        prev_pos = smoothed_pos
        prev_velocity = 0.0
        
        first_frame = trajectory[0]
        last_frame = trajectory[-1]
        
        displacement = math.sqrt(
            (last_frame["center"][0] - first_frame["center"][0])**2 + 
            (last_frame["center"][1] - first_frame["center"][1])**2
        )
        
        for i, det in enumerate(trajectory):
            raw_pos = det["center"]
            time_diff = 0.0
            
            if i == 0:
                smoothed_pos = raw_pos
            else:
                time_diff = det["timestamp"] - trajectory[i-1]["timestamp"]
                
                if self.smoothing_enabled:
                    smoothed_pos = [
                        self.alpha * raw_pos[0] + (1 - self.alpha) * prev_pos[0],
                        self.alpha * raw_pos[1] + (1 - self.alpha) * prev_pos[1]
                    ]
                else:
                    smoothed_pos = raw_pos
                    
                dx = smoothed_pos[0] - prev_pos[0]
                dy = smoothed_pos[1] - prev_pos[1]
                dist = math.sqrt(dx**2 + dy**2)
                total_distance += dist
                
                if time_diff > 0:
                    velocity = dist / time_diff
                else:
                    velocity = 0.0
                    
                velocities.append(velocity)
                
                if i > 1:
                    if time_diff > 0:
                        acceleration = (velocity - prev_velocity) / time_diff
                    else:
                        acceleration = 0.0
                    accelerations.append(acceleration)
                
                prev_velocity = velocity

            points.append({
                "frame_number": det["frame_number"],
                "timestamp": det["timestamp"],
                "raw_center": raw_pos,
                "smoothed_center": [round(smoothed_pos[0], 2), round(smoothed_pos[1], 2)],
                "confidence": det["confidence"]
            })
            
            prev_pos = smoothed_pos
            
        avg_velocity = sum(velocities) / len(velocities) if velocities else 0.0
        max_velocity = max(velocities) if velocities else 0.0
        
        avg_accel = sum(accelerations) / len(accelerations) if accelerations else 0.0
        max_accel = max(accelerations) if accelerations else 0.0

        return {
            "total_distance_pixels": round(total_distance, 2),
            "displacement_pixels": round(displacement, 2),
            "average_velocity_pixels_per_second": round(avg_velocity, 2),
            "maximum_velocity_pixels_per_second": round(max_velocity, 2),
            "average_acceleration_pixels_per_second_squared": round(avg_accel, 2),
            "maximum_acceleration_pixels_per_second_squared": round(max_accel, 2),
            "points": points
        }
        
    def _empty_trajectory(self) -> Dict[str, Any]:
        return {
            "total_distance_pixels": 0.0,
            "displacement_pixels": 0.0,
            "average_velocity_pixels_per_second": 0.0,
            "maximum_velocity_pixels_per_second": 0.0,
            "average_acceleration_pixels_per_second_squared": 0.0,
            "maximum_acceleration_pixels_per_second_squared": 0.0,
            "points": []
        }
