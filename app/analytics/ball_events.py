from typing import List, Dict, Any
from app.config.settings import settings

class BallEventDetector:
    def __init__(self):
        self.min_vertical_displacement = settings.BALL_BOUNCE_MIN_VERTICAL_DISPLACEMENT
        self.min_time_seconds = settings.BALL_BOUNCE_MIN_TIME_SECONDS
        
    def detect_events(self, trajectory_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        events = []
        points = trajectory_data.get("points", [])
        
        if not points:
            return events
            
        # Trajectory start
        events.append({
            "event_type": "trajectory_start",
            "frame_number": points[0]["frame_number"],
            "timestamp": points[0]["timestamp"],
            "position": points[0]["smoothed_center"],
            "confidence": points[0]["confidence"],
            "evidence": "First observed point in continuous trajectory"
        })
        
        # Detect bounce candidate
        # A simple bounce model: y coordinate increases (ball goes down in image space, y points down)
        # then decreases (ball goes up), with enough displacement and time separation
        last_bounce_time = -self.min_time_seconds
        
        for i in range(1, len(points) - 1):
            prev_pt = points[i-1]
            curr_pt = points[i]
            next_pt = points[i+1]
            
            y_prev = prev_pt["smoothed_center"][1]
            y_curr = curr_pt["smoothed_center"][1]
            y_next = next_pt["smoothed_center"][1]
            
            # Local maximum in y means ball reached lowest point in image (going down then up)
            if y_curr > y_prev and y_curr > y_next:
                # Check displacement evidence (was it moving down significantly?)
                # Look back a few frames to see vertical drop
                lookback = max(0, i - 5)
                y_drop = y_curr - points[lookback]["smoothed_center"][1]
                
                # Look ahead to see vertical rise
                lookahead = min(len(points) - 1, i + 5)
                y_rise = y_curr - points[lookahead]["smoothed_center"][1]
                
                if y_drop > self.min_vertical_displacement and y_rise > self.min_vertical_displacement:
                    if curr_pt["timestamp"] - last_bounce_time >= self.min_time_seconds:
                        events.append({
                            "event_type": "bounce_candidate",
                            "frame_number": curr_pt["frame_number"],
                            "timestamp": curr_pt["timestamp"],
                            "position": curr_pt["smoothed_center"],
                            "confidence": curr_pt["confidence"],
                            "evidence": f"Local y-maximum with drop={y_drop:.1f} and rise={y_rise:.1f} pixels"
                        })
                        last_bounce_time = curr_pt["timestamp"]
                        
        # Direction change candidate (horizontal)
        # Look for local extrema in x coordinate as well
        last_dir_change_time = -self.min_time_seconds
        
        for i in range(1, len(points) - 1):
            prev_pt = points[i-1]
            curr_pt = points[i]
            next_pt = points[i+1]
            
            x_prev = prev_pt["smoothed_center"][0]
            x_curr = curr_pt["smoothed_center"][0]
            x_next = next_pt["smoothed_center"][0]
            
            # Local extremum in x
            if (x_curr > x_prev and x_curr > x_next) or (x_curr < x_prev and x_curr < x_next):
                lookback = max(0, i - 5)
                lookahead = min(len(points) - 1, i + 5)
                
                x_diff1 = abs(x_curr - points[lookback]["smoothed_center"][0])
                x_diff2 = abs(x_curr - points[lookahead]["smoothed_center"][0])
                
                if x_diff1 > self.min_vertical_displacement and x_diff2 > self.min_vertical_displacement:
                    if curr_pt["timestamp"] - last_dir_change_time >= self.min_time_seconds:
                        # Avoid duplicating if it's the exact same frame as a bounce
                        if not any(e["frame_number"] == curr_pt["frame_number"] and e["event_type"] == "bounce_candidate" for e in events):
                            events.append({
                                "event_type": "direction_change_candidate",
                                "frame_number": curr_pt["frame_number"],
                                "timestamp": curr_pt["timestamp"],
                                "position": curr_pt["smoothed_center"],
                                "confidence": curr_pt["confidence"],
                                "evidence": f"Local x-extremum with diffs {x_diff1:.1f}, {x_diff2:.1f} pixels"
                            })
                            last_dir_change_time = curr_pt["timestamp"]
                        
        # Trajectory end
        events.append({
            "event_type": "trajectory_end",
            "frame_number": points[-1]["frame_number"],
            "timestamp": points[-1]["timestamp"],
            "position": points[-1]["smoothed_center"],
            "confidence": points[-1]["confidence"],
            "evidence": "Last observed point in continuous trajectory"
        })
        
        # Sort events by frame number
        events.sort(key=lambda e: e["frame_number"])
        
        return events
