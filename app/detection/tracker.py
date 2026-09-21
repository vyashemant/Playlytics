import cv2
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from app.utils.helpers import calculate_center
from app.utils.logger import get_logger
from app.config.settings import settings

logger = get_logger(__name__)

@dataclass
class PlayerTrack:
    track_id: int
    class_name: str
    class_id: int
    first_seen_frame: int
    last_seen_frame: int
    first_seen_timestamp: float
    last_seen_timestamp: float
    frames_visible: int = 0
    detection_count: int = 0
    _total_confidence: float = 0.0
    
    # Store frame-by-frame data
    frames_data: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def average_confidence(self) -> float:
        return self._total_confidence / max(1, self.detection_count)
        
    def update(self, frame_number: int, timestamp: float, bbox: List[float], center: List[float], confidence: float):
        self.last_seen_frame = frame_number
        self.last_seen_timestamp = timestamp
        self.frames_visible += 1
        self.detection_count += 1
        self._total_confidence += confidence
        
        self.frames_data.append({
            "frame_number": frame_number,
            "timestamp": timestamp,
            "bbox": bbox,
            "center": center,
            "confidence": confidence
        })
        
    def to_summary(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "frames_visible": self.frames_visible,
            "average_confidence": round(self.average_confidence, 4)
        }

class TrackerManager:
    def __init__(self):
        # We will filter by string names defined in settings
        self.target_class_names = settings.TARGET_CLASSES
        self.tracks: Dict[int, PlayerTrack] = {}
        
    def process_results(self, results, frame_idx: int, timestamp: float, class_names: Dict[int, str]) -> List[Dict[str, Any]]:
        """Extract tracking data from YOLO results and update lifecycles."""
        frame_data = []
        
        if results.boxes is None or results.boxes.id is None:
            return frame_data
            
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.int().cpu().numpy()
        classes = results.boxes.cls.int().cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        
        for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
            cls_id = int(cls)
            cls_name = class_names.get(cls_id, f"unknown_{cls_id}")
            
            # Use strict filtering based on target class names if not empty, otherwise allow all
            if self.target_class_names and cls_name not in self.target_class_names:
                continue
                
            track_id = int(track_id)
            bbox = box.tolist()
            center = calculate_center(bbox)
            conf_val = float(conf)
            
            if track_id not in self.tracks:
                self.tracks[track_id] = PlayerTrack(
                    track_id=track_id,
                    class_name=cls_name,
                    class_id=cls_id,
                    first_seen_frame=frame_idx,
                    last_seen_frame=frame_idx,
                    first_seen_timestamp=timestamp,
                    last_seen_timestamp=timestamp
                )
                
            self.tracks[track_id].update(frame_idx, timestamp, bbox, center, conf_val)
            
            track_info = {
                "track_id": track_id,
                "class_id": cls_id,
                "class_name": cls_name,
                "frame_number": frame_idx,
                "timestamp": timestamp,
                "bbox": bbox,
                "center": center,
                "confidence": conf_val
            }
            frame_data.append(track_info)
                
        return frame_data
        
    def get_track_summaries(self) -> List[Dict[str, Any]]:
        return [track.to_summary() for track in self.tracks.values()]
        
    def get_all_tracking_data(self) -> List[Dict[str, Any]]:
        tracks_out = []
        for track in self.tracks.values():
            tracks_out.append({
                "track_id": track.track_id,
                "class_name": track.class_name,
                "frames": track.frames_data
            })
        return tracks_out
        
    def annotate_frame(self, frame, frame_data: List[Dict[str, Any]], fps: float, total_frames: int):
        """Draw bounding boxes, IDs, and stats on the frame."""
        annotated_frame = frame.copy()
        
        # Display general stats
        if len(frame_data) > 0:
            frame_idx = frame_data[0]["frame_number"]
        else:
            frame_idx = 0
            
        stats_text = [
            f"Frame: {frame_idx}/{total_frames}",
            f"FPS: {int(fps)}",
            f"Players: {len([d for d in frame_data if d.get('class_name') in ['person', 'player']])}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(annotated_frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            y_offset += 35
        
        for data in frame_data:
            x1, y1, x2, y2 = map(int, data["bbox"])
            track_id = data["track_id"]
            conf = data["confidence"]
            cls_name = data.get("class_name", "unknown")
            
            if cls_name in ["person", "player"]:
                label_prefix = "ID:"
                color = (0, 255, 0)
            elif cls_name in ["sports ball", "cricket_ball"]:
                label_prefix = "Ball:"
                color = (0, 165, 255)
            elif cls_name in ["cricket_bat"]:
                label_prefix = "Bat:"
                color = (255, 0, 0)
            else:
                label_prefix = f"{cls_name}:"
                color = (255, 255, 255)
                
            label = f"{label_prefix} {track_id} | Conf: {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Background for text
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color, -1)
            
            # Text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
        return annotated_frame

