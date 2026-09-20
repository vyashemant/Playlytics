import cv2
from typing import List, Dict, Any
from app.utils.helpers import calculate_center
from app.utils.logger import get_logger

logger = get_logger(__name__)

class TrackerManager:
    def __init__(self, target_classes: List[int] = None):
        # 0 is 'person', 32 is 'sports ball' in COCO dataset
        self.target_classes = target_classes or [0, 32] 
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        
    def process_results(self, results, frame_idx: int, timestamp: float) -> List[Dict[str, Any]]:
        """Extract tracking data from YOLO results."""
        frame_data = []
        
        if results.boxes is None or results.boxes.id is None:
            return frame_data
            
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.int().cpu().numpy()
        classes = results.boxes.cls.int().cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        
        for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
            if cls in self.target_classes:
                bbox = box.tolist()
                center = calculate_center(bbox)
                
                track_info = {
                    "track_id": int(track_id),
                    "class_id": int(cls),
                    "frame_number": frame_idx,
                    "timestamp": timestamp,
                    "bbox": bbox,
                    "center": center,
                    "confidence": float(conf)
                }
                
                frame_data.append(track_info)
                self.active_tracks[int(track_id)] = track_info
                
        return frame_data
        
    def annotate_frame(self, frame, frame_data: List[Dict[str, Any]], class_names: Dict[int, str]):
        """Draw bounding boxes and IDs on the frame."""
        annotated_frame = frame.copy()
        
        for data in frame_data:
            x1, y1, x2, y2 = map(int, data["bbox"])
            track_id = data["track_id"]
            conf = data["confidence"]
            cls_id = data["class_id"]
            
            cls_name = class_names.get(cls_id, f"Cls {cls_id}")
            
            # Player label format: Player 7 | 0.91
            if cls_id == 0:
                label_prefix = "Player"
            elif cls_id == 32:
                label_prefix = "Ball"
            else:
                label_prefix = cls_name
                
            label = f"{label_prefix} {track_id} | {conf:.2f}"
            
            # Color based on class (Green for person, Orange for ball)
            color = (0, 255, 0) if cls_id == 0 else (0, 165, 255)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Background for text
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color, -1)
            
            # Text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
        return annotated_frame
