from typing import List, Dict, Any, Optional
from app.utils.logger import get_logger
from app.config.settings import settings
from app.utils.helpers import calculate_center
import os

logger = get_logger(__name__)

class BallDetector:
    def __init__(self, model_path: Optional[str] = None, shared_model=None):
        """
        Initialize the BallDetector.
        If a shared YOLO model is provided and no separate model path is specified, it will use the shared model.
        """
        self.model_path = model_path or settings.BALL_MODEL_PATH
        self.target_class_names = settings.BALL_CLASS_NAMES
        self.conf_threshold = settings.BALL_CONFIDENCE_THRESHOLD
        self.shared_model = shared_model
        
        self.standalone_model = None
        if not self.shared_model:
            from ultralytics import YOLO
            if os.path.exists(self.model_path):
                logger.info(f"Loading standalone YOLO model for ball detection from {self.model_path}")
                self.standalone_model = YOLO(self.model_path)
            else:
                logger.warning(f"Standalone ball model not found at {self.model_path}.")
                
    def extract_from_results(self, results, frame_number: int, timestamp: float, class_names: Dict[int, str]) -> List[Dict[str, Any]]:
        """Extract ball detections from existing YOLO results."""
        detections = []
        if results.boxes is None:
            return detections
            
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.int().cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        
        for box, cls, conf in zip(boxes, classes, confs):
            cls_id = int(cls)
            cls_name = class_names.get(cls_id, f"unknown_{cls_id}")
            conf_val = float(conf)
            
            # Filter by class and confidence
            if cls_name in self.target_class_names and conf_val >= self.conf_threshold:
                bbox = box.tolist()
                center = calculate_center(bbox)
                
                detections.append({
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf_val,
                    "bbox": bbox,
                    "center": center
                })
                
        return detections
        
    def detect(self, frame, frame_number: int, timestamp: float) -> List[Dict[str, Any]]:
        """Run detection on a single frame if a standalone model is used."""
        if self.standalone_model:
            results = self.standalone_model(
                frame, 
                conf=self.conf_threshold, 
                iou=settings.BALL_IOU_THRESHOLD,
                device=settings.DEVICE if settings.DEVICE else None,
                verbose=False
            )
            return self.extract_from_results(results[0], frame_number, timestamp, self.standalone_model.names)
        return []
