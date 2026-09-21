import os
from abc import ABC, abstractmethod
from ultralytics import YOLO
from app.utils.logger import get_logger
from app.config.settings import settings

logger = get_logger(__name__)

class Detector(ABC):
    @abstractmethod
    def detect(self, frame, conf=None, iou=None):
        pass
        
    @abstractmethod
    def track(self, frame, conf=None, iou=None, persist=True):
        pass

class YOLODetector(Detector):
    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.device = device or settings.DEVICE
        
        if not os.path.exists(self.model_path):
            logger.warning(f"Model not found at {self.model_path}. YOLO will attempt to download it.")
            
        logger.info(f"Loading YOLO model from {self.model_path}")
        self.model = YOLO(self.model_path)
        
    def detect(self, frame, conf=None, iou=None):
        """Run detection on a single frame."""
        conf_thresh = conf if conf is not None else settings.CONFIDENCE_THRESHOLD
        iou_thresh = iou if iou is not None else settings.IOU_THRESHOLD
        
        results = self.model(
            frame, 
            conf=conf_thresh, 
            iou=iou_thresh,
            device=self.device if self.device else None,
            verbose=False
        )
        return results[0]
        
    def track(self, frame, conf=None, iou=None, persist=True):
        """Run tracking on a single frame."""
        conf_thresh = conf if conf is not None else settings.CONFIDENCE_THRESHOLD
        iou_thresh = iou if iou is not None else settings.IOU_THRESHOLD
        
        results = self.model.track(
            frame,
            persist=persist,
            conf=conf_thresh,
            iou=iou_thresh,
            device=self.device if self.device else None,
            verbose=False,
            tracker=settings.TRACKER
        )
        return results[0]
