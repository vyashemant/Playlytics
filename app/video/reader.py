import cv2
import os
from typing import Generator, Tuple, Any

class VideoReader:
    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Video file not found: {filepath}")
            
        self.filepath = filepath
        self.cap = cv2.VideoCapture(filepath)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {filepath}")
            
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def get_info(self) -> dict:
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "resolution": f"{self.width}x{self.height}"
        }
        
    def read_frames(self) -> Generator[Tuple[int, Any], None, None]:
        """Yields frame_number (0-indexed) and frame."""
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
            
    def release(self):
        if self.cap:
            self.cap.release()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
