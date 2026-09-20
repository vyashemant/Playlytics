import cv2
import os
from typing import Any

class VideoWriter:
    def __init__(self, filepath: str, fps: int, width: int, height: int):
        self.filepath = filepath
        
        dir_name = os.path.dirname(os.path.abspath(filepath))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        # Use mp4v codec for mp4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        
        if not self.writer.isOpened():
            raise ValueError(f"Could not open video writer for: {filepath}")
            
    def write_frame(self, frame: Any):
        self.writer.write(frame)
        
    def release(self):
        if self.writer:
            self.writer.release()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
