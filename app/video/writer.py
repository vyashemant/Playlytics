import cv2
import os
from typing import Any

class VideoWriter:
    def __init__(self, filepath: str, fps: int, width: int, height: int):
        self.filepath = filepath
        
        dir_name = os.path.dirname(os.path.abspath(filepath))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        codecs = ['mp4v', 'avc1', 'XVID']
        self.writer = None
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            if writer.isOpened():
                self.writer = writer
                break
                
        if self.writer is None or not self.writer.isOpened():
            raise ValueError(f"Could not open video writer for: {filepath} with any tested codec.")
            
    def write_frame(self, frame: Any):
        if self.writer is not None:
            self.writer.write(frame)
        
    def release(self):
        if self.writer:
            self.writer.release()
            self.writer = None
            
        # Verify output file
        if not os.path.exists(self.filepath):
            raise RuntimeError(f"Output video file was not created: {self.filepath}")
            
        if os.path.getsize(self.filepath) == 0:
            raise RuntimeError(f"Output video file is empty: {self.filepath}")
            
        # Reopen and read one frame
        cap = cv2.VideoCapture(self.filepath)
        if not cap.isOpened():
            raise RuntimeError(f"Output video file cannot be opened for reading: {self.filepath}")
            
        ret, _ = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError(f"Cannot read any frames from output video: {self.filepath}")
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
