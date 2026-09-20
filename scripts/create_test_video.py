import cv2
import numpy as np
import os

def create_test_video(filepath: str, width: int = 640, height: int = 480, fps: int = 30, duration_sec: int = 3):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    
    for i in range(fps * duration_sec):
        # Create a light gray background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Draw a moving "person" (a dark rectangle that YOLO might detect as a person)
        px = int((i / (fps * duration_sec)) * width)
        py = height // 2
        cv2.rectangle(frame, (px, py-100), (px+50, py+100), (50, 50, 50), -1)
        
        # Draw a moving "ball"
        bx = int(((fps * duration_sec - i) / (fps * duration_sec)) * width)
        by = int(height // 2 + 50 * np.sin(i * 0.2))
        cv2.circle(frame, (bx, by), 10, (0, 0, 0), -1)
        
        out.write(frame)
        
    out.release()

if __name__ == "__main__":
    create_test_video("input/test_video.mp4")
    print("Test video created at input/test_video.mp4")
