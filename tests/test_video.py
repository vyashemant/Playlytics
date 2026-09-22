import pytest
import os
import cv2
import numpy as np
import tempfile
from app.video.writer import VideoWriter

def test_video_writer_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_out.mp4")
        fps = 30
        width = 640
        height = 480
        
        with VideoWriter(filepath, fps, width, height) as writer:
            # Write 5 dummy frames
            for _ in range(5):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                writer.write_frame(frame)
                
        # Assert file exists and size > 0
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
        
        # Verify it can be reopened and read
        cap = cv2.VideoCapture(filepath)
        assert cap.isOpened()
        ret, frame = cap.read()
        assert ret is True
        assert frame is not None
        cap.release()

def test_video_writer_invalid_path():
    # Attempt to write to a path that OS rejects (like a directory)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Pass a directory path where a file path is expected
        with pytest.raises((ValueError, RuntimeError, OSError)):
            # This might fail in os.makedirs or OpenCV.
            writer = VideoWriter(tmpdir, 30, 640, 480)
            writer.release()
