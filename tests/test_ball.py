import pytest
from app.detection.ball_detector import BallDetector
from app.tracking.ball_tracker import BallTracker
from app.analytics.ball_trajectory import BallTrajectoryAnalyzer
from app.analytics.ball_events import BallEventDetector

def test_ball_detector_extract():
    detector = BallDetector(shared_model=True)
    detector.conf_threshold = 0.2
    detector.target_class_names = ["sports ball"]
    
    # mock results object
    class MockResults:
        def __init__(self):
            self.boxes = self.MockBoxes()
        class MockBoxes:
            def __init__(self):
                import torch
                self.xyxy = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]])
                self.cls = torch.tensor([32, 0])
                self.conf = torch.tensor([0.9, 0.8])
    
    results = MockResults()
    class_names = {0: "person", 32: "sports ball"}
    
    detections = detector.extract_from_results(results, 1, 0.1, class_names)
    assert len(detections) == 1
    assert detections[0]["class_name"] == "sports ball"
    assert detections[0]["confidence"] == pytest.approx(0.9)

def test_ball_tracker():
    tracker = BallTracker()
    tracker.max_missed_frames = 2
    
    tracker.update([{"confidence": 0.8, "center": [15, 15], "class_name": "sports ball"}], 1)
    tracker.update([], 2)
    tracker.update([{"confidence": 0.9, "center": [16, 16], "class_name": "sports ball"}], 3)
    
    assert len(tracker.active_trajectory) == 2
    
    # Exceed max gap
    tracker.update([], 4)
    tracker.update([], 5)
    tracker.update([], 6)
    
    assert tracker.current_track_id is None
    assert len(tracker.active_trajectory) == 0
    assert len(tracker.best_trajectory) == 2
    
def test_ball_tracker_states():
    # 1. No detection
    tracker1 = BallTracker()
    tracker1.finalize()
    assert tracker1.tracking_status == "no_detection"
    
    # 2. Insufficient trajectory (< 10 frames)
    tracker2 = BallTracker()
    for i in range(5):
        tracker2.update([{"confidence": 0.9, "center": [10*i, 10*i], "class_name": "sports ball"}], i)
    tracker2.finalize()
    assert tracker2.tracking_status == "insufficient_trajectory"
    
    # 3. Unreliable candidate (stationary / low displacement)
    tracker3 = BallTracker()
    for i in range(15):
        tracker3.update([{"confidence": 0.9, "center": [10, 10], "class_name": "sports ball"}], i)
    tracker3.finalize()
    assert tracker3.tracking_status == "unreliable_candidate"
    
    # 4. Reliable trajectory
    tracker4 = BallTracker()
    for i in range(15):
        tracker4.update([{"confidence": 0.9, "center": [10 + i*10, 10 + i*10], "class_name": "sports ball"}], i)
    tracker4.finalize()
    assert tracker4.tracking_status == "reliable_trajectory"

def test_ball_trajectory_analyzer():
    analyzer = BallTrajectoryAnalyzer()
    analyzer.smoothing_enabled = False
    
    traj = [
        {"frame_number": 1, "timestamp": 0.0, "center": [0, 0], "confidence": 1.0},
        {"frame_number": 2, "timestamp": 1.0, "center": [10, 0], "confidence": 1.0}
    ]
    
    res = analyzer.analyze(traj)
    assert res["displacement_pixels"] == 10.0
    assert res["total_distance_pixels"] == 10.0
    assert res["average_velocity_pixels_per_second"] == 10.0

def test_ball_events():
    detector = BallEventDetector()
    detector.min_vertical_displacement = 5.0
    detector.min_time_seconds = 0.5
    
    # y increases then decreases
    traj_data = {
        "points": [
            {"frame_number": 1, "timestamp": 0.0, "smoothed_center": [10, 10], "confidence": 1.0},
            {"frame_number": 2, "timestamp": 1.0, "smoothed_center": [10, 20], "confidence": 1.0}, # lowest point
            {"frame_number": 3, "timestamp": 2.0, "smoothed_center": [10, 10], "confidence": 1.0}
        ]
    }
    
    events = detector.detect_events(traj_data)
    # 1 start, 1 bounce, 1 end = 3 events
    assert len(events) == 3
    assert events[1]["event_type"] == "bounce_candidate"
    assert events[1]["frame_number"] == 2
