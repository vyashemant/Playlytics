import pytest
from app.detection.tracker import TrackerManager, PlayerTrack
from app.analytics.statistics import StatisticsManager
import numpy as np

class MockBoxes:
    def __init__(self, ids, cls, conf, xyxy):
        import torch
        self.id = torch.tensor(ids) if ids is not None else None
        self.cls = torch.tensor(cls) if cls is not None else None
        self.conf = torch.tensor(conf) if conf is not None else None
        self.xyxy = torch.tensor(xyxy) if xyxy is not None else None

class MockResults:
    def __init__(self, ids, cls, conf, xyxy):
        self.boxes = MockBoxes(ids, cls, conf, xyxy)

def test_tracker_lifecycle():
    tracker = TrackerManager()
    tracker.target_class_names = ["person"]
    
    # Frame 1
    results_f1 = MockResults(
        ids=[1], 
        cls=[0], 
        conf=[0.9], 
        xyxy=[[10, 10, 20, 20]]
    )
    class_names = {0: "person", 32: "sports ball"}
    
    frame_data_f1 = tracker.process_results(results_f1, 1, 0.1, class_names)
    
    assert len(frame_data_f1) == 1
    assert 1 in tracker.tracks
    
    track = tracker.tracks[1]
    assert track.first_seen_frame == 1
    assert track.last_seen_frame == 1
    assert track.frames_visible == 1
    assert round(track.average_confidence, 4) == 0.9000
    
    # Frame 2
    results_f2 = MockResults(
        ids=[1, 2], 
        cls=[0, 32], 
        conf=[0.8, 0.95], 
        xyxy=[[12, 12, 22, 22], [50, 50, 60, 60]]
    )
    
    frame_data_f2 = tracker.process_results(results_f2, 2, 0.2, class_names)
    
    # Only "person" should be tracked since target_class_names = ["person"]
    assert len(frame_data_f2) == 1
    assert 2 not in tracker.tracks
    
    track = tracker.tracks[1]
    assert track.last_seen_frame == 2
    assert track.frames_visible == 2
    assert round(track.average_confidence, 4) == 0.8500 # (0.9 + 0.8) / 2
    
def test_statistics():
    stats = StatisticsManager()
    
    # Frame 1
    frame_data_f1 = [
        {"track_id": 1, "class_name": "person", "confidence": 0.8},
        {"track_id": 2, "class_name": "person", "confidence": 0.9}
    ]
    
    stats.update_frame_stats(frame_data_f1)
    assert stats.stats["maximum_players_in_frame"] == 2
    
    # Frame 2
    frame_data_f2 = [
        {"track_id": 1, "class_name": "person", "confidence": 0.85}
    ]
    stats.update_frame_stats(frame_data_f2)
    assert stats.stats["maximum_players_in_frame"] == 2
    
    stats.finalize({"fps": 30, "resolution": "1080p", "total_frames": 2}, 1.0)
    
    assert stats.stats["unique_tracks"] == 2
    assert stats.stats["average_players_per_frame"] == 1.5 # 3 players over 2 frames
    assert stats.stats["average_detection_confidence"] == round((0.8 + 0.9 + 0.85) / 3, 4)
