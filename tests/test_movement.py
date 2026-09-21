import pytest
from app.analytics.movement import MovementAnalyzer
from app.detection.tracker import PlayerTrack

def test_distance():
    analyzer = MovementAnalyzer(fps=1.0, width=1920, height=1080)
    track = PlayerTrack(track_id=1, class_name="person", class_id=0, first_seen_frame=1, last_seen_frame=2, first_seen_timestamp=1.0, last_seen_timestamp=2.0)
    
    # (0,0) -> (3,4) = 5
    track.frames_data = [
        {"frame_number": 1, "timestamp": 1.0, "center": [0, 0], "bbox": [0,0,0,0], "confidence": 1.0},
        {"frame_number": 2, "timestamp": 2.0, "center": [3, 4], "bbox": [0,0,0,0], "confidence": 1.0}
    ]
    
    metrics, timeline = analyzer._analyze_track(track)
    assert metrics["total_distance_pixels"] == 5.0
    
def test_displacement():
    analyzer = MovementAnalyzer(fps=1.0, width=1920, height=1080)
    track = PlayerTrack(track_id=1, class_name="person", class_id=0, first_seen_frame=1, last_seen_frame=3, first_seen_timestamp=1.0, last_seen_timestamp=3.0)
    
    track.frames_data = [
        {"frame_number": 1, "timestamp": 1.0, "center": [0, 0], "bbox": [0,0,0,0], "confidence": 1.0},
        {"frame_number": 2, "timestamp": 2.0, "center": [100, 100], "bbox": [0,0,0,0], "confidence": 1.0},
        {"frame_number": 3, "timestamp": 3.0, "center": [3, 4], "bbox": [0,0,0,0], "confidence": 1.0}
    ]
    
    metrics, timeline = analyzer._analyze_track(track)
    # distance is huge, but displacement is (0,0) -> (3,4) = 5
    assert metrics["displacement_pixels"] == 5.0

def test_velocity():
    analyzer = MovementAnalyzer(fps=1.0, width=1920, height=1080)
    track = PlayerTrack(track_id=1, class_name="person", class_id=0, first_seen_frame=1, last_seen_frame=2, first_seen_timestamp=1.0, last_seen_timestamp=3.0)
    
    # distance 10, time 2 -> 5 pixels/sec
    track.frames_data = [
        {"frame_number": 1, "timestamp": 1.0, "center": [0, 0], "bbox": [0,0,0,0], "confidence": 1.0},
        {"frame_number": 2, "timestamp": 3.0, "center": [10, 0], "bbox": [0,0,0,0], "confidence": 1.0}
    ]
    
    metrics, timeline = analyzer._analyze_track(track)
    assert metrics["average_velocity_pixels_per_second"] == 5.0
    assert metrics["maximum_velocity_pixels_per_second"] == 5.0

def test_acceleration():
    analyzer = MovementAnalyzer(fps=1.0, width=1920, height=1080)
    track = PlayerTrack(track_id=1, class_name="person", class_id=0, first_seen_frame=1, last_seen_frame=3, first_seen_timestamp=1.0, last_seen_timestamp=3.0)
    
    track.frames_data = [
        {"frame_number": 1, "timestamp": 1.0, "center": [0, 0], "bbox": [0,0,0,0], "confidence": 1.0},
        {"frame_number": 2, "timestamp": 2.0, "center": [10, 0], "bbox": [0,0,0,0], "confidence": 1.0}, # vel 10, accel 0
        {"frame_number": 3, "timestamp": 3.0, "center": [30, 0], "bbox": [0,0,0,0], "confidence": 1.0}  # vel 20, accel 10
    ]
    
    metrics, timeline = analyzer._analyze_track(track)
    assert metrics["average_acceleration_pixels_per_second_squared"] == 5.0
    assert metrics["maximum_acceleration_pixels_per_second_squared"] == 10.0

def test_stationary_detection():
    analyzer = MovementAnalyzer(fps=1.0, width=1920, height=1080)
    analyzer.threshold = 2.0
    track = PlayerTrack(track_id=1, class_name="person", class_id=0, first_seen_frame=1, last_seen_frame=3, first_seen_timestamp=1.0, last_seen_timestamp=3.0)
    
    track.frames_data = [
        {"frame_number": 1, "timestamp": 1.0, "center": [0, 0], "bbox": [0,0,0,0], "confidence": 1.0},
        {"frame_number": 2, "timestamp": 2.0, "center": [1, 0], "bbox": [0,0,0,0], "confidence": 1.0}, # dist 1 (<2) -> stationary
        {"frame_number": 3, "timestamp": 3.0, "center": [10, 0], "bbox": [0,0,0,0], "confidence": 1.0} # dist 9 (>2) -> moving
    ]
    
    metrics, timeline = analyzer._analyze_track(track)
    assert metrics["stationary_frames"] == 2 # Frame 1 and Frame 2
    assert metrics["moving_frames"] == 1 # Frame 3
    assert metrics["movement_percentage"] == round(1/3 * 100, 2)
    assert metrics["stationary_percentage"] == round(2/3 * 100, 2)
    
def test_direction():
    analyzer = MovementAnalyzer(fps=1.0, width=1920, height=1080)
    track = PlayerTrack(track_id=1, class_name="person", class_id=0, first_seen_frame=1, last_seen_frame=3, first_seen_timestamp=1.0, last_seen_timestamp=3.0)
    
    track.frames_data = [
        {"frame_number": 1, "timestamp": 1.0, "center": [0, 0], "bbox": [0,0,0,0], "confidence": 1.0},
        {"frame_number": 2, "timestamp": 2.0, "center": [10, 0], "bbox": [0,0,0,0], "confidence": 1.0}, # (0,0)->(10,0) is 0 deg
        {"frame_number": 3, "timestamp": 3.0, "center": [10, 10], "bbox": [0,0,0,0], "confidence": 1.0} # (10,0)->(10,10) is 90 deg
    ]
    
    metrics, timeline = analyzer._analyze_track(track)
    assert timeline[1]["direction_degrees"] == 0.0
    assert timeline[2]["direction_degrees"] == 90.0
    
def test_empty_track():
    analyzer = MovementAnalyzer(fps=1.0, width=1920, height=1080)
    track = PlayerTrack(track_id=1, class_name="person", class_id=0, first_seen_frame=0, last_seen_frame=0, first_seen_timestamp=0.0, last_seen_timestamp=0.0)
    
    # Should not crash
    results = analyzer.analyze([track])
    assert len(results["players"]) == 0

def test_single_position():
    analyzer = MovementAnalyzer(fps=1.0, width=1920, height=1080)
    track = PlayerTrack(track_id=1, class_name="person", class_id=0, first_seen_frame=1, last_seen_frame=1, first_seen_timestamp=1.0, last_seen_timestamp=1.0)
    
    track.frames_data = [
        {"frame_number": 1, "timestamp": 1.0, "center": [0, 0], "bbox": [0,0,0,0], "confidence": 1.0}
    ]
    
    metrics, timeline = analyzer._analyze_track(track)
    assert metrics["total_distance_pixels"] == 0.0
    assert metrics["displacement_pixels"] == 0.0
    assert metrics["average_velocity_pixels_per_second"] == 0.0
    assert metrics["movement_percentage"] == 0.0
    assert metrics["stationary_percentage"] == 100.0

def test_multiple_players():
    analyzer = MovementAnalyzer(fps=1.0, width=1920, height=1080)
    track1 = PlayerTrack(track_id=1, class_name="person", class_id=0, first_seen_frame=1, last_seen_frame=2, first_seen_timestamp=1.0, last_seen_timestamp=2.0)
    track1.frames_data = [
        {"frame_number": 1, "timestamp": 1.0, "center": [0, 0], "bbox": [0,0,0,0], "confidence": 1.0},
        {"frame_number": 2, "timestamp": 2.0, "center": [3, 4], "bbox": [0,0,0,0], "confidence": 1.0}
    ]
    
    track2 = PlayerTrack(track_id=2, class_name="person", class_id=0, first_seen_frame=1, last_seen_frame=2, first_seen_timestamp=1.0, last_seen_timestamp=2.0)
    track2.frames_data = [
        {"frame_number": 1, "timestamp": 1.0, "center": [0, 0], "bbox": [0,0,0,0], "confidence": 1.0},
        {"frame_number": 2, "timestamp": 2.0, "center": [0, 10], "bbox": [0,0,0,0], "confidence": 1.0}
    ]
    
    results = analyzer.analyze([track1, track2])
    assert len(results["players"]) == 2
    
    p1 = next(p for p in results["players"] if p["track_id"] == 1)
    p2 = next(p for p in results["players"] if p["track_id"] == 2)
    
    assert p1["metrics"]["total_distance_pixels"] == 5.0
    assert p2["metrics"]["total_distance_pixels"] == 10.0
