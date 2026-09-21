import pytest
from app.utils.helpers import calculate_center
from app.analytics.statistics import StatisticsManager

def test_calculate_center():
    bbox = [10, 10, 30, 30]
    cx, cy = calculate_center(bbox)
    assert cx == 20.0
    assert cy == 20.0
    
    with pytest.raises(ValueError):
        calculate_center([10, 10, 30])

def test_statistics_manager():
    manager = StatisticsManager()
    
    frame_data = [
        {"class_id": 0, "class_name": "person", "track_id": 1, "confidence": 0.9},
        {"class_id": 0, "class_name": "person", "track_id": 2, "confidence": 0.8},
        {"class_id": 32, "class_name": "sports ball", "track_id": 3, "confidence": 0.7}
    ]
    
    manager.update_frame_stats(frame_data)
    assert manager.stats["total_frames_processed"] == 1
    assert manager.stats["total_player_detections"] == 2
    assert len(manager.unique_ids) == 2
    
    video_info = {
        "fps": 30,
        "resolution": "1920x1080",
        "total_frames": 100
    }
    
    manager.finalize(video_info, processing_time=2.0)
    assert manager.stats["average_processing_fps"] == 0.5
    assert manager.stats["unique_player_ids_detected"] == 2
