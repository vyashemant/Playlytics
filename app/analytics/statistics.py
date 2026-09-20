import json
import os
from typing import Dict, Any

class StatisticsManager:
    def __init__(self):
        self.stats = {
            "video_duration_sec": 0.0,
            "fps": 0,
            "resolution": "",
            "total_frames_processed": 0,
            "processing_time_sec": 0.0,
            "average_processing_fps": 0.0,
            "unique_player_ids_detected": 0,
            "total_player_detections": 0
        }
        self.unique_ids = set()
        
    def update_frame_stats(self, frame_data: list):
        self.stats["total_frames_processed"] += 1
        for data in frame_data:
            if data["class_id"] == 0:  # Player (person)
                self.stats["total_player_detections"] += 1
                self.unique_ids.add(data["track_id"])
                
    def finalize(self, video_info: dict, processing_time: float):
        self.stats["fps"] = video_info["fps"]
        self.stats["resolution"] = video_info["resolution"]
        self.stats["video_duration_sec"] = video_info["total_frames"] / max(video_info["fps"], 1)
        self.stats["processing_time_sec"] = processing_time
        
        if processing_time > 0:
            self.stats["average_processing_fps"] = self.stats["total_frames_processed"] / processing_time
            
        self.stats["unique_player_ids_detected"] = len(self.unique_ids)
        
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=4)
