import json
import os
from typing import Dict, Any, List

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
            "total_player_detections": 0,
            "average_players_per_frame": 0.0,
            "maximum_players_in_frame": 0,
            "average_detection_confidence": 0.0,
            "unique_tracks": 0
        }
        self.unique_ids = set()
        self._total_players = 0
        self._total_confidence = 0.0
        self._confidence_count = 0
        self.track_summaries = []
        
    def update_frame_stats(self, frame_data: list):
        self.stats["total_frames_processed"] += 1
        
        players_in_frame = 0
        for data in frame_data:
            # Check if class name is target (e.g. 'person', 'player')
            cls_name = data.get("class_name", "")
            if cls_name in ["person", "player"] or "player" in cls_name.lower():
                self.stats["total_player_detections"] += 1
                self.unique_ids.add(data["track_id"])
                players_in_frame += 1
                
                # Confidence tracking
                self._total_confidence += data.get("confidence", 0.0)
                self._confidence_count += 1
                
        self._total_players += players_in_frame
        if players_in_frame > self.stats["maximum_players_in_frame"]:
            self.stats["maximum_players_in_frame"] = players_in_frame
                
    def finalize(self, video_info: dict, processing_time: float, track_summaries: List[Dict[str, Any]] = None):
        self.stats["fps"] = video_info["fps"]
        self.stats["resolution"] = video_info["resolution"]
        self.stats["video_duration_sec"] = video_info["total_frames"] / max(video_info["fps"], 1)
        self.stats["processing_time_sec"] = processing_time
        
        if processing_time > 0:
            self.stats["average_processing_fps"] = self.stats["total_frames_processed"] / processing_time
            
        self.stats["unique_player_ids_detected"] = len(self.unique_ids)
        self.stats["unique_tracks"] = len(self.unique_ids)
        
        frames_proc = max(1, self.stats["total_frames_processed"])
        self.stats["average_players_per_frame"] = round(self._total_players / frames_proc, 4)
        
        if self._confidence_count > 0:
            self.stats["average_detection_confidence"] = round(self._total_confidence / self._confidence_count, 4)
            
        if track_summaries is not None:
            self.track_summaries = track_summaries
        
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        out_data = {
            "statistics": self.stats,
            "track_summaries": self.track_summaries
        }
        with open(filepath, 'w') as f:
            json.dump(out_data, f, indent=4)

