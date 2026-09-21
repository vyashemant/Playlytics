import json
import os
from typing import List, Dict, Any

class TrackingDataManager:
    def __init__(self, fps: float, width: int, height: int):
        self.fps = fps
        self.width = width
        self.height = height
        self.tracks = []
        
    def add_tracks(self, tracks_data: List[Dict[str, Any]]):
        self.tracks = tracks_data
        
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "video": {
                "fps": self.fps,
                "width": self.width,
                "height": self.height
            },
            "tracks": self.tracks
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
