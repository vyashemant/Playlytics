import os
import ast
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    MODEL_PATH = os.getenv("MODEL_PATH", "models/yolo11n.pt")
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))
    IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
    INPUT_DIR = os.getenv("INPUT_DIR", "input/")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/")
    
    # Optional device (cpu, cuda, mps) - if empty string, YOLO auto-detects
    DEVICE = os.getenv("DEVICE", "")
    
    # Tracker configuration (botsort.yaml, bytetrack.yaml)
    TRACKER = os.getenv("TRACKER", "botsort.yaml")
    
    # Target classes mapping
    # 0 is 'person', 32 is 'sports ball' in COCO dataset
    # Future cricket model: 0='player', 1='cricket_ball', 2='cricket_bat', 3='wicket', 4='umpire'
    # We parse it as a list of strings from env
    _target_classes_str = os.getenv("TARGET_CLASSES", "['person']")
    try:
        TARGET_CLASSES = ast.literal_eval(_target_classes_str)
    except (ValueError, SyntaxError):
        TARGET_CLASSES = ['person']
        
    # Movement Analytics
    MOVEMENT_THRESHOLD_PIXELS = float(os.getenv("MOVEMENT_THRESHOLD_PIXELS", "2.0"))

settings = Settings()
