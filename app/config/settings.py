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

    # Ball Detection & Tracking
    BALL_MODEL_PATH = os.getenv("BALL_MODEL_PATH", MODEL_PATH)
    BALL_CONFIDENCE_THRESHOLD = float(os.getenv("BALL_CONFIDENCE_THRESHOLD", "0.2"))
    BALL_IOU_THRESHOLD = float(os.getenv("BALL_IOU_THRESHOLD", "0.45"))
    
    _ball_classes_str = os.getenv("BALL_CLASS_NAMES", "['sports ball', 'cricket_ball']")
    try:
        BALL_CLASS_NAMES = ast.literal_eval(_ball_classes_str)
    except (ValueError, SyntaxError):
        BALL_CLASS_NAMES = ['sports ball', 'cricket_ball']
        
    BALL_MAX_MISSED_FRAMES = int(os.getenv("BALL_MAX_MISSED_FRAMES", "5"))
    BALL_SMOOTHING_ENABLED = os.getenv("BALL_SMOOTHING_ENABLED", "True").lower() == "true"
    BALL_SMOOTHING_ALPHA = float(os.getenv("BALL_SMOOTHING_ALPHA", "0.5"))
    BALL_BOUNCE_MIN_VERTICAL_DISPLACEMENT = float(os.getenv("BALL_BOUNCE_MIN_VERTICAL_DISPLACEMENT", "5.0"))
    BALL_BOUNCE_MIN_TIME_SECONDS = float(os.getenv("BALL_BOUNCE_MIN_TIME_SECONDS", "0.2"))

settings = Settings()
