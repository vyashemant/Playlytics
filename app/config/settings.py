import os
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

settings = Settings()
