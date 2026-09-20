# Playlytics

Sports Performance Analytics using Computer Vision.

## Overview
This project provides the foundation for extracting sports analytics (initially Cricket) from videos using Computer Vision techniques.

> **Note**: The current version is the Computer Vision foundation and does not yet provide complete cricket performance analytics.

## Current Capabilities
- Read and process video frames using OpenCV.
- Detect players and sports balls using Ultralytics YOLO.
- Track players across frames with persistent IDs.
- Generate annotated output videos.
- Extract basic detection and processing statistics to JSON.

## Architecture
The pipeline is designed to be modular:
- `app/config/`: Centralized configuration.
- `app/video/`: Video I/O operations.
- `app/detection/`: YOLO detection and tracking wrappers.
- `app/analytics/`: Metrics and statistics collection.

## Installation
1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and adjust settings.

## Running the Application
```bash
python main.py --input input/match.mp4
```

Optional arguments:
```bash
python main.py --input input/match.mp4 --output output/match_analysis.mp4 --model models/yolo11n.pt
```

## Testing
Run unit tests with pytest:
```bash
pytest tests/
```

## Roadmap
- Phase 1: Computer Vision Foundation (Current)
- Phase 2: Cricket-specific detection and tracking
- Phase 3: Player movement analytics
- Phase 4: Ball trajectory and event detection
- Phase 5: Pose estimation
- Phase 6: Shot/bowling analysis
- Phase 7: Performance scoring
- Phase 8: React analytics dashboard
- Phase 9: Backend/API
- Phase 10: Deployment
