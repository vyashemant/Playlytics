# Playlytics

Sports Performance Analytics using Computer Vision.

## Overview
This project provides the foundation for extracting sports analytics (initially Cricket) from videos using Computer Vision techniques.

> **Note**: The current version is the Computer Vision foundation and does not yet provide complete cricket performance analytics.

## Current Capabilities
- Read and process video frames using OpenCV.
- Detect players and sports balls using Ultralytics YOLO with configurable models.
- Track players across frames with persistent IDs and track lifecycles.
- Configurable tracking (BoT-SORT / ByteTrack) and configurable target classes (defaulting to targeting `person`).
- Generate annotated output videos displaying track IDs, confidence, and basic processing info.
- Extract advanced detection and processing statistics to `statistics.json` including player counts, confidence averages, and track summaries.
- Export frame-level tracking data to `tracking_data.json` for future analytics and database integration.
- **Player Movement Analytics**: Generates `movement_analysis.json` containing pixel distance travelled, pixel displacement, pixel velocity, pixel acceleration, movement direction, moving/stationary periods, and movement percentages.

## Current Limitations
- The generic pretrained YOLO model is not a cricket-specific model.
- By default, only the `person` class is reliably tracked using the pretrained model. 
- Cricket-ball detection is NOT reliable with standard COCO models and requires a custom model.
- Bat and wicket detection are not implemented unless a custom cricket model is supplied and configured.
- **Movement metrics are currently calculated in image/pixel coordinates and are not real-world physical measurements.**
- Player real-world speed calculation is not yet implemented.
- No performance scoring exists yet.

## Architecture
The pipeline is designed to be modular:
- `app/config/`: Centralized configuration.
- `app/video/`: Video I/O operations.
- `app/detection/`: YOLO detection and tracking wrappers.
- `app/analytics/`: Metrics and statistics collection, tracking data, and movement analysis.

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
- Phase 1: Computer Vision Foundation (Completed)
- Phase 2: Cricket Detection & Tracking (Completed)
- Phase 3: Player Movement Analytics (Current)
- Phase 4: Ball Trajectory & Event Detection (Next)
- Phase 5: Pose estimation
- Phase 6: Shot/bowling analysis
- Phase 7: Performance scoring
- Phase 8: React analytics dashboard
- Phase 9: Backend/API
- Phase 10: Deployment
