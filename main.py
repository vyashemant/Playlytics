import argparse
import time
import os
import sys
import json

from app.config.settings import settings
from app.utils.logger import get_logger
from app.video.reader import VideoReader
from app.video.writer import VideoWriter
from app.detection.detector import YOLODetector
from app.detection.tracker import TrackerManager
from app.analytics.statistics import StatisticsManager
from app.analytics.tracking_data import TrackingDataManager
from app.analytics.movement import MovementAnalyzer

from app.detection.ball_detector import BallDetector
from app.tracking.ball_tracker import BallTracker
from app.analytics.ball_trajectory import BallTrajectoryAnalyzer
from app.analytics.ball_events import BallEventDetector

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Playlytics - Sports Performance Analytics")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, help="Path to output video")
    parser.add_argument("--model", type=str, help="Path to YOLO model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    input_path = args.input
    output_path = args.output or os.path.join(settings.OUTPUT_DIR, f"analyzed_{os.path.basename(input_path)}")
    model_path = args.model or settings.MODEL_PATH
    
    logger.info("Playlytics")
    logger.info("────────────────────────────")
    logger.info(f"Input: {input_path}")
    
    try:
        reader = VideoReader(input_path)
    except Exception as e:
        logger.error(f"Failed to initialize video reader: {e}")
        sys.exit(1)
        
    video_info = reader.get_info()
    logger.info(f"Resolution: {video_info['resolution']}")
    logger.info(f"FPS: {video_info['fps']}")
    logger.info(f"Frames: {video_info['total_frames']}")
    
    logger.info("\nLoading YOLO model...")
    try:
        detector = YOLODetector(model_path=model_path)
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        sys.exit(1)
        
    tracker = TrackerManager()
    stats_manager = StatisticsManager()
    tracking_data_manager = TrackingDataManager(
        fps=video_info['fps'], 
        width=video_info['width'], 
        height=video_info['height']
    )
    movement_analyzer = MovementAnalyzer(
        fps=video_info['fps'],
        width=video_info['width'],
        height=video_info['height']
    )
    
    ball_detector = BallDetector(shared_model=detector.model if settings.BALL_MODEL_PATH == settings.MODEL_PATH else None)
    ball_tracker = BallTracker()
    ball_trajectory_analyzer = BallTrajectoryAnalyzer()
    ball_event_detector = BallEventDetector()
    
    logger.info("Processing video...\n")
    
    start_time = time.time()
    
    try:
        with VideoWriter(output_path, video_info['fps'], video_info['width'], video_info['height']) as writer:
            for frame_idx, frame in reader.read_frames():
                timestamp = frame_idx / max(video_info['fps'], 1)
                
                # Run tracking
                results = detector.track(frame)
                
                # Process ball detection
                if ball_detector.shared_model:
                    ball_detections = ball_detector.extract_from_results(results, frame_idx, timestamp, detector.model.names)
                else:
                    ball_detections = ball_detector.detect(frame, frame_idx, timestamp)
                    
                ball_tracker.update(ball_detections, frame_idx)
                
                # Process and extract data
                frame_data = tracker.process_results(results, frame_idx, timestamp, class_names=detector.model.names)
                stats_manager.update_frame_stats(frame_data)
                
                # Inject ball for annotation
                if ball_tracker.active_trajectory:
                    latest_ball = ball_tracker.active_trajectory[-1]
                    if latest_ball["frame_number"] == frame_idx:
                        frame_data.append({
                            "track_id": latest_ball["track_id"],
                            "confidence": latest_ball["confidence"],
                            "bbox": latest_ball["bbox"],
                            "class_name": latest_ball["class_name"],
                            "frame_number": frame_idx,
                            "timestamp": timestamp,
                            "center": latest_ball.get("center", [])
                        })
                        
                # Annotate and write
                annotated_frame = tracker.annotate_frame(frame, frame_data, fps=video_info['fps'], total_frames=video_info['total_frames'])
                
                # Optional: draw trajectory line
                if len(ball_tracker.active_trajectory) > 1:
                    import cv2
                    for i in range(1, len(ball_tracker.active_trajectory)):
                        pt1 = tuple(map(int, ball_tracker.active_trajectory[i-1]["center"]))
                        pt2 = tuple(map(int, ball_tracker.active_trajectory[i]["center"]))
                        cv2.line(annotated_frame, pt1, pt2, (0, 165, 255), 2)
                        
                writer.write_frame(annotated_frame)
                
                # Progress update
                if frame_idx % max(1, video_info['total_frames'] // 100) == 0 or frame_idx == video_info['total_frames'] - 1:
                    progress = (frame_idx + 1) / video_info['total_frames'] * 100
                    elapsed = time.time() - start_time
                    current_fps = (frame_idx + 1) / max(elapsed, 0.001)
                    
                    sys.stdout.write(f"\rProgress: {progress:.1f}% | Players detected: {len(stats_manager.unique_ids)} | Processing FPS: {current_fps:.1f}")
                    sys.stdout.flush()
                    
    except Exception as e:
        logger.error(f"\nError during processing: {e}")
        reader.release()
        sys.exit(1)
        
    reader.release()
        
    processing_time = time.time() - start_time
    
    track_summaries = tracker.get_track_summaries()
    stats_manager.finalize(video_info, processing_time, track_summaries)
    
    tracking_data_manager.add_tracks(tracker.get_all_tracking_data())
    
    # Run Movement Analysis
    movement_data = movement_analyzer.analyze(tracker.tracks.values())
    
    # Finalize Ball Tracking
    ball_tracker.finalize()
    ball_trajectory = ball_tracker.get_trajectory()
    ball_analysis = ball_trajectory_analyzer.analyze(ball_trajectory)
    
    if ball_tracker.tracking_status == "reliable_trajectory":
        ball_events = ball_event_detector.detect_events(ball_analysis)
    else:
        ball_events = []
    
    ball_tracking_output = {
        "video": {
            "fps": video_info['fps'],
            "width": video_info['width'],
            "height": video_info['height'],
            "duration_seconds": round(video_info['total_frames'] / max(video_info['fps'], 1), 2)
        },
        "ball": {
            "tracking_status": ball_tracker.tracking_status,
            "track_id": ball_tracker.best_trajectory[0]["track_id"] if ball_tracker.best_trajectory else None,
            "frames_observed": len(ball_tracker.best_trajectory),
            "first_seen_frame": ball_tracker.best_trajectory[0]["frame_number"] if ball_tracker.best_trajectory else None,
            "last_seen_frame": ball_tracker.best_trajectory[-1]["frame_number"] if ball_tracker.best_trajectory else None,
        },
        "trajectory": ball_analysis,
        "events": ball_events
    }
    
    stats_path = os.path.join(os.path.dirname(output_path), "statistics.json")
    tracking_data_path = os.path.join(os.path.dirname(output_path), "tracking_data.json")
    movement_data_path = os.path.join(os.path.dirname(output_path), "movement_analysis.json")
    ball_data_path = os.path.join(os.path.dirname(output_path), "ball_tracking.json")
    
    stats_manager.save(stats_path)
    tracking_data_manager.save(tracking_data_path)
    
    with open(movement_data_path, "w") as f:
        json.dump(movement_data, f, indent=4)
        
    with open(ball_data_path, "w") as f:
        json.dump(ball_tracking_output, f, indent=4)
    
    print("\n")
    logger.info("Processing completed.\n")
    logger.info("Output:")
    logger.info(f"{output_path}")
    logger.info(f"{stats_path}")
    logger.info(f"{tracking_data_path}")
    logger.info(f"{movement_data_path}")
    logger.info(f"{ball_data_path}")

if __name__ == "__main__":
    main()


