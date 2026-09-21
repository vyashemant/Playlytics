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
    
    logger.info("Processing video...\n")
    
    start_time = time.time()
    
    try:
        with VideoWriter(output_path, video_info['fps'], video_info['width'], video_info['height']) as writer:
            for frame_idx, frame in reader.read_frames():
                timestamp = frame_idx / max(video_info['fps'], 1)
                
                # Run tracking
                results = detector.track(frame)
                
                # Process and extract data
                frame_data = tracker.process_results(results, frame_idx, timestamp, class_names=detector.model.names)
                stats_manager.update_frame_stats(frame_data)
                
                # Annotate and write
                annotated_frame = tracker.annotate_frame(frame, frame_data, fps=video_info['fps'], total_frames=video_info['total_frames'])
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
    
    stats_path = os.path.join(os.path.dirname(output_path), "statistics.json")
    tracking_data_path = os.path.join(os.path.dirname(output_path), "tracking_data.json")
    movement_data_path = os.path.join(os.path.dirname(output_path), "movement_analysis.json")
    
    stats_manager.save(stats_path)
    tracking_data_manager.save(tracking_data_path)
    
    with open(movement_data_path, "w") as f:
        json.dump(movement_data, f, indent=4)
    
    print("\n")
    logger.info("Processing completed.\n")
    logger.info("Output:")
    logger.info(f"{output_path}")
    logger.info(f"{stats_path}")
    logger.info(f"{tracking_data_path}")
    logger.info(f"{movement_data_path}")

if __name__ == "__main__":
    main()


