"""
Main application entry point for fire detection system
"""
import cv2
import argparse
import time
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.detector import FireDetector
from src.video_handler import VideoHandler
from src.alert_system import AlertSystem
from src.preprocessor import Preprocessor
from src.utils import setup_logging, calculate_fps, draw_fps, draw_info_panel
from config.settings import get_settings

logger = logging.getLogger(__name__)

class FireDetectionApp:
    """Main fire detection application"""
    
    def __init__(self, config=None):
        """Initialize application"""
        self.config = config or get_settings()
        self.detector = FireDetector(self.config)
        self.alert_system = AlertSystem(self.config)
        self.preprocessor = Preprocessor()
        
        self.running = False
        self.paused = False
        self.frame_count = 0
        self.start_time = None
        self.detection_history = []
    
    def process_frame(self, frame):
        """Process single frame"""
        if frame is None:
            return None, False, []
        
        # Preprocessing
        resize_w = self.config.get('video.resize_width', 640)
        resize_h = self.config.get('video.resize_height', 480)
        processed = self.preprocessor.preprocessing_pipeline(
            frame,
            resize_width=resize_w,
            resize_height=resize_h,
            denoise=False,
            equalize=True
        )
        
        # Detection
        fire_detected, regions, result_frame = self.detector.detect(processed)
        
        self.detection_history.append(fire_detected)
        if len(self.detection_history) > 30:
            self.detection_history.pop(0)
        
        # Alert if fire detected
        if fire_detected and len(regions) > 0:
            self.alert_system.trigger_alert(result_frame, regions, source="video_stream")
            logger.warning(f"FIRE DETECTED! {len(regions)} region(s)")
        
        return result_frame, fire_detected, regions
    
    def run_webcam(self):
        """Run detection from webcam"""
        logger.info("Starting webcam capture...")
        self.running = True
        
        with VideoHandler(skip_frames=self.config.get('video.skip_frames', 1)) as video:
            properties = video.get_properties()
            logger.info(f"Webcam properties: {properties}")
            
            self.start_time = time.time()
            
            for frame in video.read_frames():
                if not self.running:
                    break
                
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                result_frame, fire_detected, regions = self.process_frame(frame)
                self.frame_count += 1
                
                # Draw info
                elapsed = time.time() - self.start_time
                fps = calculate_fps(self.frame_count, elapsed)
                result_frame = draw_fps(result_frame, fps)
                
                info = {
                    'Status': 'FIRE!' if fire_detected else 'CLEAR',
                    'Regions': len(regions),
                    'Alerts': self.alert_system.get_alert_count()
                }
                result_frame = draw_info_panel(result_frame, info, (10, 60))
                
                # Display
                cv2.imshow('Fire Detection', result_frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.paused = not self.paused
                    logger.info(f"Paused: {self.paused}")
        
        cv2.destroyAllWindows()
        logger.info("Webcam capture ended")
    
    def run_video(self, video_path: str):
        """Run detection on video file"""
        logger.info(f"Starting video analysis: {video_path}")
        self.running = True
        
        try:
            with VideoHandler(video_path, skip_frames=self.config.get('video.skip_frames', 1)) as video:
                properties = video.get_properties()
                logger.info(f"Video properties: {properties}")
                
                self.start_time = time.time()
                
                for frame in video.read_frames():
                    if not self.running:
                        break
                    
                    if self.paused:
                        time.sleep(0.1)
                        continue
                    
                    result_frame, fire_detected, regions = self.process_frame(frame)
                    self.frame_count += 1
                    
                    # Draw info
                    elapsed = time.time() - self.start_time
                    fps = calculate_fps(self.frame_count, elapsed)
                    result_frame = draw_fps(result_frame, fps)
                    
                    info = {
                        'Frame': self.frame_count,
                        'Status': 'FIRE!' if fire_detected else 'CLEAR',
                        'Regions': len(regions)
                    }
                    result_frame = draw_info_panel(result_frame, info, (10, 60))
                    
                    # Display
                    cv2.imshow('Fire Detection - Video', result_frame)
                    
                    # Check for key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        self.paused = not self.paused
        
        except Exception as e:
            logger.error(f"Error processing video: {e}")
        
        cv2.destroyAllWindows()
        logger.info("Video analysis ended")
    
    def run_image(self, image_path: str):
        """Run detection on single image"""
        logger.info(f"Processing image: {image_path}")
        
        try:
            with VideoHandler(image_path) as video:
                frame = video.read_frame()
                if frame is None:
                    logger.error(f"Could not read image: {image_path}")
                    return
                
                result_frame, fire_detected, regions = self.process_frame(frame)
                
                # Draw info
                info = {
                    'Status': 'FIRE DETECTED!' if fire_detected else 'NO FIRE',
                    'Regions': len(regions),
                    'Confidence': f"{regions[0]['confidence']:.2f}" if regions else "N/A"
                }
                result_frame = draw_info_panel(result_frame, info, (10, 60))
                
                # Display
                cv2.imshow('Fire Detection - Image', result_frame)
                logger.info(f"Fire detected: {fire_detected}")
                logger.info(f"Fire regions found: {len(regions)}")
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
    
    def get_statistics(self) -> dict:
        """Get detection statistics"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'frames_processed': self.frame_count,
            'elapsed_time': elapsed,
            'fps': calculate_fps(self.frame_count, elapsed),
            'total_alerts': self.alert_system.get_alert_count(),
            'fire_detections': sum(self.detection_history)
        }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Fire Detection System')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--threshold', type=float, help='Detection threshold')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--enable-logging', action='store_true', help='Enable file logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = None
    if args.enable_logging:
        log_file = './logs/fire_detection.log'
    setup_logging(log_file, level='INFO')
    
    logger.info("Fire Detection System Started")
    
    # Create app
    app = FireDetectionApp()
    
    # Override threshold if provided
    if args.threshold:
        app.detector.detection_threshold = args.threshold
        logger.info(f"Detection threshold set to {args.threshold}")
    
    # Run based on input source
    try:
        if args.image:
            app.run_image(args.image)
        elif args.video:
            app.run_video(args.video)
        else:
            app.run_webcam()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
    finally:
        # Print statistics
        stats = app.get_statistics()
        logger.info(f"Statistics: {stats}")
        logger.info("Fire Detection System Stopped")

if __name__ == "__main__":
    main()
