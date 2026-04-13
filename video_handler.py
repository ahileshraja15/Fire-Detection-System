"""
Video and image input handling
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Iterator
import logging

logger = logging.getLogger(__name__)

class VideoHandler:
    """Handle video capture from camera, video file, or image"""
    
    def __init__(self, source: Optional[str] = None, skip_frames: int = 1):
        """
        Initialize video handler
        
        Args:
            source: Video file path, image path, or None for webcam
            skip_frames: Number of frames to skip between processing
        """
        self.source = source
        self.skip_frames = skip_frames
        self.cap = None
        self.is_image = False
        self.current_frame = None
        self.frame_count = 0
        self._init_capture()
    
    def _init_capture(self):
        """Initialize video capture"""
        if self.source is None:
            # Use default webcam
            self.cap = cv2.VideoCapture(0)
            logger.info("Initialized webcam capture")
        elif isinstance(self.source, str):
            source_path = Path(self.source)
            
            if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                # Image file
                self.current_frame = cv2.imread(str(source_path))
                if self.current_frame is None:
                    raise ValueError(f"Cannot read image: {source_path}")
                self.is_image = True
                logger.info(f"Loaded image: {source_path}")
            else:
                # Video file
                self.cap = cv2.VideoCapture(str(source_path))
                if not self.cap.isOpened():
                    raise ValueError(f"Cannot open video: {source_path}")
                logger.info(f"Opened video: {source_path}")
        else:
            # Assume it's a camera index
            self.cap = cv2.VideoCapture(int(self.source))
            logger.info(f"Initialized camera {self.source}")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame"""
        if self.is_image:
            return self.current_frame
        
        success, frame = self.cap.read()
        if success:
            self.frame_count += 1
        return frame if success else None
    
    def read_frames(self) -> Iterator[np.ndarray]:
        """
        Generator that yields frames
        Respects skip_frames setting
        """
        frame_index = 0
        
        if self.is_image:
            yield self.current_frame
        else:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                if frame_index % self.skip_frames == 0:
                    self.frame_count += 1
                    yield frame
                
                frame_index += 1
    
    def get_properties(self) -> dict:
        """Get video properties"""
        if self.is_image:
            h, w = self.current_frame.shape[:2]
            return {
                'width': w,
                'height': h,
                'fps': 0,
                'frame_count': 1
            }
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
    
    def set_property(self, prop_id: int, value: float):
        """Set video capture property"""
        if not self.is_image:
            self.cap.set(prop_id, value)
    
    def resize_frame(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize frame to specified dimensions"""
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    
    def release(self):
        """Release video capture resource"""
        if self.cap is not None:
            self.cap.release()
            logger.info("Video capture released")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
