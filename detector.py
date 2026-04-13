"""
Main fire detection module
"""
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List
from config.settings import get_settings

logger = logging.getLogger(__name__)

class FireDetector:
    """Fire detection using color space analysis and contour detection"""
    
    def __init__(self, config=None):
        """Initialize fire detector with configuration"""
        self.config = config or get_settings()
        self.detection_threshold = self.config.get('detection.threshold', 0.5)
        self.min_area = self.config.get('detection.min_area', 500)
        self.max_area = self.config.get('detection.max_area', 50000)
        self.flame_color_range = self._get_color_range()
        
    def _get_color_range(self):
        """Get HSV color range for fire detection"""
        color_config = self.config.get('detection.color_range', {})
        return {
            'h_min': color_config.get('h_min', 0),
            'h_max': color_config.get('h_max', 25),
            's_min': color_config.get('s_min', 50),
            's_max': color_config.get('s_max', 255),
            'v_min': color_config.get('v_min', 100),
            'v_max': color_config.get('v_max', 255)
        }
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, List[dict], np.ndarray]:
        """
        Detect fire in a frame
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (fire_detected, fire_regions, processed_frame)
        """
        if frame is None or frame.size == 0:
            return False, [], frame
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for flame colors
        lower = np.array([
            self.flame_color_range['h_min'],
            self.flame_color_range['s_min'],
            self.flame_color_range['v_min']
        ])
        upper = np.array([
            self.flame_color_range['h_max'],
            self.flame_color_range['s_max'],
            self.flame_color_range['v_max']
        ])
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours
        fire_regions = self._analyze_contours(contours, frame)
        fire_detected = len(fire_regions) > 0
        
        # Draw results on frame
        result_frame = frame.copy()
        for region in fire_regions:
            x, y, w, h = region['bbox']
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(result_frame, f"FIRE ({region['confidence']:.2f})", 
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return fire_detected, fire_regions, result_frame
    
    def _analyze_contours(self, contours, frame) -> List[dict]:
        """Analyze contours to identify fire regions"""
        fire_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate confidence based on area and shape
            confidence = min(1.0, area / (frame.shape[0] * frame.shape[1]) * 100)
            
            if confidence > self.detection_threshold:
                fire_regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'confidence': confidence,
                    'contour': contour
                })
        
        return fire_regions
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[Tuple[bool, List[dict]]]:
        """Detect fire in multiple frames"""
        results = []
        for frame in frames:
            detected, regions, _ = self.detect(frame)
            results.append((detected, regions))
        return results
