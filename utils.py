"""
Utility functions for fire detection system
"""
import cv2
import numpy as np
import logging
from typing import Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_logging(log_file: str = None, level: str = "INFO"):
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to {log_file}")

def draw_fps(frame: np.ndarray, fps: float, position: Tuple[int, int] = (10, 30)):
    """Draw FPS counter on frame"""
    cv2.putText(frame, f"FPS: {fps:.1f}", position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

def draw_text(frame: np.ndarray, text: str, position: Tuple[int, int],
              color: Tuple[int, int, int] = (255, 255, 255),
              font_size: float = 0.7):
    """Draw text on frame"""
    cv2.putText(frame, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2)
    return frame

def draw_info_panel(frame: np.ndarray, info: dict, position: Tuple[int, int] = (10, 30)):
    """Draw information panel on frame"""
    y_offset = position[1]
    for key, value in info.items():
        text = f"{key}: {value}"
        cv2.putText(frame, text, (position[0], y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 25
    return frame

def calculate_fps(frame_count: int, elapsed_time: float) -> float:
    """Calculate FPS from frame count and elapsed time"""
    if elapsed_time > 0:
        return frame_count / elapsed_time
    return 0.0

def check_fire_persistence(detections: list, threshold: int = 3) -> bool:
    """
    Check if fire is detected consistently across frames
    
    Args:
        detections: List of boolean detection results
        threshold: Number of consecutive detections required
        
    Returns:
        True if fire detected persistently
    """
    if len(detections) < threshold:
        return False
    
    return all(detections[-threshold:])

def validate_region(bbox: Tuple[int, int, int, int], 
                   min_area: int = 100,
                   max_area: int = 50000) -> bool:
    """Validate fire region boundaries"""
    x, y, w, h = bbox
    area = w * h
    return min_area <= area <= max_area

def get_region_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Get center point of bounding box"""
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)

def distance_between_points(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def merge_overlapping_regions(regions: list, overlap_threshold: float = 0.3) -> list:
    """Merge overlapping fire regions"""
    if not regions:
        return []
    
    merged = []
    used = set()
    
    for i, region1 in enumerate(regions):
        if i in used:
            continue
        
        x1, y1, w1, h1 = region1['bbox']
        merged_region = region1.copy()
        
        for j, region2 in enumerate(regions[i+1:], start=i+1):
            if j in used:
                continue
            
            x2, y2, w2, h2 = region2['bbox']
            
            # Calculate IoU (Intersection over Union)
            intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * \
                         max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            union = w1 * h1 + w2 * h2 - intersection
            iou = intersection / union if union > 0 else 0
            
            if iou > overlap_threshold:
                # Merge regions
                x_new = min(x1, x2)
                y_new = min(y1, y2)
                w_new = max(x1 + w1, x2 + w2) - x_new
                h_new = max(y1 + h1, y2 + h2) - y_new
                
                merged_region['bbox'] = (x_new, y_new, w_new, h_new)
                merged_region['confidence'] = max(merged_region['confidence'],
                                                 region2['confidence'])
                used.add(j)
        
        merged.append(merged_region)
    
    return merged

def save_detection_log(filepath: str, frame_num: int, detected: bool, 
                      region_count: int = 0):
    """Save detection event to log file"""
    with open(filepath, 'a') as f:
        status = "FIRE" if detected else "CLEAR"
        f.write(f"Frame {frame_num}: {status} ({region_count} regions)\n")
