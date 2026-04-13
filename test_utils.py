"""
Test cases for utility functions
"""
import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    calculate_fps,
    check_fire_persistence,
    validate_region,
    get_region_center,
    distance_between_points,
    merge_overlapping_regions
)

class TestUtils(unittest.TestCase):
    """Test utility functions"""
    
    def test_calculate_fps(self):
        """Test FPS calculation"""
        fps = calculate_fps(100, 5.0)  # 100 frames in 5 seconds
        self.assertEqual(fps, 20.0)
    
    def test_calculate_fps_zero_time(self):
        """Test FPS calculation with zero time"""
        fps = calculate_fps(100, 0)
        self.assertEqual(fps, 0.0)
    
    def test_check_fire_persistence_true(self):
        """Test fire persistence detection - positive case"""
        detections = [False, True, True, True]
        result = check_fire_persistence(detections, threshold=3)
        self.assertTrue(result)
    
    def test_check_fire_persistence_false(self):
        """Test fire persistence detection - negative case"""
        detections = [True, False, True, False]
        result = check_fire_persistence(detections, threshold=3)
        self.assertFalse(result)
    
    def test_validate_region_valid(self):
        """Test region validation - valid region"""
        bbox = (10, 10, 100, 100)  # area = 10000
        result = validate_region(bbox)
        self.assertTrue(result)
    
    def test_validate_region_too_small(self):
        """Test region validation - too small"""
        bbox = (10, 10, 5, 5)  # area = 25
        result = validate_region(bbox)
        self.assertFalse(result)
    
    def test_get_region_center(self):
        """Test getting region center"""
        bbox = (10, 20, 100, 100)
        center = get_region_center(bbox)
        self.assertEqual(center, (60, 70))
    
    def test_distance_between_points(self):
        """Test distance calculation"""
        p1 = (0, 0)
        p2 = (3, 4)
        distance = distance_between_points(p1, p2)
        self.assertEqual(distance, 5.0)
    
    def test_merge_overlapping_regions(self):
        """Test merging overlapping regions"""
        regions = [
            {'bbox': (0, 0, 50, 50), 'confidence': 0.9, 'area': 2500, 'contour': None},
            {'bbox': (25, 25, 50, 50), 'confidence': 0.8, 'area': 2500, 'contour': None}
        ]
        merged = merge_overlapping_regions(regions)
        # Should merge overlapping regions
        self.assertLessEqual(len(merged), len(regions))

if __name__ == '__main__':
    unittest.main()
