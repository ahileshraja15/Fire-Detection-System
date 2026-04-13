"""
Test cases for fire detector module
"""
import unittest
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import FireDetector
from config.settings import Settings

class TestFireDetector(unittest.TestCase):
    """Test fire detector functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = FireDetector()
        self.test_image_size = (480, 640, 3)
    
    def test_detector_initialization(self):
        """Test detector initializes correctly"""
        self.assertIsNotNone(self.detector)
        self.assertGreater(self.detector.detection_threshold, 0)
        self.assertGreater(self.detector.min_area, 0)
    
    def test_detect_empty_frame(self):
        """Test detection with empty/None frame"""
        detected, regions, frame = self.detector.detect(None)
        self.assertFalse(detected)
        self.assertEqual(len(regions), 0)
    
    def test_detect_no_fire(self):
        """Test detection on frame without fire"""
        frame = np.zeros(self.test_image_size, dtype=np.uint8)
        detected, regions, result = self.detector.detect(frame)
        self.assertFalse(detected)
        self.assertEqual(len(regions), 0)
    
    def test_detect_with_fire_colors(self):
        """Test detection with fire-colored pixels"""
        frame = np.zeros(self.test_image_size, dtype=np.uint8)
        # Add fire-colored region (orange/yellow in BGR)
        frame[100:200, 100:200] = [0, 165, 255]  # BGR: orange
        
        detected, regions, result = self.detector.detect(frame)
        self.assertTrue(detected)
        self.assertGreater(len(regions), 0)
    
    def test_detect_batch(self):
        """Test batch detection"""
        frames = [np.zeros(self.test_image_size, dtype=np.uint8) for _ in range(5)]
        results = self.detector.detect_batch(frames)
        self.assertEqual(len(results), 5)
    
    def test_region_analysis(self):
        """Test region analysis"""
        frame = np.zeros(self.test_image_size, dtype=np.uint8)
        frame[100:200, 100:200] = [0, 165, 255]
        
        detected, regions, _ = self.detector.detect(frame)
        if regions:
            self.assertIn('bbox', regions[0])
            self.assertIn('area', regions[0])
            self.assertIn('confidence', regions[0])

class TestSettings(unittest.TestCase):
    """Test configuration settings"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.settings = Settings()
    
    def test_settings_load(self):
        """Test settings load correctly"""
        self.assertIsNotNone(self.settings.config)
    
    def test_get_setting(self):
        """Test get method"""
        threshold = self.settings.get('detection.threshold')
        self.assertIsNotNone(threshold)
    
    def test_get_default_value(self):
        """Test get with default value"""
        value = self.settings.get('nonexistent.key', 'default')
        self.assertEqual(value, 'default')
    
    def test_set_setting(self):
        """Test set method"""
        self.settings.set('test.value', 123)
        result = self.settings.get('test.value')
        self.assertEqual(result, 123)

if __name__ == '__main__':
    unittest.main()
