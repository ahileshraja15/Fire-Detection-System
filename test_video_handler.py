"""
Test cases for video handler module
"""
import unittest
import numpy as np
import cv2
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.video_handler import VideoHandler

class TestVideoHandler(unittest.TestCase):
    """Test video handler functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_image_path = Path(self.temp_dir.name) / "test_image.jpg"
        self.test_video_path = Path(self.temp_dir.name) / "test_video.mp4"
        
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(self.test_image_path), test_image)
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.temp_dir.cleanup()
    
    def test_image_loading(self):
        """Test loading image file"""
        handler = VideoHandler(str(self.test_image_path))
        frame = handler.read_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape[:2], (480, 640))
        handler.release()
    
    def test_get_properties(self):
        """Test getting video properties"""
        handler = VideoHandler(str(self.test_image_path))
        props = handler.get_properties()
        
        self.assertIn('width', props)
        self.assertIn('height', props)
        self.assertGreater(props['width'], 0)
        self.assertGreater(props['height'], 0)
        
        handler.release()
    
    def test_resize_frame(self):
        """Test frame resizing"""
        handler = VideoHandler(str(self.test_image_path))
        frame = handler.read_frame()
        
        resized = handler.resize_frame(frame, 320, 240)
        self.assertEqual(resized.shape[:2], (240, 320))
        
        handler.release()
    
    def test_context_manager(self):
        """Test context manager functionality"""
        with VideoHandler(str(self.test_image_path)) as handler:
            frame = handler.read_frame()
            self.assertIsNotNone(frame)

if __name__ == '__main__':
    unittest.main()
