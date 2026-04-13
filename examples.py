"""
Example usage of fire detection system
"""
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.detector import FireDetector
from src.video_handler import VideoHandler
from src.alert_system import AlertSystem
from src.preprocessor import Preprocessor
from src.utils import setup_logging
from config.settings import get_settings
import cv2

def example_detect_from_image(image_path):
    """Example: Detect fire in a single image"""
    print("Example 1: Detecting fire in image")
    
    setup_logging()
    config = get_settings()
    detector = FireDetector(config)
    preprocessor = Preprocessor()
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return
    
    # Preprocess
    processed = preprocessor.preprocessing_pipeline(image)
    
    # Detect
    fire_detected, regions, result = detector.detect(processed)
    
    print(f"Fire detected: {fire_detected}")
    print(f"Number of regions: {len(regions)}")
    
    if regions:
        for i, region in enumerate(regions):
            print(f"  Region {i+1}: confidence={region['confidence']:.2f}, area={region['area']}")
    
    # Display
    cv2.imshow('Fire Detection Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def example_detect_batch_frames():
    """Example: Detect fire in multiple frames"""
    print("Example 2: Batch detection")
    
    setup_logging()
    config = get_settings()
    detector = FireDetector(config)
    
    import numpy as np
    
    # Create sample frames
    frames = [
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.uint8),
    ]
    
    # Add some fire color to middle frame
    frames[1][100:200, 100:200] = [0, 165, 255]  # Orange
    
    # Detect in batch
    results = detector.detect_batch(frames)
    
    for i, (detected, regions) in enumerate(results):
        print(f"Frame {i}: {'FIRE' if detected else 'CLEAR'} ({len(regions)} regions)")

def example_with_alert_system():
    """Example: Using alert system"""
    print("Example 3: Alert system")
    
    setup_logging()
    config = get_settings()
    alert_system = AlertSystem(config)
    
    import numpy as np
    
    # Create a frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:200, 100:200] = [0, 165, 255]
    
    # Simulate fire regions
    fire_regions = [
        {
            'bbox': (100, 100, 100, 100),
            'confidence': 0.95,
            'area': 10000
        }
    ]
    
    # Trigger alert
    alert = alert_system.trigger_alert(frame, fire_regions, "example_camera")
    
    print(f"Alert triggered: {alert['id']}")
    print(f"Alert time: {alert['timestamp']}")
    print(f"Snapshot saved: {alert['snapshot_path']}")

def example_video_input_handling():
    """Example: Video input handling"""
    print("Example 4: Video handling")
    
    setup_logging()
    
    # Create a temporary test image for demonstration
    import numpy as np
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(tmp.name, test_image)
        
        # Use VideoHandler with image
        with VideoHandler(tmp.name) as handler:
            props = handler.get_properties()
            print(f"Properties: {props}")
            
            frame = handler.read_frame()
            print(f"Frame shape: {frame.shape}")
            
            # Resize
            resized = handler.resize_frame(frame, 320, 240)
            print(f"Resized shape: {resized.shape}")

def example_preprocessing():
    """Example: Image preprocessing"""
    print("Example 5: Preprocessing")
    
    import numpy as np
    
    preprocessor = Preprocessor()
    
    # Create sample image
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # Apply different preprocessing
    resized = preprocessor.resize(image, 320, 240)
    print(f"Resized: {resized.shape}")
    
    normalized = preprocessor.normalize(image)
    print(f"Normalized: min={normalized.min()}, max={normalized.max()}")
    
    blurred = preprocessor.blur(image)
    print(f"Blurred: shape={blurred.shape}")
    
    # Full pipeline
    processed = preprocessor.preprocessing_pipeline(
        image,
        resize_width=320,
        resize_height=240,
        denoise=True,
        equalize=True
    )
    print(f"Processed: shape={processed.shape}")

if __name__ == "__main__":
    """Run examples"""
    print("Fire Detection System - Usage Examples\n")
    
    try:
        # Run examples
        print("=" * 50)
        example_detect_batch_frames()
        
        print("\n" + "=" * 50)
        example_with_alert_system()
        
        print("\n" + "=" * 50)
        example_video_input_handling()
        
        print("\n" + "=" * 50)
        example_preprocessing()
        
        print("\n" + "=" * 50)
        # Uncomment to test with actual image
        # example_detect_from_image("path/to/image.jpg")
        
        print("\nExamples completed!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
