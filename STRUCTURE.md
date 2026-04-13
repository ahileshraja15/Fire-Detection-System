# Fire Detection System - Project Structure

## Directory Layout

### Root Level
- **main.py** - Application entry point
- **requirements.txt** - Python dependencies
- **setup.py** - Package setup script
- **README.md** - Full documentation
- **QUICKSTART.md** - Quick start guide

### `/config`
Configuration management:
- **config.json** - Main configuration file
- **settings.py** - Settings loader and manager

### `/src`
Core modules:
- **detector.py** - Fire detection algorithm
- **video_handler.py** - Video/camera input handling
- **alert_system.py** - Alert and notification system
- **preprocessor.py** - Image preprocessing utilities
- **utils.py** - Helper functions
- **__init__.py** - Package initializer

### `/tests`
Unit tests:
- **test_detector.py** - Tests for FireDetector
- **test_video_handler.py** - Tests for VideoHandler
- **test_utils.py** - Tests for utilities
- **__init__.py** - Test package initializer

### `/data`
Data files (optional):
- sample_video.mp4 - Example video file
- Sample images for testing

### `/models`
Pre-trained models:
- fire_model.pkl - Trained detection model

### `/logs`
Application logs (created at runtime):
- fire_detection.log - Main log file

### `/alerts`
Alert artifacts (created at runtime):
- snapshots/ - Fire detection screenshots

## Key Files Description

### main.py
Main application with FireDetectionApp class:
- Handles webcam, video file, and image inputs
- Coordinates detection, preprocessing, and alerting
- Provides real-time display and statistics

### detector.py
FireDetector class:
- HSV color space analysis
- Contour-based fire detection
- Configurable sensitivity and area thresholds
- Batch processing support

### video_handler.py
VideoHandler class:
- Unified interface for multiple input sources
- Camera, video file, and image support
- Frame resizing and property management
- Context manager support

### alert_system.py
AlertSystem class:
- Fire detection alerts
- Email notifications
- System notifications
- Snapshot saving
- Alert history tracking

### preprocessor.py
Preprocessor class:
- Image resizing
- Noise reduction
- Histogram equalization
- Gamma correction
- Edge detection

### utils.py
Utility functions:
- FPS calculation
- Region validation and analysis
- Fire persistence checking
- Region merging
- Logging setup

### config/settings.py
Settings class:
- JSON configuration loading
- Nested key access
- Default values
- Configuration persistence

## Data Flow

```
Input (Webcam/Video/Image)
        ↓
VideoHandler (read frames)
        ↓
Preprocessor (enhance image)
        ↓
FireDetector (detect fire)
        ↓
AlertSystem (notify if fire)
        ↓
Display/Output
```

## Configuration Schema

```
{
  "detection": {
    "threshold": number (0-1),
    "min_area": number (pixels),
    "max_area": number (pixels),
    "color_range": { h/s/v ranges }
  },
  "video": {
    "skip_frames": number,
    "resize_width": number,
    "resize_height": number,
    "fps": number
  },
  "alerts": {
    "email_enabled": boolean,
    "notification_enabled": boolean,
    "save_snapshot": boolean
  },
  "logging": {
    "level": "INFO|DEBUG|WARNING|ERROR",
    "file": "log file path"
  }
}
```

## Module Dependencies

- **detector.py** → utils.py, config.settings
- **video_handler.py** → (opencv, numpy)
- **alert_system.py** → (email, cv2)
- **preprocessor.py** → (cv2, numpy)
- **main.py** → detector, video_handler, alert_system, preprocessor, utils
- **tests/** → src modules + unittest

## Extensibility Points

1. **Custom Detection Algorithms** - Replace FireDetector.detect()
2. **New Alert Channels** - Add methods to AlertSystem
3. **Additional Preprocessing** - Extend Preprocessor class
4. **Custom Video Sources** - Add to VideoHandler._init_capture()
5. **Configuration Providers** - Subclass Settings class
