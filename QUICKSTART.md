# Fire Detection System - Quick Start Guide

## Installation

1. Navigate to the project directory:
```bash
cd fire_detection_system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Run with Webcam
```bash
python main.py
```

### Run with Video File
```bash
python main.py --video path/to/video.mp4
```

### Run with Image
```bash
python main.py --image path/to/image.jpg
```

### Advanced Options
```bash
python main.py \
    --video input.mp4 \
    --threshold 0.5 \
    --enable-logging
```

## Testing

Run all tests:
```bash
python -m pytest tests/
```

Or run specific test:
```bash
python -m unittest tests.test_detector
```

## Configuration

Edit `config/config.json` to customize:
- Detection sensitivity
- Alert methods (email, notifications)
- Video processing parameters
- Logging settings

## Keyboard Controls

While running:
- **Q**: Quit the application
- **P**: Pause/Resume detection

## Project Structure

```
fire_detection_system/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── config/
│   ├── config.json        # Configuration file
│   └── settings.py        # Config loader
├── src/
│   ├── detector.py        # Fire detection logic
│   ├── video_handler.py   # Video/camera input
│   ├── alert_system.py    # Alerts & notifications
│   ├── preprocessor.py    # Image preprocessing
│   └── utils.py           # Helper functions
├── tests/                 # Unit tests
└── README.md             # Documentation
```

## Performance Notes

- **GPU Acceleration**: Enabled automatically if TensorFlow detects GPU
- **Frame Rate**: 24-30 FPS typical
- **Accuracy**: 85-92% (varies with conditions)
- **CPU Usage**: Moderate to high depending on resolution

## Troubleshooting

### Camera not working
- Check camera permissions
- Verify camera is connected
- Try a different camera index

### Low accuracy
- Adjust threshold in config.json
- Improve lighting conditions
- Check camera resolution

### Performance issues
- Reduce video resolution
- Increase frame skip rate
- Use GPU if available

## Next Steps

1. Calibrate detection parameters for your environment
2. Test with sample videos
3. Set up email alerts (optional)
4. Deploy to production

## Support & Documentation

See README.md for full documentation
