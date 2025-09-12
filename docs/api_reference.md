# API Reference

This document provides detailed information about the SmartEyes Obstacle Detection System API.

## SmartEyesDetector Class

The main class for obstacle detection functionality.

### Constructor

```python
SmartEyesDetector(confidence_threshold=0.5, nms_threshold=0.4)
```

**Parameters:**
- `confidence_threshold` (float): Minimum confidence for detections (0.0-1.0)
- `nms_threshold` (float): Non-maximum suppression threshold (0.0-1.0)

### Methods

#### load_model(config_path, weights_path, classes_path)

Load YOLO model and classes.

**Parameters:**
- `config_path` (str): Path to YOLO config file (.cfg)
- `weights_path` (str): Path to YOLO weights file (.weights)
- `classes_path` (str): Path to classes file (.names)

**Raises:**
- `Exception`: If model loading fails

**Example:**
```python
detector = SmartEyesDetector()
detector.load_model('yolo/yolov3.cfg', 'yolo/yolov3.weights', 'yolo/coco.names')
```

#### detect_obstacles(image)

Detect obstacles in the given image.

**Parameters:**
- `image` (numpy.ndarray): Input image (BGR format)

**Returns:**
- `tuple`: (processed_image, detections)
  - `processed_image` (numpy.ndarray): Image with bounding boxes drawn
  - `detections` (list): List of detection dictionaries

**Detection Dictionary Format:**
```python
{
    'class': str,           # Object class name
    'confidence': float,    # Confidence score (0.0-1.0)
    'bbox': tuple          # Bounding box (x, y, w, h)
}
```

**Raises:**
- `ValueError`: If model is not loaded

**Example:**
```python
image = cv2.imread('input.jpg')
processed_image, detections = detector.detect_obstacles(image)

for detection in detections:
    print(f"Found {detection['class']} with confidence {detection['confidence']:.2f}")
```

#### process_video(video_path, output_path=None)

Process video file for obstacle detection.

**Parameters:**
- `video_path` (str): Path to input video file
- `output_path` (str, optional): Path to save processed video

**Example:**
```python
detector.process_video('input.mp4', 'output.mp4')
```

#### process_camera(camera_index=0)

Process live camera feed for obstacle detection.

**Parameters:**
- `camera_index` (int): Camera index (default: 0)

**Example:**
```python
detector.process_camera(0)  # Use default camera
```

## Image Utility Functions

### load_image(image_path)

Load an image from file path.

**Parameters:**
- `image_path` (str): Path to the image file

**Returns:**
- `numpy.ndarray` or `None`: Loaded image or None if failed

**Example:**
```python
image = load_image('path/to/image.jpg')
if image is not None:
    print(f"Image loaded: {image.shape}")
```

### save_image(image, output_path, quality=95)

Save an image to file.

**Parameters:**
- `image` (numpy.ndarray): Image to save
- `output_path` (str): Output file path
- `quality` (int): JPEG quality (1-100)

**Example:**
```python
save_image(processed_image, 'output.jpg', quality=90)
```

### resize_image(image, target_size, maintain_aspect_ratio=True)

Resize an image to target size.

**Parameters:**
- `image` (numpy.ndarray): Input image
- `target_size` (tuple): Target size (width, height)
- `maintain_aspect_ratio` (bool): Whether to maintain aspect ratio

**Returns:**
- `numpy.ndarray`: Resized image

**Example:**
```python
resized = resize_image(image, (640, 480), maintain_aspect_ratio=True)
```

### draw_bounding_box(image, bbox, label, confidence, color=(0, 255, 0), thickness=2)

Draw bounding box with label on image.

**Parameters:**
- `image` (numpy.ndarray): Input image
- `bbox` (tuple): Bounding box (x, y, w, h)
- `label` (str): Label text
- `confidence` (float): Confidence score
- `color` (tuple): BGR color
- `thickness` (int): Line thickness

**Returns:**
- `numpy.ndarray`: Image with bounding box drawn

**Example:**
```python
result = draw_bounding_box(image, (100, 100, 50, 50), "person", 0.85)
```

## Video Utility Functions

### VideoProcessor Class

Utility class for video processing operations.

#### Constructor

```python
VideoProcessor(input_path, output_path=None)
```

**Parameters:**
- `input_path` (str): Path to input video
- `output_path` (str, optional): Path to output video

#### Methods

##### open_video()

Open video file for reading.

**Returns:**
- `bool`: True if successful, False otherwise

##### setup_writer(codec='mp4v')

Setup video writer for output.

**Parameters:**
- `codec` (str): Video codec

**Returns:**
- `bool`: True if successful, False otherwise

##### read_frame()

Read next frame from video.

**Returns:**
- `tuple`: (frame, success_flag)

##### write_frame(frame)

Write frame to output video.

**Parameters:**
- `frame` (numpy.ndarray): Frame to write

##### close()

Close video files.

### get_video_info(video_path)

Get video information.

**Parameters:**
- `video_path` (str): Path to video file

**Returns:**
- `dict` or `None`: Video information dictionary

**Video Info Dictionary:**
```python
{
    'width': int,        # Video width
    'height': int,       # Video height
    'fps': float,        # Frames per second
    'frame_count': int,  # Total frame count
    'duration': float,   # Duration in seconds
    'codec': int         # Codec fourcc
}
```

**Example:**
```python
info = get_video_info('video.mp4')
print(f"Video: {info['width']}x{info['height']} @ {info['fps']} FPS")
```

## Configuration

### Configuration File Format

The system uses YAML configuration files. Default configuration is in `config/default_config.yaml`.

**Example Configuration:**
```yaml
model:
  config_path: "yolo/yolov3.cfg"
  weights_path: "yolo/yolov3.weights"
  classes_path: "yolo/coco.names"
  input_size: 416

detection:
  confidence_threshold: 0.5
  nms_threshold: 0.4
  max_detections: 100

display:
  show_confidence: true
  show_class_names: true
  font_scale: 0.5
  font_thickness: 2
```

### Loading Configuration

```python
import yaml

with open('config/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use configuration
detector = SmartEyesDetector(
    confidence_threshold=config['detection']['confidence_threshold'],
    nms_threshold=config['detection']['nms_threshold']
)
```

## Command Line Interface

### Basic Usage

```bash
# Live camera detection
python smart_eyes_detection.py --mode camera

# Image processing
python smart_eyes_detection.py --mode image --input image.jpg --output result.jpg

# Video processing
python smart_eyes_detection.py --mode video --input video.mp4 --output processed.mp4
```

### Advanced Usage

```bash
# Custom model and parameters
python smart_eyes_detection.py \
    --mode camera \
    --config custom_model.cfg \
    --weights custom_model.weights \
    --classes custom_classes.names \
    --confidence 0.6 \
    --nms 0.3
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | camera | Detection mode (image/video/camera) |
| `--input` | str | - | Input file path |
| `--output` | str | - | Output file path |
| `--config` | str | yolo/yolov3.cfg | YOLO config file |
| `--weights` | str | yolo/yolov3.weights | YOLO weights file |
| `--classes` | str | yolo/coco.names | Classes file |
| `--confidence` | float | 0.5 | Confidence threshold |
| `--nms` | float | 0.4 | NMS threshold |

## Error Handling

### Common Exceptions

#### ValueError
Raised when invalid parameters are provided or model is not loaded.

```python
try:
    detector.detect_obstacles(image)
except ValueError as e:
    print(f"Error: {e}")
```

#### FileNotFoundError
Raised when model files or input files are not found.

```python
try:
    detector.load_model('yolo/yolov3.cfg', 'yolo/yolov3.weights', 'yolo/coco.names')
except FileNotFoundError as e:
    print(f"Model file not found: {e}")
```

#### cv2.error
Raised by OpenCV when video/image operations fail.

```python
try:
    cap = cv2.VideoCapture('video.mp4')
except cv2.error as e:
    print(f"OpenCV error: {e}")
```

## Performance Optimization

### GPU Acceleration

Enable GPU support for faster processing:

```python
# Check GPU availability
import cv2
print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")

# Use GPU-enabled OpenCV
# Install: pip install opencv-python-gpu
```

### Memory Management

For large images or videos:

```python
# Process in chunks
def process_large_video(video_path, chunk_size=100):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        frames = []
        for _ in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        if not frames:
            break
        
        # Process chunk
        for frame in frames:
            processed_frame, detections = detector.detect_obstacles(frame)
            # Save or display result
        
        frame_count += len(frames)
    
    cap.release()
```

### Threading

For batch processing:

```python
from concurrent.futures import ThreadPoolExecutor

def process_images_parallel(image_paths, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, path) for path in image_paths]
        results = [future.result() for future in futures]
    return results
```

## Examples

### Complete Detection Pipeline

```python
import cv2
from smart_eyes_detection import SmartEyesDetector
from utils.image_utils import load_image, save_image

# Initialize detector
detector = SmartEyesDetector(confidence_threshold=0.5)

# Load model
detector.load_model('yolo/yolov3.cfg', 'yolo/yolov3.weights', 'yolo/coco.names')

# Process image
image = load_image('input.jpg')
if image is not None:
    processed_image, detections = detector.detect_obstacles(image)
    
    # Print results
    print(f"Found {len(detections)} objects:")
    for detection in detections:
        print(f"- {detection['class']}: {detection['confidence']:.2f}")
    
    # Save result
    save_image(processed_image, 'output.jpg')
```

### Custom Detection Classes

```python
# Filter detections by class
def filter_detections(detections, target_classes):
    return [d for d in detections if d['class'] in target_classes]

# Process with filtered classes
detections = detector.detect_obstacles(image)
person_detections = filter_detections(detections, ['person', 'bicycle', 'car'])
```

### Real-time Processing with FPS

```python
import time

def process_with_fps(detector, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    fps_counter = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, detections = detector.detect_obstacles(frame)
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter % 30 == 0:
            elapsed = time.time() - start_time
            fps = fps_counter / elapsed
            print(f"FPS: {fps:.2f}")
        
        # Display frame
        cv2.imshow('Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```
