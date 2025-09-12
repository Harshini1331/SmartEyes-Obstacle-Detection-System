#!/usr/bin/env python3
"""
Basic detection example for SmartEyes Obstacle Detection System.
This example demonstrates how to use the SmartEyes detector for basic obstacle detection.
"""

import sys
import os
import cv2
import argparse
from pathlib import Path

# Add parent directory to path to import SmartEyes modules
sys.path.append(str(Path(__file__).parent.parent))

from smart_eyes_detection import SmartEyesDetector
from utils.image_utils import load_image, save_image, draw_bounding_box
from utils.video_utils import VideoProcessor

def detect_in_image(image_path, output_path=None, confidence=0.5):
    """
    Detect obstacles in a single image.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save output image
        confidence (float): Confidence threshold
    """
    # Initialize detector
    detector = SmartEyesDetector(confidence_threshold=confidence)
    
    # Load model (assuming YOLO files are in yolo/ directory)
    try:
        detector.load_model('yolo/yolov3.cfg', 'yolo/yolov3.weights', 'yolo/coco.names')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure YOLO model files are in the yolo/ directory")
        return
    
    # Load image
    image = load_image(image_path)
    if image is None:
        return
    
    print(f"Processing image: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Detect obstacles
    processed_image, detections = detector.detect_obstacles(image)
    
    # Print detection results
    print(f"\nFound {len(detections)} obstacles:")
    for i, detection in enumerate(detections):
        print(f"  {i+1}. {detection['class']} (confidence: {detection['confidence']:.2f})")
    
    # Save result
    if output_path:
        save_image(processed_image, output_path)
        print(f"Result saved to: {output_path}")
    
    # Display result
    cv2.imshow('SmartEyes Detection Result', processed_image)
    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_in_video(video_path, output_path=None, confidence=0.5):
    """
    Detect obstacles in a video.
    
    Args:
        video_path (str): Path to input video
        output_path (str): Path to save output video
        confidence (float): Confidence threshold
    """
    # Initialize detector
    detector = SmartEyesDetector(confidence_threshold=confidence)
    
    # Load model
    try:
        detector.load_model('yolo/yolov3.cfg', 'yolo/yolov3.weights', 'yolo/coco.names')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure YOLO model files are in the yolo/ directory")
        return
    
    # Process video
    print(f"Processing video: {video_path}")
    detector.process_video(video_path, output_path)
    
    if output_path:
        print(f"Processed video saved to: {output_path}")

def detect_live_camera(camera_index=0, confidence=0.5):
    """
    Detect obstacles using live camera feed.
    
    Args:
        camera_index (int): Camera index
        confidence (float): Confidence threshold
    """
    # Initialize detector
    detector = SmartEyesDetector(confidence_threshold=confidence)
    
    # Load model
    try:
        detector.load_model('yolo/yolov3.cfg', 'yolo/yolov3.weights', 'yolo/coco.names')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure YOLO model files are in the yolo/ directory")
        return
    
    # Process live camera
    print(f"Starting live detection with camera {camera_index}")
    print("Press 'q' to quit")
    detector.process_camera(camera_index)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='SmartEyes Basic Detection Example')
    parser.add_argument('--mode', choices=['image', 'video', 'camera'], required=True,
                       help='Detection mode')
    parser.add_argument('--input', type=str, help='Input file path (for image/video mode)')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (for camera mode)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    if args.mode == 'image':
        if not args.input:
            print("Error: Input image path required for image mode")
            return
        detect_in_image(args.input, args.output, args.confidence)
        
    elif args.mode == 'video':
        if not args.input:
            print("Error: Input video path required for video mode")
            return
        detect_in_video(args.input, args.output, args.confidence)
        
    elif args.mode == 'camera':
        detect_live_camera(args.camera, args.confidence)

if __name__ == "__main__":
    main()
