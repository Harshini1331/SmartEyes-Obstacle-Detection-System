#!/usr/bin/env python3
"""
SmartEyes Obstacle Detection System
A computer vision-based obstacle detection system using OpenCV and YOLO.

Author: Harshini
Date: 2024
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartEyesDetector:
    """
    SmartEyes Obstacle Detection System using computer vision techniques.
    """
    
    def __init__(self, confidence_threshold=0.5, nms_threshold=0.4):
        """
        Initialize the SmartEyes detector.
        
        Args:
            confidence_threshold (float): Minimum confidence for detections
            nms_threshold (float): Non-maximum suppression threshold
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.net = None
        self.classes = []
        self.colors = []
        
    def load_model(self, config_path, weights_path, classes_path):
        """
        Load YOLO model and classes.
        
        Args:
            config_path (str): Path to YOLO config file
            weights_path (str): Path to YOLO weights file
            classes_path (str): Path to classes file
        """
        try:
            # Load YOLO network
            self.net = cv2.dnn.readNet(weights_path, config_path)
            
            # Load class names
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Generate random colors for each class
            np.random.seed(42)
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            
            logger.info(f"Model loaded successfully with {len(self.classes)} classes")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def detect_obstacles(self, image):
        """
        Detect obstacles in the given image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            tuple: (processed_image, detections)
        """
        if self.net is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        height, width = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Set input to the network
        self.net.setInput(blob)
        
        # Get output layer names
        output_layer_names = self.net.getUnconnectedOutLayersNames()
        
        # Run forward pass
        outputs = self.net.forward(output_layer_names)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Get bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        # Draw detections
        processed_image = image.copy()
        detections = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                confidence = confidences[i]
                
                # Draw bounding box
                color = [int(c) for c in self.colors[class_id]]
                cv2.rectangle(processed_image, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                label = f"{self.classes[class_id]}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(processed_image, (x, y - label_size[1] - 10), 
                            (x + label_size[0], y), color, -1)
                cv2.putText(processed_image, label, (x, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                detections.append({
                    'class': self.classes[class_id],
                    'confidence': confidence,
                    'bbox': (x, y, w, h)
                })
        
        return processed_image, detections
    
    def process_video(self, video_path, output_path=None):
        """
        Process video file for obstacle detection.
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to output video (optional)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        logger.info(f"Processing video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect obstacles
            processed_frame, detections = self.detect_obstacles(frame)
            
            # Add frame info
            info_text = f"Frame: {frame_count} | Detections: {len(detections)}"
            cv2.putText(processed_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write frame if output is specified
            if writer:
                writer.write(processed_frame)
            
            # Display frame
            cv2.imshow('SmartEyes Detection', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        end_time = time.time()
        logger.info(f"Processing completed. Processed {frame_count} frames in {end_time - start_time:.2f} seconds")
    
    def process_camera(self, camera_index=0):
        """
        Process live camera feed for obstacle detection.
        
        Args:
            camera_index (int): Camera index (default: 0)
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            logger.error(f"Error opening camera: {camera_index}")
            return
        
        logger.info("Starting live detection. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect obstacles
            processed_frame, detections = self.detect_obstacles(frame)
            
            # Add detection info
            info_text = f"Detections: {len(detections)}"
            cv2.putText(processed_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('SmartEyes Live Detection', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Live detection stopped.")

def main():
    """Main function to run the SmartEyes detection system."""
    parser = argparse.ArgumentParser(description='SmartEyes Obstacle Detection System')
    parser.add_argument('--mode', choices=['image', 'video', 'camera'], default='camera',
                       help='Detection mode')
    parser.add_argument('--input', type=str, help='Input image/video path')
    parser.add_argument('--output', type=str, help='Output path for processed video')
    parser.add_argument('--config', type=str, default='yolo/yolov3.cfg',
                       help='YOLO config file path')
    parser.add_argument('--weights', type=str, default='yolo/yolov3.weights',
                       help='YOLO weights file path')
    parser.add_argument('--classes', type=str, default='yolo/coco.names',
                       help='Classes file path')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.4,
                       help='NMS threshold')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = SmartEyesDetector(args.confidence, args.nms)
    
    try:
        # Load model
        detector.load_model(args.config, args.weights, args.classes)
        
        if args.mode == 'image':
            if not args.input:
                logger.error("Input image path required for image mode")
                return
            
            # Process single image
            image = cv2.imread(args.input)
            if image is None:
                logger.error(f"Error loading image: {args.input}")
                return
            
            processed_image, detections = detector.detect_obstacles(image)
            
            # Save result
            output_path = args.output or f"result_{Path(args.input).stem}.jpg"
            cv2.imwrite(output_path, processed_image)
            
            logger.info(f"Processed image saved to: {output_path}")
            logger.info(f"Found {len(detections)} obstacles")
            
            # Display result
            cv2.imshow('SmartEyes Detection Result', processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        elif args.mode == 'video':
            if not args.input:
                logger.error("Input video path required for video mode")
                return
            
            detector.process_video(args.input, args.output)
            
        elif args.mode == 'camera':
            detector.process_camera()
            
    except Exception as e:
        logger.error(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
