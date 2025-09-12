#!/usr/bin/env python3
"""
Batch processing example for SmartEyes Obstacle Detection System.
This example demonstrates how to process multiple images or videos in batch.
"""

import sys
import os
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from smart_eyes_detection import SmartEyesDetector
from utils.image_utils import load_image, save_image
from utils.video_utils import VideoProcessor

class BatchProcessor:
    """Batch processor for multiple images and videos."""
    
    def __init__(self, confidence_threshold=0.5, nms_threshold=0.4, max_workers=4):
        """
        Initialize batch processor.
        
        Args:
            confidence_threshold (float): Confidence threshold for detection
            nms_threshold (float): NMS threshold for detection
            max_workers (int): Maximum number of worker threads
        """
        self.detector = SmartEyesDetector(confidence_threshold, nms_threshold)
        self.max_workers = max_workers
        self.results = []
        
    def load_model(self, config_path, weights_path, classes_path):
        """Load YOLO model."""
        self.detector.load_model(config_path, weights_path, classes_path)
    
    def process_image(self, image_path, output_dir=None):
        """
        Process a single image.
        
        Args:
            image_path (str): Path to input image
            output_dir (str): Output directory for processed image
            
        Returns:
            dict: Processing result
        """
        try:
            # Load image
            image = load_image(image_path)
            if image is None:
                return {
                    'input_path': image_path,
                    'status': 'error',
                    'error': 'Failed to load image',
                    'detections': []
                }
            
            # Detect obstacles
            processed_image, detections = self.detector.detect_obstacles(image)
            
            # Save processed image if output directory is specified
            if output_dir:
                output_path = Path(output_dir) / f"processed_{Path(image_path).name}"
                save_image(processed_image, str(output_path))
            
            return {
                'input_path': image_path,
                'status': 'success',
                'detections': detections,
                'detection_count': len(detections)
            }
            
        except Exception as e:
            return {
                'input_path': image_path,
                'status': 'error',
                'error': str(e),
                'detections': []
            }
    
    def process_video(self, video_path, output_dir=None):
        """
        Process a single video.
        
        Args:
            video_path (str): Path to input video
            output_dir (str): Output directory for processed video
            
        Returns:
            dict: Processing result
        """
        try:
            # Setup output path
            output_path = None
            if output_dir:
                output_path = Path(output_dir) / f"processed_{Path(video_path).name}"
                output_path = str(output_path)
            
            # Process video
            processor = VideoProcessor(video_path, output_path)
            if not processor.open_video():
                return {
                    'input_path': video_path,
                    'status': 'error',
                    'error': 'Failed to open video',
                    'detections': []
                }
            
            processor.setup_writer()
            
            total_detections = 0
            frame_count = 0
            
            while True:
                frame, ret = processor.read_frame()
                if not ret:
                    break
                
                # Detect obstacles in frame
                processed_frame, detections = self.detector.detect_obstacles(frame)
                total_detections += len(detections)
                
                # Write processed frame
                processor.write_frame(processed_frame)
                frame_count += 1
            
            processor.close()
            
            return {
                'input_path': video_path,
                'status': 'success',
                'detections': [],
                'detection_count': total_detections,
                'frame_count': frame_count
            }
            
        except Exception as e:
            return {
                'input_path': video_path,
                'status': 'error',
                'error': str(e),
                'detections': []
            }
    
    def process_images_batch(self, image_paths, output_dir=None):
        """
        Process multiple images in parallel.
        
        Args:
            image_paths (list): List of image paths
            output_dir (str): Output directory for processed images
            
        Returns:
            list: List of processing results
        """
        print(f"Processing {len(image_paths)} images...")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_image, path, output_dir): path
                for path in image_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                result = future.result()
                results.append(result)
                
                # Print progress
                status = "✓" if result['status'] == 'success' else "✗"
                print(f"{status} {Path(result['input_path']).name}")
        
        return results
    
    def process_videos_batch(self, video_paths, output_dir=None):
        """
        Process multiple videos in parallel.
        
        Args:
            video_paths (list): List of video paths
            output_dir (str): Output directory for processed videos
            
        Returns:
            list: List of processing results
        """
        print(f"Processing {len(video_paths)} videos...")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_video, path, output_dir): path
                for path in video_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                result = future.result()
                results.append(result)
                
                # Print progress
                status = "✓" if result['status'] == 'success' else "✗"
                print(f"{status} {Path(result['input_path']).name}")
        
        return results
    
    def save_results(self, results, output_file):
        """
        Save processing results to JSON file.
        
        Args:
            results (list): List of processing results
            output_file (str): Output JSON file path
        """
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    def print_summary(self, results):
        """Print processing summary."""
        total = len(results)
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = total - successful
        
        total_detections = sum(r.get('detection_count', 0) for r in results if r['status'] == 'success')
        
        print(f"\n{'='*50}")
        print(f"BATCH PROCESSING SUMMARY")
        print(f"{'='*50}")
        print(f"Total files processed: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total detections: {total_detections}")
        print(f"Success rate: {successful/total*100:.1f}%")
        print(f"{'='*50}")

def find_media_files(directory, extensions):
    """
    Find all media files in directory with given extensions.
    
    Args:
        directory (str): Directory to search
        extensions (list): List of file extensions
        
    Returns:
        list: List of found file paths
    """
    directory = Path(directory)
    files = []
    
    for ext in extensions:
        files.extend(directory.glob(f"**/*{ext}"))
    
    return [str(f) for f in files]

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='SmartEyes Batch Processing')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing images/videos')
    parser.add_argument('--output', type=str, help='Output directory for processed files')
    parser.add_argument('--type', choices=['image', 'video', 'both'], default='both',
                       help='Type of files to process')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of worker threads')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--results', type=str, default='batch_results.json',
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Find media files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    all_files = find_media_files(args.input, image_extensions + video_extensions)
    
    image_files = [f for f in all_files if Path(f).suffix.lower() in image_extensions]
    video_files = [f for f in all_files if Path(f).suffix.lower() in video_extensions]
    
    print(f"Found {len(image_files)} images and {len(video_files)} videos")
    
    # Initialize processor
    processor = BatchProcessor(
        confidence_threshold=args.confidence,
        max_workers=args.workers
    )
    
    # Load model
    try:
        processor.load_model('yolo/yolov3.cfg', 'yolo/yolov3.weights', 'yolo/coco.names')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process files
    start_time = time.time()
    results = []
    
    if args.type in ['image', 'both'] and image_files:
        print(f"\nProcessing {len(image_files)} images...")
        image_results = processor.process_images_batch(image_files, args.output)
        results.extend(image_results)
    
    if args.type in ['video', 'both'] and video_files:
        print(f"\nProcessing {len(video_files)} videos...")
        video_results = processor.process_videos_batch(video_files, args.output)
        results.extend(video_results)
    
    end_time = time.time()
    
    # Print summary
    processor.print_summary(results)
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    
    # Save results
    processor.save_results(results, args.results)

if __name__ == "__main__":
    main()
