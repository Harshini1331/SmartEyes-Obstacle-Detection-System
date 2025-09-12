"""
Video utility functions for SmartEyes Obstacle Detection System.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Video processing utilities."""
    
    def __init__(self, input_path, output_path=None):
        """
        Initialize video processor.
        
        Args:
            input_path (str): Path to input video
            output_path (str): Path to output video (optional)
        """
        self.input_path = input_path
        self.output_path = output_path
        self.cap = None
        self.writer = None
        self.fps = 0
        self.width = 0
        self.height = 0
        self.frame_count = 0
        self.total_frames = 0
        
    def open_video(self):
        """Open video file for reading."""
        try:
            self.cap = cv2.VideoCapture(self.input_path)
            if not self.cap.isOpened():
                logger.error(f"Failed to open video: {self.input_path}")
                return False
            
            # Get video properties
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video opened: {self.width}x{self.height} @ {self.fps} FPS")
            return True
            
        except Exception as e:
            logger.error(f"Error opening video: {e}")
            return False
    
    def setup_writer(self, codec='mp4v'):
        """Setup video writer for output."""
        if not self.output_path:
            return False
        
        try:
            # Create output directory
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (self.width, self.height)
            )
            
            if not self.writer.isOpened():
                logger.error(f"Failed to setup video writer: {self.output_path}")
                return False
            
            logger.info(f"Video writer setup: {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up video writer: {e}")
            return False
    
    def read_frame(self):
        """Read next frame from video."""
        if not self.cap or not self.cap.isOpened():
            return None, False
        
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
        return frame, ret
    
    def write_frame(self, frame):
        """Write frame to output video."""
        if self.writer and self.writer.isOpened():
            self.writer.write(frame)
    
    def get_progress(self):
        """Get processing progress percentage."""
        if self.total_frames == 0:
            return 0
        return (self.frame_count / self.total_frames) * 100
    
    def close(self):
        """Close video files."""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        logger.info("Video files closed")

def extract_frames(video_path, output_dir, frame_interval=1):
    """
    Extract frames from video.
    
    Args:
        video_path (str): Path to input video
        output_dir (str): Directory to save frames
        frame_interval (int): Extract every nth frame
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            output_path = Path(output_dir) / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    logger.info(f"Extracted {saved_count} frames from {frame_count} total frames")

def create_video_from_frames(frames_dir, output_path, fps=30):
    """
    Create video from sequence of frames.
    
    Args:
        frames_dir (str): Directory containing frames
        output_path (str): Output video path
        fps (int): Frames per second
    """
    frames_path = Path(frames_dir)
    frame_files = sorted(frames_path.glob("*.jpg"))
    
    if not frame_files:
        logger.error(f"No frame files found in: {frames_dir}")
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    height, width = first_frame.shape[:2]
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        logger.error(f"Failed to create video writer: {output_path}")
        return
    
    # Write frames
    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        if frame is not None:
            writer.write(frame)
    
    writer.release()
    logger.info(f"Video created: {output_path}")

def get_video_info(video_path):
    """
    Get video information.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        dict: Video information
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        'codec': cap.get(cv2.CAP_PROP_FOURCC)
    }
    
    cap.release()
    return info

def resize_video(input_path, output_path, target_size, maintain_aspect_ratio=True):
    """
    Resize video to target size.
    
    Args:
        input_path (str): Input video path
        output_path (str): Output video path
        target_size (tuple): Target size (width, height)
        maintain_aspect_ratio (bool): Whether to maintain aspect ratio
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {input_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate new dimensions
    if maintain_aspect_ratio:
        scale = min(target_size[0] / width, target_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        new_width, new_height = target_size
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    if not writer.isOpened():
        logger.error(f"Failed to create video writer: {output_path}")
        cap.release()
        return
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        writer.write(resized_frame)
    
    cap.release()
    writer.release()
    logger.info(f"Video resized and saved: {output_path}")

def add_timestamp_to_video(input_path, output_path, font_scale=0.7, color=(255, 255, 255)):
    """
    Add timestamp to video frames.
    
    Args:
        input_path (str): Input video path
        output_path (str): Output video path
        font_scale (float): Font scale for timestamp
        color (tuple): BGR color for timestamp
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {input_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        logger.error(f"Failed to create video writer: {output_path}")
        cap.release()
        return
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate timestamp
        current_time = time.time() - start_time
        timestamp = f"Time: {current_time:.2f}s | Frame: {frame_count}"
        
        # Add timestamp to frame
        cv2.putText(frame, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        
        writer.write(frame)
        frame_count += 1
    
    cap.release()
    writer.release()
    logger.info(f"Video with timestamp saved: {output_path}")

def calculate_fps(process_func, *args, **kwargs):
    """
    Calculate FPS for a processing function.
    
    Args:
        process_func: Function to measure
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        float: Average FPS
    """
    start_time = time.time()
    frame_count = 0
    
    # Run for a short duration to measure FPS
    while time.time() - start_time < 5.0:  # Run for 5 seconds
        process_func(*args, **kwargs)
        frame_count += 1
    
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    
    logger.info(f"Average FPS: {fps:.2f}")
    return fps
