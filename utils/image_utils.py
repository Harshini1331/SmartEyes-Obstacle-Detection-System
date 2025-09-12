"""
Image utility functions for SmartEyes Obstacle Detection System.
"""

import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_image(image_path):
    """
    Load an image from file path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Loaded image or None if failed
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        logger.info(f"Successfully loaded image: {image_path}")
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

def save_image(image, output_path, quality=95):
    """
    Save an image to file.
    
    Args:
        image (numpy.ndarray): Image to save
        output_path (str): Output file path
        quality (int): JPEG quality (1-100)
    """
    try:
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Determine compression parameters based on file extension
        ext = Path(output_path).suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif ext == '.png':
            cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(output_path, image)
        
        logger.info(f"Image saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {e}")

def resize_image(image, target_size, maintain_aspect_ratio=True):
    """
    Resize an image to target size.
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Target size (width, height)
        maintain_aspect_ratio (bool): Whether to maintain aspect ratio
        
    Returns:
        numpy.ndarray: Resized image
    """
    if maintain_aspect_ratio:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas with target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Center the resized image on canvas
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def preprocess_image(image, target_size=(416, 416)):
    """
    Preprocess image for YOLO detection.
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Target size for YOLO
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Resize image
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values
    normalized = rgb_image.astype(np.float32) / 255.0
    
    return normalized

def draw_bounding_box(image, bbox, label, confidence, color=(0, 255, 0), thickness=2):
    """
    Draw bounding box with label on image.
    
    Args:
        image (numpy.ndarray): Input image
        bbox (tuple): Bounding box (x, y, w, h)
        label (str): Label text
        confidence (float): Confidence score
        color (tuple): BGR color
        thickness (int): Line thickness
        
    Returns:
        numpy.ndarray: Image with bounding box drawn
    """
    x, y, w, h = bbox
    
    # Draw bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    
    # Prepare label text
    label_text = f"{label}: {confidence:.2f}"
    
    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)
    
    # Draw label background
    cv2.rectangle(image, (x, y - text_height - baseline), 
                 (x + text_width, y), color, -1)
    
    # Draw label text
    cv2.putText(image, label_text, (x, y - baseline), 
               font, font_scale, (255, 255, 255), text_thickness)
    
    return image

def create_image_grid(images, grid_size=(2, 2), spacing=10):
    """
    Create a grid of images.
    
    Args:
        images (list): List of images
        grid_size (tuple): Grid dimensions (rows, cols)
        spacing (int): Spacing between images
        
    Returns:
        numpy.ndarray: Grid image
    """
    rows, cols = grid_size
    if len(images) == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Get dimensions of first image
    h, w = images[0].shape[:2]
    
    # Calculate grid dimensions
    grid_h = rows * h + (rows - 1) * spacing
    grid_w = cols * w + (cols - 1) * spacing
    
    # Create grid canvas
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Place images in grid
    for i, img in enumerate(images[:rows * cols]):
        row = i // cols
        col = i % cols
        
        y_start = row * (h + spacing)
        y_end = y_start + h
        x_start = col * (w + spacing)
        x_end = x_start + w
        
        # Resize image if necessary
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        
        grid[y_start:y_end, x_start:x_end] = img
    
    return grid

def apply_filters(image, filter_type="gaussian", kernel_size=5):
    """
    Apply various filters to image.
    
    Args:
        image (numpy.ndarray): Input image
        filter_type (str): Type of filter to apply
        kernel_size (int): Kernel size for filters
        
    Returns:
        numpy.ndarray: Filtered image
    """
    if filter_type == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif filter_type == "median":
        return cv2.medianBlur(image, kernel_size)
    elif filter_type == "bilateral":
        return cv2.bilateralFilter(image, kernel_size, 80, 80)
    elif filter_type == "sharpen":
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    else:
        logger.warning(f"Unknown filter type: {filter_type}")
        return image

def enhance_contrast(image, alpha=1.2, beta=10):
    """
    Enhance image contrast.
    
    Args:
        image (numpy.ndarray): Input image
        alpha (float): Contrast control (1.0 = no change)
        beta (int): Brightness control (0 = no change)
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def detect_edges(image, low_threshold=50, high_threshold=150):
    """
    Detect edges in image using Canny edge detection.
    
    Args:
        image (numpy.ndarray): Input image
        low_threshold (int): Lower threshold for edge detection
        high_threshold (int): Upper threshold for edge detection
        
    Returns:
        numpy.ndarray: Edge image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, low_threshold, high_threshold)
