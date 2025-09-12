#!/usr/bin/env python3
"""
Test cases for SmartEyes Obstacle Detection System.
"""

import unittest
import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from smart_eyes_detection import SmartEyesDetector
from utils.image_utils import load_image, save_image, resize_image, draw_bounding_box
from utils.video_utils import VideoProcessor

class TestSmartEyesDetector(unittest.TestCase):
    """Test cases for SmartEyesDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = SmartEyesDetector(confidence_threshold=0.5, nms_threshold=0.4)
        
        # Create a test image
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(self.test_image, (100, 100), (200, 200), (0, 255, 0), -1)
        cv2.rectangle(self.test_image, (300, 300), (400, 400), (255, 0, 0), -1)
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.confidence_threshold, 0.5)
        self.assertEqual(self.detector.nms_threshold, 0.4)
        self.assertIsNone(self.detector.net)
        self.assertEqual(len(self.detector.classes), 0)
        self.assertEqual(len(self.detector.colors), 0)
    
    def test_detector_without_model(self):
        """Test detector behavior without loaded model."""
        with self.assertRaises(ValueError):
            self.detector.detect_obstacles(self.test_image)
    
    def test_detector_parameters(self):
        """Test detector parameter validation."""
        # Test valid parameters
        detector1 = SmartEyesDetector(0.3, 0.2)
        self.assertEqual(detector1.confidence_threshold, 0.3)
        self.assertEqual(detector1.nms_threshold, 0.2)
        
        # Test edge cases
        detector2 = SmartEyesDetector(0.0, 1.0)
        self.assertEqual(detector2.confidence_threshold, 0.0)
        self.assertEqual(detector2.nms_threshold, 1.0)

class TestImageUtils(unittest.TestCase):
    """Test cases for image utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.test_path = "test_image.jpg"
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_path):
            os.remove(self.test_path)
    
    def test_save_and_load_image(self):
        """Test image save and load functionality."""
        # Save image
        save_image(self.test_image, self.test_path)
        self.assertTrue(os.path.exists(self.test_path))
        
        # Load image
        loaded_image = load_image(self.test_path)
        self.assertIsNotNone(loaded_image)
        self.assertEqual(loaded_image.shape, self.test_image.shape)
    
    def test_resize_image(self):
        """Test image resizing functionality."""
        # Test with aspect ratio maintained
        resized = resize_image(self.test_image, (200, 150), maintain_aspect_ratio=True)
        self.assertEqual(resized.shape[:2], (150, 200))
        
        # Test without aspect ratio maintained
        resized = resize_image(self.test_image, (200, 150), maintain_aspect_ratio=False)
        self.assertEqual(resized.shape[:2], (150, 200))
    
    def test_draw_bounding_box(self):
        """Test bounding box drawing functionality."""
        bbox = (10, 10, 50, 50)
        label = "test_object"
        confidence = 0.85
        
        result = draw_bounding_box(self.test_image.copy(), bbox, label, confidence)
        
        # Check that image was modified (should be different from original)
        self.assertFalse(np.array_equal(result, self.test_image))
        self.assertEqual(result.shape, self.test_image.shape)

class TestVideoUtils(unittest.TestCase):
    """Test cases for video utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_video_path = "test_video.mp4"
        self.test_output_path = "test_output.mp4"
        
        # Create a simple test video
        self.create_test_video()
    
    def tearDown(self):
        """Clean up test files."""
        for path in [self.test_video_path, self.test_output_path]:
            if os.path.exists(path):
                os.remove(path)
    
    def create_test_video(self):
        """Create a simple test video."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.test_video_path, fourcc, 30, (640, 480))
        
        for i in range(30):  # 1 second at 30 FPS
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            writer.write(frame)
        
        writer.release()
    
    def test_video_processor_initialization(self):
        """Test VideoProcessor initialization."""
        processor = VideoProcessor(self.test_video_path, self.test_output_path)
        self.assertEqual(processor.input_path, self.test_video_path)
        self.assertEqual(processor.output_path, self.test_output_path)
        self.assertIsNone(processor.cap)
        self.assertIsNone(processor.writer)
    
    def test_video_processor_open_video(self):
        """Test video opening functionality."""
        processor = VideoProcessor(self.test_video_path)
        self.assertTrue(processor.open_video())
        self.assertIsNotNone(processor.cap)
        self.assertTrue(processor.cap.isOpened())
        processor.close()
    
    def test_video_processor_setup_writer(self):
        """Test video writer setup."""
        processor = VideoProcessor(self.test_video_path, self.test_output_path)
        processor.open_video()
        self.assertTrue(processor.setup_writer())
        self.assertIsNotNone(processor.writer)
        self.assertTrue(processor.writer.isOpened())
        processor.close()
    
    def test_get_video_info(self):
        """Test video information extraction."""
        from utils.video_utils import get_video_info
        
        info = get_video_info(self.test_video_path)
        self.assertIsNotNone(info)
        self.assertEqual(info['width'], 640)
        self.assertEqual(info['height'], 480)
        self.assertEqual(info['fps'], 30.0)
        self.assertEqual(info['frame_count'], 30)

class TestIntegration(unittest.TestCase):
    """Integration test cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_path = "integration_test.jpg"
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_path):
            os.remove(self.test_path)
    
    def test_full_image_processing_pipeline(self):
        """Test complete image processing pipeline."""
        # Save test image
        save_image(self.test_image, self.test_path)
        
        # Load image
        loaded_image = load_image(self.test_path)
        self.assertIsNotNone(loaded_image)
        
        # Resize image
        resized = resize_image(loaded_image, (320, 240))
        self.assertEqual(resized.shape[:2], (240, 320))
        
        # Draw bounding box
        bbox = (50, 50, 100, 100)
        result = draw_bounding_box(resized.copy(), bbox, "test", 0.9)
        self.assertIsNotNone(result)

def run_tests():
    """Run all test cases."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSmartEyesDetector))
    test_suite.addTest(unittest.makeSuite(TestImageUtils))
    test_suite.addTest(unittest.makeSuite(TestVideoUtils))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
