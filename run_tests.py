#!/usr/bin/env python3
"""
Test runner for SmartEyes Obstacle Detection System.
"""

import sys
import os
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("✗ FAILED")
        print("Error:", e.stderr)
        return False

def main():
    """Run all tests."""
    print("SmartEyes Obstacle Detection System - Test Suite")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("smart_eyes_detection.py").exists():
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
    
    tests = [
        ("python -c 'import cv2; print(f\"OpenCV version: {cv2.__version__}\")'", "OpenCV Import Test"),
        ("python -c 'import numpy as np; print(f\"NumPy version: {np.__version__}\")'", "NumPy Import Test"),
        ("python -c 'from smart_eyes_detection import SmartEyesDetector; print(\"SmartEyes import successful\")'", "SmartEyes Import Test"),
        ("python -c 'from utils.image_utils import load_image; print(\"Image utils import successful\")'", "Image Utils Import Test"),
        ("python -c 'from utils.video_utils import VideoProcessor; print(\"Video utils import successful\")'", "Video Utils Import Test"),
        ("python -m pytest tests/ -v", "Unit Tests"),
    ]
    
    success_count = 0
    total_tests = len(tests)
    
    for command, description in tests:
        if run_command(command, description):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Test Results: {success_count}/{total_tests} tests passed")
    print('='*60)
    
    if success_count == total_tests:
        print("✓ All tests passed! The system is ready to use.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
