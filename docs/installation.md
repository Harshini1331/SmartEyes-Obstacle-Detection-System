# Installation Guide

This guide will help you install and set up the SmartEyes Obstacle Detection System on your system.

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 2 GB free space
- **CPU**: Intel Core i3 or equivalent

### Recommended Requirements
- **RAM**: 16 GB or more
- **CPU**: Intel Core i7 or equivalent
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster processing)
- **Storage**: 10 GB free space (for models and datasets)

## Installation Methods

### Method 1: Using pip (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/Harshini1331/SmartEyes-Obstacle-Detection-System.git
   cd SmartEyes-Obstacle-Detection-System
   ```

2. **Create a virtual environment**
   ```bash
   # Create virtual environment
   python -m venv smarteyes_env
   
   # Activate virtual environment
   # On Windows:
   smarteyes_env\Scripts\activate
   # On macOS/Linux:
   source smarteyes_env/bin/activate
   ```

3. **Install the package**
   ```bash
   pip install -e .
   ```

### Method 2: Manual Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Harshini1331/SmartEyes-Obstacle-Detection-System.git
   cd SmartEyes-Obstacle-Detection-System
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO model files**
   ```bash
   # Create yolo directory
   mkdir yolo
   cd yolo
   
   # Download YOLO v3 files
   wget https://pjreddie.com/media/files/yolov3.weights
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
   
   # Return to project root
   cd ..
   ```

### Method 3: Using Docker

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libxrender-dev \
       libgomp1 \
       wget \
       && rm -rf /var/lib/apt/lists/*
   
   # Copy requirements and install Python dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Copy project files
   COPY . .
   
   # Download YOLO model files
   RUN mkdir yolo && cd yolo && \
       wget https://pjreddie.com/media/files/yolov3.weights && \
       wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg && \
       wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
   
   CMD ["python", "smart_eyes_detection.py", "--mode", "camera"]
   ```

2. **Build and run Docker container**
   ```bash
   # Build image
   docker build -t smarteyes .
   
   # Run container
   docker run -it --rm -v $(pwd)/output:/app/output smarteyes
   ```

## GPU Support (Optional)

For faster processing, you can enable GPU support:

### NVIDIA GPU with CUDA

1. **Install CUDA toolkit**
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
   - Follow installation instructions for your OS

2. **Install cuDNN**
   - Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - Extract and copy files to CUDA installation directory

3. **Install GPU-enabled OpenCV**
   ```bash
   pip uninstall opencv-python
   pip install opencv-python-gpu
   ```

4. **Verify GPU support**
   ```python
   import cv2
   print(cv2.cuda.getCudaEnabledDeviceCount())
   ```

### AMD GPU with ROCm

1. **Install ROCm**
   - Follow [ROCm installation guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)

2. **Install OpenCV with ROCm support**
   ```bash
   pip install opencv-python-headless
   ```

## Verification

After installation, verify that everything is working correctly:

1. **Test basic functionality**
   ```bash
   python -c "import cv2; print('OpenCV version:', cv2.__version__)"
   ```

2. **Run test suite**
   ```bash
   python -m pytest tests/ -v
   ```

3. **Test with sample image**
   ```bash
   python examples/basic_detection.py --mode image --input sample_image.jpg
   ```

## Troubleshooting

### Common Installation Issues

1. **OpenCV installation fails**
   ```bash
   # Try installing from conda
   conda install opencv
   
   # Or install specific version
   pip install opencv-python==4.8.1.78
   ```

2. **Permission errors on Windows**
   ```bash
   # Run as administrator or use --user flag
   pip install --user -r requirements.txt
   ```

3. **Memory errors during installation**
   ```bash
   # Increase pip timeout and use no-cache
   pip install --no-cache-dir --timeout 1000 -r requirements.txt
   ```

4. **YOLO model download fails**
   ```bash
   # Download manually and place in yolo/ directory
   # Or use alternative download method
   curl -O https://pjreddie.com/media/files/yolov3.weights
   ```

### Platform-Specific Issues

#### Windows
- Ensure Visual C++ Redistributable is installed
- Use Windows Subsystem for Linux (WSL) if needed
- Check PATH environment variables

#### macOS
- Install Xcode command line tools: `xcode-select --install`
- Use Homebrew for system dependencies: `brew install python3`

#### Linux
- Install system dependencies:
  ```bash
  sudo apt-get update
  sudo apt-get install python3-dev python3-pip libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
  ```

## Next Steps

After successful installation:

1. **Read the [User Guide](user_guide.md)** to learn how to use the system
2. **Check the [API Reference](api_reference.md)** for detailed documentation
3. **Run the [examples](examples/)** to see the system in action
4. **Customize the configuration** in `config/default_config.yaml`

## Getting Help

If you encounter issues during installation:

1. Check the [troubleshooting section](#troubleshooting) above
2. Search existing [GitHub Issues](https://github.com/Harshini1331/SmartEyes-Obstacle-Detection-System/issues)
3. Create a new issue with detailed error information
4. Join our community discussions

## Uninstallation

To remove the SmartEyes system:

1. **Deactivate virtual environment**
   ```bash
   deactivate
   ```

2. **Remove virtual environment**
   ```bash
   rm -rf smarteyes_env  # On Windows: rmdir /s smarteyes_env
   ```

3. **Remove project directory**
   ```bash
   rm -rf SmartEyes-Obstacle-Detection-System
   ```

4. **Uninstall package (if installed with pip)**
   ```bash
   pip uninstall smarteyes-obstacle-detection
   ```
