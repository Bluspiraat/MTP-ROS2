# MTP-ROS2
Repository used for the ROS2 implementation of my master thesis

# Docker image
The docker image is based on nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04. This docker image is based on Ubuntu Jammy, uses Python 3.10 and has support for Cuda 12.4 and cuDNN 9.1.0.
The installed ROS version is Humble as this is developed for Ubuntu Jammmy.

# Docker container
It requires inputs:
- gpu: Used for GPU acceleration of the segmentation algorithms.
- video: It requires a video input, /dev/video0 is the input for the camera node.

# Packages used:
Some additional Python packages are used
- "numpy<2.0": CVBridge for ROS2 is based on numpy <2, this is therefore a hard constraint.
- torch torchvision torchaudio (--extra-index-url https://download.pytorch.org/whl/cu124): This installs torch for Cuda 12.4 which is supported by the base image. The library is used to run the Mono Depth algorithm.
- onnxruntime-gpu: Used to run the ONNX export of GA-Nav.
- opencv-python: Used for displaying and processing images.
- timm: Used in the GA-Nav segmentation head and is therefore a dependency.