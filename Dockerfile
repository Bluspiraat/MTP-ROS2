# ==== Base docker file for ROS2 Jazzy with GPU support ====
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Setup environment variables
ARG ROS_DISTRO=humble
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install ROS2 Humble Desktop and Build Tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget curl gnupg \
    python3-pip python3-dev \
    # GUI/OpenCV runtime libraries
    libsm6 libxext6 libxrender-dev \
    # Install locales and necessary repository tools
    locales \
    software-properties-common && \
    \
    # Setup Locales for ROS
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8 && \
    \
    # Add university repository
    add-apt-repository universe && \
    apt-get update && \
    \
    # Set up ROS sources using DEB package
    export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && \
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" &&\
    dpkg -i /tmp/ros2-apt-source.deb && \
    apt-get update && \
    apt-get upgrade -y && \
    # Install ROS 2 Desktop
    apt-get install --no-install-recommends -y \
    ros-humble-desktop \
    # ROS dependencies and build tools
    python3-colcon-common-extensions \
    python3-rosdep \
    # Clean up APT lists
    && rm -rf /var/lib/apt/lists/*

# Install python libraries
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio \
    onnxruntime-gpu \
    opencv-python \
    timm

SHELL ["/bin/bash", "-c"]
# Source the base ROS 2 setup automatically
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Set working directory and default command
WORKDIR /ros2_ws 
CMD ["/bin/bash"]