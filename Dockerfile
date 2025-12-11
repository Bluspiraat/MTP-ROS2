FROM osrf/ros:jazzy-desktop-full

# Install Python and essential tools
RUN apt update && apt install -y \
    python3-pip python3-setuptools python3-dev python3-venv python3-full\
    build-essential cmake git wget curl unzip \
    python3-colcon-common-extensions python3-rosdep \
    && rm -rf /var/lib/apt/lists/* 

# Create a venv for Python packages
RUN python3 -m venv /opt/pyenv

# Initialize rosdep
RUN rosdep init || true
RUN rosdep update

RUN /opt/pyenv/bin/pip install --upgrade pip

# Install Python dependencies
RUN /opt/pyenv/bin/pip install\
    "numpy<2" \
    opencv-python \
    onnxruntime \
    timm 

# Set workspace
WORKDIR /ros2_ws

# Make venv available in ROS environment
ENV PATH="/opt/pyenv/bin:${PATH}"
ENV PYTHONPATH="/opt/pyenv/lib/python3.12/site-packages:${PYTHONPATH}"

SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/jazzy/setup.bash && \
    python3 -m pip install --force-reinstall "numpy<2"

ENTRYPOINT ["/bin/bash"]