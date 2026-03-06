# ==== Base docker file for Jetson ROS2 Humble ====
FROM dustynv/ros:humble-desktop-l4t-r36.4.0

# Install python libraries
# Note: Dustin's image often has optimized NumPy; check if "numpy<1.24" is strictly required
RUN pip3 install --no-cache-dir --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 onnxruntime-gpu

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN /ros2_install.sh usb_cam && \
    /ros2_install.sh nmea_navsat_driver && \
    /ros2_install.sh grid_map_msgs && \
    /ros2_install.sh imu_filter_madgwick

SHELL ["/bin/bash", "-c"]

# Source the base ROS 2 setup automatically
RUN echo "source /opt/ros/${ROS_DISTRO}/install/setup.bash" >> /root/.bashrc

# Set working directory and default command
WORKDIR /ros2_ws 

CMD ["/bin/bash"]