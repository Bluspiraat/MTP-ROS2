# ==== Base docker file for Jetson ROS2 Humble ====
FROM dustynv/ros:humble-desktop-l4t-r36.4.0

# Install python libraries
RUN pip3 install --no-cache-dir --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 onnxruntime-gpu

# Install new ROS repository keys
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN /ros2_install.sh usb_cam && \
    /ros2_install.sh nmea_navsat_driver && \
    /ros2_install.sh grid_map_msgs && \
    /ros2_install.sh imu_filter_madgwick && \
    /ros2_install.sh topic_tools && \
    /ros2_install.sh robot_localization

ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV ROS_DOMAIN_ID=0

SHELL ["/bin/bash", "-c"]

# Change bashrc to use up to date configuration
RUN sed -i '\|/opt/ros/humble/setup.bash|d' /root/.bashrc && \
    echo "source /opt/ros/humble/install/setup.bash" >> /root/.bashrc

# Set working directory and default command
WORKDIR /ros2_ws 

CMD ["/bin/bash"]
