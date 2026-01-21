# Commands to run the nodes:
ros2 run mtp_gridmap "ros_node_name"

## Start and connect to docker container
'docker run -it --rm --device=/dev/video2:/dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix mtp_gridmap:v0.4.3'  

## Additional parameters to include:
--device=/dev/video0:/dev/video0" Maps the host's video device to the container.
"--device=/dev/dri:/dev/dri" Maps the host's Direct Rendering Infrastructure devices to the container for GPU acceleration.
"--device=/dev/ttyUSB0:/dev/ttyUSB0" Maps the host's USB serial IMU device to the container, this device is the ESP and connected to a micro ROS agent.
"--gpus", "all" Grants the container access to all available GPUs on the host machine.
"--ipc=host" Shares the host's IPC namespace with the container.
"--net=host" Shares the host's network namespace with the container.

## Notes
The micro_ros_msgs is required for the micro_ros_agent package. These together read information from the ESP32. Which is then converted by the imu_filter_magdwick_node to add quaternions.

The micro_ros packages are installed using colcon build. Imu_filter_magdwick is installed through from the ROS packaged index.