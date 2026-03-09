# Commands to run the nodes:
ros2 run mtp_gridmap "ros_node_name"
ros2 run imu_filter_madgwick imu_filter_madgwick_node --ros-args -p use_mag:=false -p fixed_frame:="odom" -p publish_tf:=true -r /imu/data_raw:=/imu/data_raw
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200 -p frame_id:=imu_link
ros2 run nmea_navsat_driver nmea_serial_driver --ros-args -p port:=/dev/ttyUSB1 -p frame_id:="gnss_link" -p baud:=4800
ros2 run image_proc rectify_node --ros-args -r image:=/image_raw -r camera_info:=/camera_info
ros2 bag play rosbags/demo_runs/grass/ --clock -p -r 1
ros2 launch src/launch/camera_launch.py
ros2 run topic_tools throttle messages /image_raw 1.0 /image_throttled
ros2 run rviz2 rviz2 --ros-args -p use_sim_time:=true

## Start and connect to docker container
'docker run -it --rm --device=/dev/video2:/dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix mtp_gridmap:v0.4.3'  

## Additional parameters to include:
--device=/dev/video0:/dev/video0" Maps the host's video device to the container.
"--device=/dev/dri:/dev/dri" Maps the host's Direct Rendering Infrastructure devices to the container for GPU acceleration.
"--device=/dev/ttyUSB0:/dev/ttyUSB0" Maps the host's USB serial IMU device to the container, this device is the ESP and connected to a micro ROS agent.
"--device=/dev/ttyUSB1:/dev/ttyUSB1" Maps the host's USB serial GNSS device to the container, this device is the BU-353N GNSS receiver.
"--gpus", "all" Grants the container access to all available GPUs on the host machine.
"--ipc=host" Shares the host's IPC namespace with the container.
"--net=host" Shares the host's network namespace with the container.

## Notes
The micro_ros_msgs is required for the micro_ros_agent package. These together read information from the ESP32. Which is then converted by the imu_filter_magdwick_node to add quaternions.

The micro_ros packages are installed using colcon build. Imu_filter_magdwick is installed through from the ROS packaged index.

## Settings to communicate over network:

### Global settings
sudo ufw disable --> Disable laptop firewall to allow communication

### Settings per window
export 
export ROS_DOMAIN=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp