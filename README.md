# MTP-ROS2
Repository used for the ROS2 implementation of my master thesis. It uses a Docker image which is brought online through a docker-compose.

# Usage guide
The steps below are used to build, start and access the docker container to start algorithms.
The connection with the Jetson is over USB-C. The Jetson assigns the following default IPs to itself and the connected laptop/pc. Jetson (192.168.55.1) and connected device (192.168.55.100).

## Preparation/Building
The docker image can be build by moving to the root of the directory and calling: `docker compose build`.

## Mandatory connections for the algorithm
The docker container uses USB devices, they should be plugged in with the following order:
1) MPU6050 which communicates through an ESP-32. This is /dev/ttyUSB0.
2) BU-353B5, which is the satellite receiver. This is /dev/ttyUSB1.
3) The camera, this is /dev/video0.

## Getting into the docker container
The docker container is started with:
`docker compose up -d`.

Connections to the docker container can be started with:
`docker exec -it mtp_gridmap_container bash`

## Building, sourcing and starting ROS packages
Once in the container the followings steps are made to run the code.
1) `cd src`
2) `colcon build`
3) `source install/setup.bash`

Then three options are available to start the three different core parts of the system.
1) `ros2 launch mtp_gridmap start_costmap.yaml`
2) `ros2 launch mtp_gridmap start_ekf.py`
3) `ros2 launch mtp_gridmap start_sensors.yaml`
4) `ros2 launch mtp_gridmap start_robot.yaml`

Each starts their own respective part of the system.
- 'start_costmap.yaml': Starts the camera feed preprocessing, neuralnetwork, pointcloud creation and gridmap nodes.
- 'start_ekf.yaml': Starts the extended kalman filter nodes.
- 'start_sensors.yaml': Starts the NMEA navsat driver, usb camera and the MPU6050 reader.
- 'start_robot.yaml': Starts all of the three mentioned packages and the planner server.