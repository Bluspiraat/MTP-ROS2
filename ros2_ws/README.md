# Commands to run the nodes:
1. source /opt/ros/jazzy/setup.bash
2. colcon build
3. source install/setup.bash
4. ros2 run mtp_gridmap "ros_node_name"

## Build docker container
Navigate to the directory and run:
'docker build -t name:tag .' 

## Start and connect to docker container
'docker run -it --rm --device=/dev/video2:/dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix mtp_gridmap:v0.4.3'  
