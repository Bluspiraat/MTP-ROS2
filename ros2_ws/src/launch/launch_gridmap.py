from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    laptop_camera_node = Node(
        package='mtp_gridmap',
        executable='camera_node',
        name='camera_node',
        output='screen'
    )

    gridmap_node = Node(
        package='mtp_gridmap',
        executable='ganav_node',
        name='ganav_node',
        output='screen'
    )

    return LaunchDescription([
        laptop_camera_node,
        gridmap_node
    ])