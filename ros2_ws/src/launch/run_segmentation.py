from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    camera_node = Node(
        package='mtp_gridmap',
        executable='camera_node',
        name='camera_node',
        output='screen'
    )

    ganav_node = Node(
        package='mtp_gridmap',
        executable='ganav_node',
        name='ganav_node',
        output='screen'
    )

    depth_node = Node(
        package='mtp_gridmap',
        executable='depth_node',
        name='depth_node',
        output='screen'
    )

    return LaunchDescription([
        camera_node,
        ganav_node,
        depth_node
    ])