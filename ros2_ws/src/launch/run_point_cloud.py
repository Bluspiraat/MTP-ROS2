from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
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

    point_cloud_node = Node(
        package='mtp_gridmap',
        executable='point_cloud_node',
        name='point_cloud_node',
        output='screen'
    )

    return LaunchDescription([
        ganav_node,
        depth_node,
        point_cloud_node,
    ])