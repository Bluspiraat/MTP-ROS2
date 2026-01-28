import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('mtp_gridmap')
    
    # 1. Paths to files
    urdf_file = os.path.join(pkg_share, 'urdf', 'robot.urdf')
    ekf_config_global = os.path.join(pkg_share, 'config', 'ekf_global.yaml')
    madgwick_config = os.path.join(pkg_share, 'config', 'madgwick.yaml')
    navsat_config = os.path.join(pkg_share, 'config', 'navsat_transform.yaml')

    # 2. Read URDF
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        # Robot State Publisher (TF Tree)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_desc}]
        ),

        # Your Velocity Bridge (Python script)
        Node(
            package='mtp_gridmap',
            executable='gnss_bridge.py',
            name='gnss_bridge'
        ),

        Node(
            package='imu_filter_madgwick',
            executable='imu_filter_madgwick_node',
            name='madgwick_filter',
            output='screen',
            parameters=[madgwick_config],
            remappings=[
                ('/imu/data', '/imu')            # Output filtered IMU data
            ]
        ),

        Node(
            package='mtp_gridmap',
            executable='heading_node',
            name='gps_heading_node'
        ),

        # NavSat Transform (GPS to Meters)
        Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform_node',
            output='screen',
            parameters=[navsat_config],
            remappings=[
                ('/gps/fix', '/fix'),
            ]
        ),

        # EKF Node (The "Brain")
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_global_node',
            output='screen',
            parameters=[ekf_config_global]
        )
    ])