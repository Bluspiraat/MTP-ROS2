import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_share = get_package_share_directory('mtp_gridmap')
    camera_yaml = os.path.join(pkg_share, 'camera_info', 'navigation_camera.yaml')

    # Declaration of argument
    throttle_rate_arg = DeclareLaunchArgument(
        'throttle_rate',
        default_value='5.0',
        description='Publish rate of the camera in Hz'
    )

    throttle_rate = LaunchConfiguration('throttle_rate')

    camera_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        parameters=[{
            'camera_info_url': f'file://{camera_yaml}',
            'camera_name': 'navigation_camera',
        }]
    )

    throttle__image_node = Node(
        package='topic_tools',
        executable='throttle',
        name='image_throttler',
        arguments=['messages', '/image_raw', throttle_rate, '/image_raw_throttled']
    )

    rectification_node = Node(
        package='image_proc',
        executable='rectify_node',
        remappings=[
            ('image', '/image_raw_throttled'),
            ('camera_info', '/camera_info'),
        ]
    )

    return LaunchDescription([
        throttle_rate_arg,
        camera_node,
        throttle__image_node,
        rectification_node,
    ])