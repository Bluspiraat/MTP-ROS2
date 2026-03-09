from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mtp_gridmap'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', glob('share/models/*')),
        ('share/' + package_name + '/camera_info', glob('share/camera_info/*')),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.yaml')),
        # Include all config (YAML) files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Include URDF files
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf'))
],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'torch',
        'mmsegmentation'],
    zip_safe=True,
    maintainer='bluspiraat',
    maintainer_email='matthijs.sluijk@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = mtp_gridmap.camera_node:main',
            'ganav_node = mtp_gridmap.ganav_node:main',
            'depth_node = mtp_gridmap.depth_node:main',
            'point_cloud_node = mtp_gridmap.point_cloud_node:main',
            'grid_map_node = mtp_gridmap.grid_map_node:main',
            'gnss_bridge.py = mtp_gridmap.gnss_bridge:main',
            'heading_node = mtp_gridmap.heading_node:main',
            'planner_server_node = mtp_gridmap.planner_server_node:main',
            'planner_client_node = mtp_gridmap.planner_client_node:main',
        ],
    },
)
