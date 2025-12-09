from setuptools import find_packages, setup

package_name = 'mtp_gridmap'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
        ],
    },
)
