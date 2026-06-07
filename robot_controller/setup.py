from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hutaolover',
    maintainer_email='hesctory@gmail.com',
    description='Robot controller: wall-follow until flag found, stop when close enough.',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'robot_control = robot_controller.robot_control:main',
            'flag_detector = robot_controller.flag_detector:main',
            'robot_mapper  = robot_controller.robot_mapper:main',
        ],
    },
)
