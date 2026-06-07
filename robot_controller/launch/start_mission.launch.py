from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_controller',
            executable='robot_mapper',
            name='robot_mapper',
            output='screen'
        ),
        Node(
            package='robot_controller',
            executable='flag_detector',
            name='flag_detector',
            output='screen'
        ),
        Node(
            package='robot_controller',
            executable='robot_control',
            name='robot_control',
            output='screen'
        ),
    ])
