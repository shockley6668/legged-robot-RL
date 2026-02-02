import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_control',
            executable='inference_node',
            name='inference_node',
            output='screen',
            parameters=[
                {'model_path': '/root/legged-robot/modelt_teacher.onnx'}
            ]
        )
    ])
