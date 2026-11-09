from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('input_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('output_topic', default_value='/camera/image_corrupted'),
        DeclareLaunchArgument('corruption', default_value='brightness'),
        DeclareLaunchArgument('severity', default_value='1'),
        Node(
            package='camera_corruption',
            executable='corruption_node',
            name='camera_corruption_node',
            output='screen',
            parameters=[{
                'input_topic': LaunchConfiguration('input_topic'),
                'output_topic': LaunchConfiguration('output_topic'),
                'corruption': LaunchConfiguration('corruption'),
                'severity': LaunchConfiguration('severity'),
            }],
        ),
    ])
