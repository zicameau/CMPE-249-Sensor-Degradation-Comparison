from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument("corruption", default_value="none"),
        DeclareLaunchArgument("severity", default_value="0"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("input_topic", default_value="/lidar/points"),
        DeclareLaunchArgument("output_topic", default_value="/lidar/points_corrupted"),
    ]
    node = Node(
        package="lidar_corruption",
        executable="corruption_node",
        parameters=[{
            "corruption": ParameterValue(LaunchConfiguration("corruption"), value_type=str),
            # launch args are strings; the node declares ints, so cast explicitly
            "severity": ParameterValue(LaunchConfiguration("severity"), value_type=int),
            "seed": ParameterValue(LaunchConfiguration("seed"), value_type=int),
            "input_topic": ParameterValue(LaunchConfiguration("input_topic"), value_type=str),
            "output_topic": ParameterValue(LaunchConfiguration("output_topic"), value_type=str),
        }],
        output="screen",
    )
    return LaunchDescription(args + [node])
