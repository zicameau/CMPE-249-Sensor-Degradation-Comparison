from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    spec = [("input_topic", "/lidar/points", str), ("fixed_frame", "", str), ("sweeps_num", "9", int),
            ("score_threshold", "0.3", float), ("corruption", "none", str), ("severity", "0", int),
            ("seed", "0", int), ("checkpoint", "pretrained/lidar-only-det.pth", str)]
    args = [DeclareLaunchArgument(n, default_value=d) for n, d, _ in spec]
    node = Node(
        package="lidar_detector",
        executable="detector_node",
        parameters=[{n: ParameterValue(LaunchConfiguration(n), value_type=t) for n, _, t in spec}],
        output="screen",
    )
    return LaunchDescription(args + [node])
