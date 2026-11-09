## Verified ROS 2 Configuration

**Development environment**
- Ubuntu 24.04.5 LTS via WSL2
- ROS 2 Jazzy
- Python 3.12.3
- Build system: `colcon` / `ament_python`

**Camera corruption node**
- Node: `camera_corruption_node`
- Input topic: `/camera/image_raw`
- Output topic: `/camera/image_corrupted`
- Message type: `sensor_msgs/msg/Image`
- Tested corruption: `brightness`
- Tested severity: `1`
- Brightness multiplier: `0.5`

**Validation performed**

The node was built and launched successfully using ROS 2 Jazzy.

A synthetic 480 × 640 BGR image with uniform pixel intensity 200 was published to `/camera/image_raw`.

The node published a corresponding image to `/camera/image_corrupted` with:
- Image dimensions: 480 × 640 × 3
- Mean pixel intensity: 100.0
- Frame ID: `camera_front`

This confirms successful image reception, brightness transformation, and publication.

**Not yet validated**
- Gaussian blur processing
- Exact timestamp preservation
- Consistency with offline corruption functions
- YOLO11 detector integration
- nuScenes replay integration