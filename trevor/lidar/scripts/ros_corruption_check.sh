#!/usr/bin/env bash
# Build the ROS 2 package with colcon, start the node as its own process, run the end-to-end check.
# Inside the sdc-lidar-ros image:  bash /workspace/sdc/lidar/scripts/ros_corruption_check.sh [pcd file]
set -eo pipefail
source /opt/ros/foxy/setup.bash
WS=/tmp/ros_ws
mkdir -p $WS/src && rm -rf $WS/src/lidar_corruption && cp -r /workspace/sdc/lidar/ros2/lidar_corruption $WS/src/
(cd $WS && colcon build --packages-select lidar_corruption 2>&1 | tail -5)
source $WS/install/setup.bash

files=(/dataset/nuscenes/samples/LIDAR_TOP/*.pcd.bin)
PCD=${1:-${files[0]}}
export PYTHONPATH=/workspace/sdc/common:/workspace/sdc/lidar:$PYTHONPATH
ros2 run lidar_corruption corruption_node > /tmp/node.log 2>&1 &
NODE=$!
trap "kill $NODE 2>/dev/null || true" EXIT
sleep 3
python /workspace/sdc/lidar/scripts/ros_corruption_check.py --pcd "$PCD"
echo "--- node log ---"; head -20 /tmp/node.log
