#!/usr/bin/env bash
# Build both ROS 2 packages, start the detector node (GPU), replay real samples and compare with the offline results.
# Inside the sdc-lidar-ros image (needs --gpus all, /dataset/nuscenes, /results, pretrained/):
#   bash /workspace/sdc/lidar/scripts/ros_detector_check.sh
set -eo pipefail
source /opt/ros/foxy/setup.bash
WS=/tmp/ros_ws
mkdir -p $WS/src
for p in lidar_corruption lidar_detector; do
  rm -rf $WS/src/$p && cp -r /workspace/sdc/lidar/ros2/$p $WS/src/
done
(cd $WS && colcon build --packages-select lidar_corruption lidar_detector 2>&1 | tail -6)
source $WS/install/setup.bash
export PYTHONPATH=/workspace/sdc/common:/workspace/sdc/lidar:$PYTHONPATH

cd /workspace/bevfusion
ln -sfn /dataset data
ros2 run lidar_detector detector_node --ros-args -p fixed_frame:=global -p reliable:=true -p verbose:=true > /tmp/detector.log 2>&1 &
NODE=$!
trap "kill $NODE 2>/dev/null || true" EXIT
python /workspace/sdc/lidar/scripts/ros_detector_check.py "$@" || RC=$?
echo "--- detector node log ---"; grep -E "INFO|WARN\]|ERROR" /tmp/detector.log | tail -60
exit ${RC:-0}
