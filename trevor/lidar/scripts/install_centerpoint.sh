#!/usr/bin/env bash
# Install the LiDAR-only CenterPoint stretch model into a BEVFusion checkout (run inside the sdc-lidar image).
#   bash /workspace/sdc/lidar/scripts/install_centerpoint.sh        # cwd: /workspace/bevfusion
# Result: configs/nuscenes/det/centerhead/secfpn/lidar/voxelnet_0p075.yaml and
#         pretrained/centerpoint_0075voxel_bevfusion.pth
# Then run the sweep with
#   --config configs/nuscenes/det/centerhead/secfpn/lidar/voxelnet_0p075.yaml \
#   --checkpoint pretrained/centerpoint_0075voxel_bevfusion.pth
set -euo pipefail
SDC="${SDC:-/workspace/sdc}"
URL="https://download.openmmlab.com/mmdetection3d/v0.1.0_models/centerpoint/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20200925_230905-358fbe3b.pth"
mkdir -p configs/nuscenes/det/centerhead/secfpn/lidar pretrained
cp "$SDC/lidar/configs/centerpoint_voxelnet_0p075.yaml" configs/nuscenes/det/centerhead/secfpn/lidar/voxelnet_0p075.yaml
[ -f pretrained/centerpoint_0075voxel.pth ] || wget -q -O pretrained/centerpoint_0075voxel.pth "$URL"
python "$SDC/lidar/scripts/convert_centerpoint_ckpt.py" \
    --src pretrained/centerpoint_0075voxel.pth --dst pretrained/centerpoint_0075voxel_bevfusion.pth
