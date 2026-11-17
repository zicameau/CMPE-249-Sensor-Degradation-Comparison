#!/usr/bin/env bash
# Build /workspace/bevfusion/data/nuscenes for the LiDAR runs (inside the sdc-lidar image).
#
#   /dataset/nuscenes      raw nuScenes (samples/, sweeps/, maps/, full metadata) - mounted read-only is fine
#   /dataset/nuscenes_val  val-only metadata from make_val_subset.py (v1.0-trainval/) + nuscenes_infos_val.pkl
#
# The val-only metadata keeps the devkit under the 8 GB Docker VM limit (the full trainval tables get OOM-killed).
# For mini runs set MINI=1: uses /dataset/nuscenes directly (v1.0-mini metadata + mini_infos_val.pkl).
set -euo pipefail
cd /workspace/bevfusion
rm -rf data
mkdir -p data/nuscenes
cd data/nuscenes
ln -s /dataset/nuscenes/samples samples
ln -s /dataset/nuscenes/sweeps sweeps
ln -s /dataset/nuscenes/maps maps
if [ "${MINI:-0}" = "1" ]; then
    ln -s /dataset/nuscenes/v1.0-mini v1.0-mini
    ln -s /dataset/nuscenes/mini_infos_val.pkl mini_infos_val.pkl
else
    ln -s /dataset/nuscenes_val/v1.0-trainval v1.0-trainval
    ln -s /dataset/nuscenes_val/nuscenes_infos_val.pkl nuscenes_infos_val.pkl
fi
ls -l
