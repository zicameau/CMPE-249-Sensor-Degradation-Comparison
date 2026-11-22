# YOLO11s Camera Perception Lane

## Overview

This project evaluates **YOLO11s as a camera-only object detection baseline** for autonomous-driving perception under controlled image degradation using nuScenes and ROS 2.

The camera lane measures detection accuracy, robustness, and CPU inference latency. It also supports a shared CAM_FRONT 2D evaluation comparing YOLO11s, TransFusion-L (LiDAR), and BEVFusion (camera–LiDAR fusion).

**Status:** Camera implementation, degradation experiments, shared three-model evaluation, and regression testing complete. The TransFusion-L and BEVFusion comparison results use a reproducible diagnostic Z-coordinate adjustment whose exporter-level validity remains unverified.

## Shared Three-Model Comparison

| Model | CAM_FRONT mAP50 | Evaluation Status |
|---|---:|---|
| YOLO11s | 0.3444 | Verified clean |
| TransFusion-L | 0.4790 | Diagnostic Z-adjusted |
| BEVFusion | 0.4985 | Diagnostic Z-adjusted |

All three models were evaluated against the same 81-sample CAM_FRONT validation subset using a common 2D AP50 protocol.

BEVFusion achieved the highest projected 2D mAP50, followed by TransFusion-L and YOLO11s. However, the TransFusion-L and BEVFusion results remain provisional because the original exporters' Z-coordinate conventions have not been independently verified.

These results represent **projected 2D detection performance**, not native nuScenes 3D detection accuracy or a matched comparison of sensor degradation robustness.

For detailed methodology, reproducibility instructions, and evaluation limitations, see `results/shared_2d_comparison/README.md`.


## Implementation

The camera lane includes:

- Pretrained YOLO11s object detection using Ultralytics.
- nuScenes CAM_FRONT image replay through ROS 2.
- ROS 2 image corruption and detector nodes.
- Brightness reduction and Gaussian blur at three severity levels each.
- Projection of nuScenes 3D annotations into 2D camera bounding boxes.
- Class-aware AP50, precision, recall, distance-based recall, and per-scene evaluation.
- ROS 2 inference and callback latency measurement.
- Shared-format metric exports for team integration.

### ROS 2 Architecture

```text
nuScenes CAM_FRONT replay
          |
          v
  /camera/image_raw
          |
          v
 Camera corruption node
          |
          v
/camera/image_corrupted
          |
          v
 YOLO11s detector node
          |
          v
  /camera/detections
```

The detector publishes `vision_msgs/Detection2DArray` messages. For the clean latency baseline, replay images are published directly to `/camera/image_corrupted`.

## Dataset and Evaluation

| Configuration | Value |
|---|---|
| Dataset | nuScenes v1.0-mini |
| Evaluation subset | 81 mini-validation samples |
| Scenes | scene-0103 and scene-0916 |
| Camera | CAM_FRONT |
| Model | YOLO11s |
| Target classes