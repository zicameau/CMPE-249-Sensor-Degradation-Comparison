# YOLO11s Camera Perception Lane

## Overview

This project evaluates **YOLO11s as a camera-only object detection baseline** for autonomous-driving perception under controlled image degradation.

Using the nuScenes dataset and ROS 2, the camera lane measures detection accuracy, robustness, and CPU inference latency. It also provides a common camera-space evaluation framework to support comparisons with the CenterPoint LiDAR-only and BEVFusion camera–LiDAR fusion lanes.

**Status:** Independent camera implementation, experiments, and validation complete. Cross-model integration is pending teammate results.

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