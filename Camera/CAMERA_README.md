# YOLO11s Camera Perception Lane

## Overview

This lane evaluates pretrained YOLO11s for camera-based object detection under controlled image degradation using nuScenes and ROS 2.

The evaluation compares clean images against three brightness degradation levels and three Gaussian blur levels.

## ROS 2 Architecture

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
/camera/detections (vision_msgs/Detection2DArray)
```

For the clean latency baseline, the replay publisher sends images directly to /camera/image_corrupted.

## Dataset

- Dataset: nuScenes v1.0-mini
- Evaluation subset: 81 mini-val samples
- Scenes: scene-0103 (40 samples), scene-0916 (41 samples)
- Camera: CAM_FRONT
- Original resolution: 1600 x 900
- Sample manifest: configs/mini_val_sample_tokens.txt
- Target classes: car, truck, bus, person, bicycle, motorcycle

## Corruption Settings

| Condition | Severity | Parameter |
|---|---:|---|
| Clean | 0 | Unmodified image |
| Brightness | 1 | Intensity multiplier 0.50 |
| Brightness | 2 | Intensity multiplier 0.25 |
| Brightness | 3 | Intensity multiplier 0.10 |
| Gaussian blur | 1 | Sigma 2 |
| Gaussian blur | 2 | Sigma 5 |
| Gaussian blur | 3 | Sigma 10 |

## Accuracy Evaluation

The offline evaluation uses pretrained YOLO11s predictions and 2D camera-space bounding boxes projected from nuScenes 3D annotations.

Metrics include:

- AP50 per class using continuous precision-recall interpolation
- Macro mAP50 over classes with ground-truth instances
- Precision and recall at confidence threshold 0.25
- Distance-stratified recall: near (0-20 m), medium (20-40 m), far (40+ m)
- Per-scene accuracy

The bus class has no ground-truth instances in this subset and is excluded from the macro mAP50 calculation.

## ROS 2 Latency Evaluation

- Runtime: ROS 2 Jazzy under Ubuntu 24.04 / WSL
- Inference device: CPU
- Replay interval: 1 second per image
- Frames per condition: 81
- Initial callbacks excluded from latency summary: 5
- Measured callbacks per condition: 76
- Model startup and explicit synthetic warm-up excluded

Inference latency measures the YOLO prediction call. Callback latency measures detector callback processing through detection publication. Neither includes publisher loading, ROS transport, or corruption-node processing time. Estimated inference FPS is calculated as 1000 / mean inference latency; it is not measured ROS throughput.

Each condition was benchmarked once, so small latency differences should not be interpreted as statistically significant.

## Results

| Condition | mAP50 | Mean inference (ms) | P95 inference (ms) | Detections |
|---|---:|---:|---:|---:|
| clean_0 | 0.3444 | 45.37 | 50.57 | 1027 |
| brightness_1 | 0.3689 | 45.26 | 51.22 | 983 |
| brightness_2 | 0.3390 | 44.55 | 50.66 | 882 |
| brightness_3 | 0.2030 | 41.80 | 48.92 | 583 |
| blur_1 | 0.3218 | 39.11 | 47.53 | 927 |
| blur_2 | 0.2316 | 39.06 | 46.13 | 762 |
| blur_3 | 0.1290 | 40.03 | 47.64 | 468 |

Accuracy values come from offline evaluation. Latency and detection counts come from the separate ROS 2 replay benchmarks.

## Main Findings

- Severe brightness degradation reduced detection accuracy.
- Gaussian blur caused substantial accuracy degradation at higher severities.
- Mean inference latency varied less dramatically than detection accuracy.
- Detection counts generally decreased as corruption severity increased.
- Observed latency differences are descriptive, not causal conclusions.

## Key Files

- `configs/mini_val_sample_tokens.txt`: shared evaluation samples
- `scripts/run_camera_sweep.py`: offline corruption and accuracy sweep
- `scripts/export_camera_metrics.py`: shared metric export
- `sdc_camera/projection.py`: 3D ground-truth projection to camera boxes
- `sdc_camera/evaluation_2d.py`: 2D detection evaluation
- `sdc_camera/distance_recall.py`: distance-based recall
- `sdc_camera/prediction_adapter.py`: 3D detection prediction projection adapter
- `ros2/replay_nuscenes_camera.py`: ROS 2 dataset replay
- `ros2/camera_corruption/`: image degradation node
- `ros2/camera_detector/`: YOLO11s inference node
- `results/camera_shared_metrics.csv`: shared metric schema export
- `results/camera_ros2_latency_summary.csv`: ROS 2 latency results
- `results/camera_accuracy_latency_summary.csv`: combined summary

## Team Integration

Other perception lanes should use the shared mini-val sample manifest and the same corruption definitions where applicable.

The 3D prediction adapter can project detections into camera space for comparison, but requires the corresponding teammate prediction files.

YOLO11s 2D mAP50 should not be directly equated with nuScenes 3D mAP without a clearly defined common evaluation protocol.

## Limitations

- Only two scenes and 81 mini-val samples were evaluated.
- The subset does not represent the full nuScenes validation distribution.
- Latency measurements were collected from single CPU runs.
- Corruption-node processing and ROS transport latency were not measured.
- The evaluation does not establish statistically significant differences.
- Cross-model comparison awaits the other perception lanes' outputs.
