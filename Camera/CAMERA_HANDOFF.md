# YOLO11s Camera Lane — Team Handoff

## Status

The independent camera lane experiments are complete.

- Model: YOLO11s
- Dataset: nuScenes v1.0-mini
- Evaluation subset: 81 mini-val samples from two scenes
- Camera: CAM_FRONT
- Corruption conditions: clean, brightness 1–3, Gaussian blur 1–3
- Accuracy evaluation: complete
- ROS 2 CPU latency evaluation: complete
- Shared metrics export: complete
- Validation: 7/7 conditions passed

## Primary Integration Files

### 1. Unified metrics

`results/camera_all_shared_metrics.csv`

Contains 203 records covering 14 metric types across seven conditions.

Shared schema:

`model,lane,corruption,severity,seed,hardware,metric,class,value,ci_low,ci_high`

This is the primary file for cross-lane metrics integration.

### 2. Human-readable performance summary

`results/camera_accuracy_latency_summary.csv`

Contains one row per corruption condition with mAP50,
inference latency, callback latency, and detection counts.

### 3. Shared sample manifest

`configs/mini_val_sample_tokens.txt`

Contains 81 unique nuScenes mini-val sample tokens.

Other lanes should use the same samples for matched comparisons
where their model inputs and evaluation protocols permit.

### 4. Camera-space evaluation data

`results/camera_shared_mini_val/ground_truth.csv`

`results/camera_shared_mini_val/predictions.csv`

These provide the projected camera-space ground truth and
YOLO11s predictions used for offline 2D evaluation.

### 5. Detailed evaluation outputs

`results/camera_shared_mini_val/metrics.json`

`results/camera_shared_mini_val/distance_recall.json`

`results/camera_shared_mini_val/per_scene_metrics.json`

These contain per-class, distance-based, and per-scene results.

## Evaluation Details

- Confidence threshold for fixed-threshold precision/recall: 0.25
- IoU threshold for AP50 and distance recall: 0.50
- Near distance: 0–20 m
- Medium distance: 20–40 m
- Far distance: 40+ m
- Classes: car, truck, bus, person, bicycle, motorcycle
- Bus has no ground-truth instances in this subset
- Macro mAP50 excludes classes without ground truth

## ROS 2 Timing Details

- ROS 2 Jazzy, Ubuntu 24.04 under WSL
- YOLO11s inference on CPU
- 81 callbacks recorded per condition
- First five callbacks excluded from latency statistics
- 76 measured callbacks per condition
- 532 measured callbacks across seven conditions
- Replay rate: 1 Hz

Inference latency excludes ROS transport and image corruption time.
Estimated inference FPS is not observed end-to-end ROS throughput.

Accuracy and latency were evaluated in separate runs.

## Corruption Definitions

| Condition | Parameter |
|---|---|
| clean_0 | No corruption |
| brightness_1 | Multiply intensity by 0.50 |
| brightness_2 | Multiply intensity by 0.25 |
| brightness_3 | Multiply intensity by 0.10 |
| blur_1 | Gaussian sigma 2 |
| blur_2 | Gaussian sigma 5 |
| blur_3 | Gaussian sigma 10 |

## Cross-Lane Integration Notes

1. Align model results by sample token and corruption condition.
2. Verify that corruption severity definitions match across lanes.
3. Preserve the distinction between 2D AP50 and nuScenes 3D metrics.
4. Compare latency only with explicit hardware and timing-methodology context.
5. Do not interpret blank confidence interval or seed fields as zero.
6. Do not interpret absent-class AP as zero.

## Pending Dependencies

- BEVFusion results from the fusion lane
- CenterPoint results from the LiDAR lane
- Agreed cross-model comparison protocol
- Optional projection of teammate 3D predictions into camera space

The camera lane includes `sdc_camera/prediction_adapter.py`
for projecting official-format nuScenes 3D detections into 2D.

This adapter requires teammate prediction files before
cross-model camera-space evaluation can be performed.

## Documentation

See `CAMERA_README.md` for the camera architecture,
experimental methodology, results, and limitations.
