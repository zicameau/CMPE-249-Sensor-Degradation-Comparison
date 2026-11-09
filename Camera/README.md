## YOLO11 Camera Lane — Experimental Progress

### Overview

This lane evaluates YOLO11s as a camera-only object detection baseline using the nuScenes dataset. The objective is to measure detection accuracy and robustness under controlled camera degradation and establish a common 2D evaluation framework for comparison with LiDAR-only and camera–LiDAR fusion detectors.

### Completed Implementation

- Modular YOLO11 inference using Ultralytics.
- Camera corruption functions for brightness reduction and Gaussian blur.
- Projection of nuScenes 3D ground-truth annotations into 2D camera bounding boxes.
- Class-aware IoU matching and AP50 evaluation.
- Reproducible mini-validation experiments using a fixed sample manifest.
- Unit tests validated on Windows and Google Colab.

### Experimental Results

Dataset: nuScenes v1.0-mini, mini-validation split  
Camera: CAM_FRONT  
Model: YOLO11s  
Evaluated samples: 81 keyframes  
Metric: Projected 2D mAP@0.50

| Condition | mAP50 | Resilience Rate |
|---|---:|---:|
| Clean | 0.3528 | 1.000 |
| Brightness 1 | 0.3735 | 1.058 |
| Brightness 2 | 0.3445 | 0.976 |
| Brightness 3 | 0.2077 | 0.589 |
| Blur 1 | 0.3306 | 0.937 |
| Blur 2 | 0.2397 | 0.679 |
| Blur 3 | 0.1333 | 0.378 |

### Preliminary Findings

Gaussian blur caused a progressive decline in detection accuracy, with resilience falling below 0.8 at severity 2. Brightness reduction produced a smaller initial effect, with resilience falling below 0.8 at severity 3.

These findings are preliminary and do not yet include scene-level confidence intervals.

### Evaluation Limitations

- The current evaluation covers CAM_FRONT only.
- Projected 3D ground-truth boxes are approximations of visible 2D object extents.
- Cross-model comparisons require identical sample selection, class mappings, projection rules, and corruption implementations.
- Results are from nuScenes mini-validation, not the full nuScenes validation benchmark.

### Next Development Milestones

1. Implement the ROS 2 YOLO11 detector node.
2. Implement the ROS 2 camera corruption node.
3. Integrate with the shared nuScenes replay pipeline.
4. Export results using the team's common schema.
5. Add distance-binned recall and scene-level bootstrap confidence intervals.
6. Generate qualitative failure-case visualizations.

## Quick start

From this folder:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m pip install -e '.[eval]'
python scripts/run_camera_sweep.py --dataroot /path/to/nuscenes --out results/camera --limit 20
```



See `notebooks/YOLO11_Investigation_proj.ipynb` for the original experimental checkpoint.
