# Shared CAM_FRONT 2D Detection Comparison

## Objective

Compare YOLO11s, TransFusion-L, and BEVFusion using a common 2D detection evaluation protocol on nuScenes v1.0-mini validation samples.

YOLO11s produces native 2D bounding boxes, while TransFusion-L and BEVFusion produce 3D bounding boxes that are projected into the CAM_FRONT image plane using nuScenes calibration information.

## Dataset and Evaluation Protocol

- **Dataset:** nuScenes v1.0-mini
- **Evaluation subset:** 81 shared validation samples
- **Camera:** CAM_FRONT
- **Ground-truth annotations:** 1,310 bounding boxes
- **Classes:** car, truck, bus, person, bicycle, motorcycle
- **Metric:** VOC-style continuous AP at IoU = 0.50
- **Mean AP:** Calculated across classes containing ground-truth annotations

Although six classes are included in the evaluation taxonomy, the shared subset contains no bus ground-truth annotations. Therefore, mAP50 is calculated across five classes.

The same ground-truth annotations and evaluation implementation are used for all three models.

## Results

| Model | mAP50 | Status |
|---|---:|---|
| YOLO11s | 0.3444 | Verified clean evaluation |
| TransFusion-L | 0.4790 | Diagnostic Z-adjusted evaluation |
| BEVFusion | 0.4985 | Diagnostic Z-adjusted evaluation |

BEVFusion achieved the highest projected 2D mAP50, followed by TransFusion-L and YOLO11s.

These results describe performance under the shared CAM_FRONT 2D evaluation protocol and should not be interpreted as native nuScenes 3D detection scores.

## Diagnostic Z-Coordinate Adjustment

Initial projections of the TransFusion-L and BEVFusion predictions exhibited a systematic vertical displacement relative to the 2D ground-truth boxes.

A diagnostic adjustment was evaluated:

`translation_z_adjusted = translation_z_original - box_height / 2`

The adjustment substantially improved projected 2D detection performance.

The evaluation script supports this adjustment through the optional argument:

`--z-correction half-height`

The adjustment is applied in memory without modifying the original prediction JSON files.

**Important:** The original model exporters' coordinate conventions have not been independently verified. Consequently, the adjusted results are considered provisional diagnostic results rather than conclusively corrected model outputs.

The original LiDAR export also reports camera usage in its metadata, which requires verification against the original model configuration.

## Reproducibility

From the Camera repository root, regenerate the adjusted TransFusion-L evaluation:

```powershell
python scripts/evaluate_3d_predictions_2d.py --predictions results/shared_exports/Lidar_results_nusc.json --dataroot "C:\SJSU\datasets\nuScenes" --version v1.0-mini --name lidar_corrected --z-correction half-height --strict-missing
```

Regenerate the adjusted BEVFusion evaluation:

```powershell
python scripts/evaluate_3d_predictions_2d.py --predictions results/shared_exports/BevFusion_results_nusc.json --dataroot "C:\SJSU\datasets\nuScenes" --version v1.0-mini --name bevfusion_corrected --z-correction half-height --strict-missing
```

Regenerate the consolidated comparison:

```powershell
python scripts/build_shared_comparison.py
```

Run regression tests:

```powershell
python -m pytest tests -q
```

## Limitations

1. The evaluation includes only 81 shared validation samples, limiting the generalizability of the findings.
2. Class representation is imbalanced, particularly for bicycles and trucks.
3. Bus AP cannot be evaluated because no bus ground-truth annotations are present.
4. The 3D-to-2D Z-coordinate adjustment is diagnostic and requires exporter-level verification.
5. Projected 2D AP50 does not capture the full accuracy of 3D detection, including depth, orientation, and velocity estimation.
6. Statistical significance between model performance differences has not been established.
7. The results represent a shared 2D evaluation, not a comprehensive comparison of sensor degradation robustness.

## Conclusion

The shared evaluation provides a consistent method for comparing camera-only, LiDAR-based, and camera–LiDAR fusion detection outputs in the CAM_FRONT image plane.

Under the diagnostic Z-adjusted evaluation, BEVFusion achieved the highest mAP50. However, the results should be interpreted with the documented coordinate-convention uncertainty and dataset limitations.