# Sensor degradation tests (BEVFusion)

Code lives in the `bevfusion/` submodule (fork branch `cmpe249-fixes`):

- `mmdet3d/datasets/pipelines/degradation.py` — `SensorDegradation` pipeline step
- `tools/test.py` — `--degrade` and `--metrics-out` options
- `tools/run_degradation_sweep.py` — runs every condition and writes a summary CSV

## Degradation options

Passed as `--degrade key=value ...` to `tools/test.py`. Point-cloud changes are applied after the 10 LiDAR sweeps are merged; image changes are applied to the original 1600×900 frames before resizing.

| Option | Sensor | Effect | Simulates |
|---|---|---|---|
| `point_drop=0.5` | LiDAR | Removes that fraction of points at random | Sparse or partially blocked scan |
| `reduce_beams=16` | LiDAR | Keeps 16, 4, or 1 of the 32 beams (BEVFusion's built-in) | Lower-resolution or failing LiDAR |
| `sweeps_num=0` | LiDAR | Number of previous sweeps merged (default 9) | Missing scans / no temporal history |
| `max_range=30` | LiDAR | Removes points beyond this distance (m) | Limited sensing range (fog, rain) |
| `brightness=0.25` | Camera | Scales image brightness (1.0 = unchanged) | Low light, night |
| `blur=5` | Camera | Gaussian blur radius in pixels | Dirty lens, motion blur, rain |
| `drop_cams=[CAM_FRONT]` | Camera | Replaces listed cameras with black frames | Camera failure / missing frames |

Random point drop is seeded per sample, so each condition removes the same points on every run.

## Running

Inside the container (`docker start -ai bevf`, then `cd /workspace/bevfusion`).

One condition:

```bash
torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth --eval bbox \
  --degrade point_drop=0.5 brightness=0.25 \
  --metrics-out results/example.json
```

Full sweep (24 conditions, about 15 s inference + ~30 s startup each on an RTX 4070 SUPER):

```bash
python tools/run_degradation_sweep.py
```

Only some groups:

```bash
python tools/run_degradation_sweep.py --groups lidar_point_drop cam_drop
```

Groups: `lidar_point_drop`, `lidar_beams`, `lidar_sweeps`, `lidar_range`, `cam_brightness`, `cam_blur`, `cam_drop`. The clean baseline always runs first. Output goes to `results/degradation/fusion/`: one `.json` and `.log` per condition, plus `summary.csv` (mAP, NDS, per-class AP, fps). Finished conditions are skipped, so an interrupted sweep can be rerun.

The same script can evaluate the LiDAR-only checkpoint for comparison:

```bash
python tools/run_degradation_sweep.py \
  --config configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml \
  --checkpoint pretrained/lidar-only-det.pth \
  --out-dir results/degradation/lidar_only \
  --groups lidar_point_drop lidar_beams lidar_sweeps lidar_range
```

## Notes

- `wall_time_s` / `fps` cover the whole test loop (data loading and degradation included), not model latency alone. ROS 2 end-to-end latency is measured separately.
- Objects beyond `max_range` stay in the ground truth on purpose: the detector is scored on what it can no longer see.
- Dropping cameras does not remove LiDAR depth hints projected into the camera branch; it removes only the image content.
