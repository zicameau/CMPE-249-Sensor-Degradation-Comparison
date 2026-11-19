# nuScenes result files

Official nuScenes-format output for BEVFusion (camera + LiDAR), clean baseline only. Degraded and ROS 2 runs are not stored here (about 500 MB in total); regenerate them with the commands below.

## `offline/baseline__clean/`

Clean nuScenes-mini val (81 keyframes), no degradation, released `bevfusion-det.pth`, fork commit [`f9ba929`](https://github.com/christiankso/bevfusion/commit/f9ba929). mAP 0.582, NDS 0.584.

| File | Contents |
|---|---|
| `results_nusc.json` | Every predicted box for every keyframe in the nuScenes submission format: `results` maps each sample token to a list of boxes with `translation`, `size`, `rotation`, `velocity`, `detection_name`, `detection_score`, `attribute_name`. Global coordinates. |
| `metrics_summary.json` | Scores from the official nuScenes evaluator: `mean_ap`, `nd_score`, the five error metrics (`tp_errors`), and per-class AP and errors. |
| `metrics_details.json` | Precision/recall curves per class and match distance (0.5, 1, 2, 4 m) behind those scores. |

## Regenerating

Inside the container, from `/workspace/bevfusion`:

```bash
# one run (add --degrade ... for a degraded condition)
torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth --eval bbox \
  --eval-options jsonfile_prefix=results/my_run

# every offline degradation condition -> results/degradation/fusion/nusc/<condition>/
python tools/run_degradation_sweep.py --save-nusc-json

# a saved ROS 2 run -> <prefix>/keyframes/, realtime/, realtime_extrapolated/
python ros2/evaluate_ros_results.py results/ros/stress/rate1.pkl \
  --jsonfile-prefix results/ros/stress/nusc/rate1
```

`ros2/run_timing_stress.sh` saves these files for every run automatically.
