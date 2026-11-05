# BEVFusion Camera+LiDAR Under Sensor Degradation — nuScenes-mini

How the BEVFusion fusion model's 3D detection accuracy changes as the LiDAR or camera input is degraded, one stressor at a time. Clean baseline: [baseline_fusion.md](baseline_fusion.md). Degradation options and how to rerun: [../docs/degradation.md](../docs/degradation.md).

- **Date:** 2026-11-05
- **Model / checkpoint:** BEVFusion camera+LiDAR, `pretrained/bevfusion-det.pth` (unmodified)
- **Code:** fork [christiankso/bevfusion](https://github.com/christiankso/bevfusion) @ [`f720c2f`](https://github.com/christiankso/bevfusion/commit/f720c2f)
- **Data:** nuScenes v1.0-mini val, 81 keyframes
- **Hardware:** RTX 4070 SUPER, WSL2 + Docker
- **Command:** `python tools/run_degradation_sweep.py` (inside the container)
- **Raw data:** [degradation/fusion/summary.csv](degradation/fusion/summary.csv) and one JSON per condition in [degradation/fusion/](degradation/fusion/)

## Summary

| Stressor | Mildest level tested | Where mAP drops more than 10% | Worst level tested |
|---|---|---|---|
| LiDAR random point drop | 25%: −1.1% | between 75% (−8.4%) and 90% (−23.1%) | 90%: −23.1% |
| LiDAR beams (of 32) | 16: −8.6% | between 16 and 4 beams (−14.8%) | 1 beam: −55.1% |
| LiDAR sweeps merged (of 10) | 5: −1.4% | at 1 sweep (keyframe only): −13.2% | NDS −23.7% (velocity) |
| LiDAR max range | 40 m: −4.7% | between 40 m and 30 m (−16.2%) | 10 m: −87.1% |
| Camera brightness | 0.5×: −2.2% | at 0.1× (−10.3%) | 0.1×: −10.3% |
| Camera blur | 2 px: −1.8% | at 10 px (−10.1%) | 10 px: −10.1% |
| Cameras dropped | front: −3.4% | never (all 6 cameras off: −9.6%) | all: −9.6% |

All changes are relative to the clean mAP of 0.582.

**Key findings**

1. **The fusion model depends mainly on LiDAR.** Turning off all six cameras costs only 9.6% mAP (0.582 → 0.526). Every camera stressor levels off at about that same value: brightness 0.1× (0.522), blur 10 px (0.523), all cameras black (0.526). So the camera branch adds about 0.056 mAP in total, and no camera failure can remove more than that. LiDAR failures, by contrast, can remove almost everything (10 m range: 0.075 mAP).
2. **Losing beams hurts much more than losing the same number of points at random.** 16 beams keeps about half the points and costs −8.6% mAP; dropping 50% of points at random costs only −2.3%. Random dropout leaves every scan line partly intact; removing beams leaves vertical gaps in which whole objects can disappear.
3. **Random sparsity is tolerated up to about 75%.** Each frame merges 10 sweeps (~200k points), so there is a lot of redundancy. Accuracy falls sharply only at 90%.
4. **Missing scans mainly break velocity and heading, not detection.** With only the keyframe (no previous sweeps), mAP drops 13.2% but NDS drops 23.7%: velocity error rises from 0.42 to 1.02 m/s and orientation error from 0.47 to 0.78 rad. The model infers motion from the time offsets between merged sweeps.
5. **Range loss is the most damaging stressor, and the camera does not compensate.** At a 20 m LiDAR range mAP falls 43% even though all six cameras still see far objects. In BEVFusion the camera branch uses LiDAR points as depth hints, so camera features are less useful where there is no LiDAR. Traffic cones, which are mostly close to the car, hold up best (0.888 → 0.832 at 20 m).
6. **Small, thin objects depend most on the camera.** Traffic cones (0.888 → 0.552 with all cameras off) and bicycles (0.534 → 0.442) are hit hardest by camera loss. Cars, buses, and pedestrians barely change (pedestrian 0.926 → 0.930), so LiDAR alone carries them.

## Full results

ΔmAP and ΔNDS are relative to the clean run. Per-class columns are AP. Trailer, construction vehicle, and barrier are omitted: mini-val has none (AP 0 in every condition).

| Group | Condition | mAP | ΔmAP | NDS | ΔNDS | car | truck | bus | ped. | moto. | bicycle | cone |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | clean | 0.582 | — | 0.584 | — | 0.926 | 0.815 | 0.994 | 0.926 | 0.734 | 0.534 | 0.888 |
| LiDAR point drop | 25% | 0.576 | −1.1% | 0.579 | −0.8% | 0.920 | 0.806 | 0.994 | 0.922 | 0.721 | 0.515 | 0.879 |
| | 50% | 0.568 | −2.3% | 0.575 | −1.5% | 0.907 | 0.806 | 0.981 | 0.911 | 0.709 | 0.492 | 0.876 |
| | 75% | 0.533 | −8.4% | 0.551 | −5.7% | 0.886 | 0.795 | 0.975 | 0.859 | 0.634 | 0.404 | 0.773 |
| | 90% | 0.447 | −23.1% | 0.479 | −18.0% | 0.816 | 0.759 | 0.913 | 0.737 | 0.511 | 0.245 | 0.491 |
| LiDAR beams | 16 | 0.532 | −8.6% | 0.551 | −5.6% | 0.829 | 0.804 | 0.986 | 0.860 | 0.631 | 0.426 | 0.782 |
| | 4 | 0.496 | −14.8% | 0.533 | −8.7% | 0.814 | 0.788 | 0.968 | 0.837 | 0.585 | 0.370 | 0.593 |
| | 1 | 0.261 | −55.1% | 0.368 | −37.0% | 0.425 | 0.475 | 0.642 | 0.339 | 0.322 | 0.100 | 0.308 |
| LiDAR sweeps | 4 prev. (5 total) | 0.574 | −1.4% | 0.575 | −1.5% | 0.922 | 0.821 | 0.994 | 0.923 | 0.711 | 0.550 | 0.817 |
| | 1 prev. (2 total) | 0.533 | −8.4% | 0.532 | −8.9% | 0.908 | 0.804 | 0.993 | 0.902 | 0.673 | 0.345 | 0.706 |
| | 0 prev. (keyframe) | 0.505 | −13.2% | 0.445 | −23.7% | 0.880 | 0.773 | 0.950 | 0.877 | 0.572 | 0.333 | 0.667 |
| LiDAR range | 40 m | 0.554 | −4.7% | 0.569 | −2.6% | 0.881 | 0.707 | 0.889 | 0.921 | 0.726 | 0.530 | 0.889 |
| | 30 m | 0.487 | −16.2% | 0.534 | −8.5% | 0.765 | 0.602 | 0.787 | 0.790 | 0.612 | 0.434 | 0.883 |
| | 20 m | 0.330 | −43.3% | 0.445 | −23.8% | 0.460 | 0.340 | 0.543 | 0.510 | 0.386 | 0.226 | 0.832 |
| | 10 m | 0.075 | −87.1% | 0.221 | −62.1% | 0.122 | 0.040 | 0.000 | 0.073 | 0.089 | 0.018 | 0.407 |
| Camera brightness | 0.5× | 0.569 | −2.2% | 0.577 | −1.2% | 0.924 | 0.781 | 0.994 | 0.925 | 0.731 | 0.455 | 0.880 |
| | 0.25× | 0.545 | −6.4% | 0.566 | −3.0% | 0.922 | 0.752 | 0.994 | 0.919 | 0.653 | 0.393 | 0.814 |
| | 0.1× | 0.522 | −10.3% | 0.556 | −4.8% | 0.902 | 0.785 | 0.993 | 0.927 | 0.659 | 0.419 | 0.531 |
| Camera blur | 2 px | 0.571 | −1.8% | 0.576 | −1.2% | 0.925 | 0.804 | 0.994 | 0.921 | 0.729 | 0.437 | 0.900 |
| | 5 px | 0.547 | −6.1% | 0.567 | −2.8% | 0.913 | 0.757 | 0.994 | 0.910 | 0.656 | 0.379 | 0.857 |
| | 10 px | 0.523 | −10.1% | 0.556 | −4.8% | 0.891 | 0.715 | 0.992 | 0.915 | 0.638 | 0.391 | 0.689 |
| Cameras dropped | front | 0.562 | −3.4% | 0.575 | −1.4% | 0.916 | 0.803 | 0.994 | 0.919 | 0.724 | 0.488 | 0.776 |
| | front + back | 0.557 | −4.2% | 0.575 | −1.5% | 0.911 | 0.797 | 0.994 | 0.920 | 0.676 | 0.522 | 0.754 |
| | all 6 | 0.526 | −9.6% | 0.558 | −4.3% | 0.903 | 0.774 | 0.994 | 0.930 | 0.663 | 0.442 | 0.552 |

### Error metrics for selected conditions

| Condition | mATE (m) | mASE | mAOE (rad) | mAVE (m/s) | mAAE |
|---|---:|---:|---:|---:|---:|
| clean | 0.404 | 0.447 | 0.472 | 0.422 | 0.327 |
| sweeps: 4 prev. | 0.407 | 0.448 | 0.493 | 0.455 | 0.319 |
| sweeps: 1 prev. | 0.425 | 0.453 | 0.541 | 0.607 | 0.322 |
| sweeps: keyframe only | 0.433 | 0.458 | 0.778 | 1.022 | 0.406 |
| beams: 1 | 0.536 | 0.493 | 0.791 | 0.485 | 0.323 |
| range: 20 m | 0.441 | 0.460 | 0.487 | 0.515 | 0.296 |
| cameras: all off | 0.404 | 0.438 | 0.470 | 0.410 | 0.323 |

Lower is better. With all cameras off, the error metrics are unchanged from clean: box position, size, heading, and velocity all come from LiDAR.

## Limitations

- **Small evaluation set.** 81 frames from 2 scenes. Differences under about 2% mAP are within run-to-run noise (the clean rerun gave 0.5818 vs 0.5822 earlier). Bicycle (44 objects), bus (41), and traffic cone (37) are especially noisy; for example, 2 px blur lowering bicycle AP from 0.534 to 0.437 is likely noise.
- **Synthetic stressors.** Darkening, blur, random dropout, and range cropping are controlled approximations. Real fog, rain, or night also add noise, glare, false returns, and correlated effects across sensors.
- **Range limit keeps far ground truth on purpose.** Objects beyond the range limit stay in the ground truth, so the score reflects what a car with that sensor would miss, not detector quality inside the range.
- **Timing is not model latency.** The sweep records wall time for the whole test loop (1 GPU), so it is dominated by data loading: 4–6 fps for most conditions and 10 fps with no previous sweeps, because 9 fewer LiDAR files are read per frame. Model latency and end-to-end ROS 2 latency are measured separately.
- **One model.** These are fusion-model results only. Comparing against the LiDAR-only checkpoint under the same LiDAR stressors (command in [../docs/degradation.md](../docs/degradation.md)) would show how much the camera helps under each LiDAR failure.
