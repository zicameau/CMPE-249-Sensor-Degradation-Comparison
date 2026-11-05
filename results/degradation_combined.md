# BEVFusion Fusion Robustness — Combined Degradation and Camera Rescue

What the camera+LiDAR fusion model can do that a single-sensor model cannot: how much the camera recovers when LiDAR is degraded, and how the model holds up when both sensors are degraded at once. Single-sensor results (one stressor at a time) are in [degradation_fusion.md](degradation_fusion.md).

- **Date:** 2026-11-05
- **Model / checkpoint:** BEVFusion camera+LiDAR, `pretrained/bevfusion-det.pth` (unmodified)
- **Code:** fork [christiankso/bevfusion](https://github.com/christiankso/bevfusion) @ [`d7d7342`](https://github.com/christiankso/bevfusion/commit/d7d7342)
- **Data:** nuScenes v1.0-mini val, 81 keyframes
- **Command:** `python tools/run_degradation_sweep.py --groups combined lidar_stress_no_cams`
- **Raw data:** [degradation/fusion/summary.csv](degradation/fusion/summary.csv) (groups `combined` and `lidar_stress_no_cams`)

All percentages are relative to the clean run: mAP 0.582, NDS 0.584.

## Summary

1. **The camera's value grows as LiDAR gets worse.** With clean LiDAR, the cameras add 0.056 mAP. With heavily degraded LiDAR they add 2–3.5× that: +0.115 at 90% point loss, +0.120 at 4 beams, +0.135 with no previous sweeps, +0.199 at 1 beam. At 1 beam, fusion scores 0.261 mAP; the same LiDAR without cameras scores 0.063.
2. **The camera does not recover lost range.** When LiDAR range is cut to 30 m or 20 m, the cameras add +0.058 and +0.050, no more than with clean LiDAR. Fusion fills in sparse or low-resolution LiDAR, but it cannot detect objects LiDAR does not reach. BEVFusion's camera branch uses LiDAR points as depth hints, so camera features beyond the LiDAR range get little help.
3. **Combined degradation is roughly additive, not catastrophic.** Each weather scenario's drop is close to the sum of its parts' single-sensor drops (estimates below). No scenario showed the two sensors failing together in a way worse than their separate effects.
4. **Range loss dominates the weather scenarios.** Heavy fog (25 m range) is the worst scenario at −36.8% mAP; heavy rain (35 m) is −18.9%. Scenarios without a range limit stay within about −11%.
5. **Boxes that are found stay accurate.** In every combined scenario, translation, orientation, and velocity errors stay close to clean (mATE 0.41 vs 0.40). Accuracy is lost through missed objects, not worse boxes.

## Camera rescue under LiDAR stress

Each LiDAR stressor is run twice: cameras on (the normal fusion model) and all six cameras off. **Camera gain** = mAP(cameras on) − mAP(cameras off).

| LiDAR condition | mAP, cameras on | mAP, cameras off | Camera gain | NDS on | NDS off |
|---|---:|---:|---:|---:|---:|
| clean | 0.582 | 0.526 | **+0.056** | 0.584 | 0.558 |
| 75% points dropped | 0.533 | 0.441 | **+0.091** | 0.551 | 0.506 |
| 90% points dropped | 0.447 | 0.332 | **+0.115** | 0.479 | 0.421 |
| 16 beams | 0.532 | 0.430 | **+0.102** | 0.551 | 0.498 |
| 4 beams | 0.496 | 0.376 | **+0.120** | 0.533 | 0.470 |
| 1 beam | 0.261 | 0.063 | **+0.199** | 0.368 | 0.246 |
| no previous sweeps | 0.505 | 0.371 | **+0.135** | 0.445 | 0.388 |
| range 30 m | 0.487 | 0.429 | **+0.058** | 0.534 | 0.510 |
| range 20 m | 0.330 | 0.280 | **+0.050** | 0.445 | 0.429 |

### Which classes the camera rescues

AP with cameras on / off. Bold: the camera recovers at least 0.2 AP.

| LiDAR condition | car | truck | bus | ped. | moto. | bicycle | cone |
|---|---|---|---|---|---|---|---|
| 90% points dropped | 0.816 / 0.695 | 0.759 / 0.592 | 0.913 / 0.878 | 0.737 / 0.640 | **0.511 / 0.303** | 0.245 / 0.127 | **0.491 / 0.086** |
| 4 beams | 0.814 / 0.725 | 0.788 / 0.609 | 0.968 / 0.964 | 0.837 / 0.783 | **0.585 / 0.380** | 0.370 / 0.289 | **0.593 / 0.007** |
| 1 beam | 0.425 / 0.287 | **0.475 / 0.118** | **0.642 / 0.046** | 0.339 / 0.147 | **0.322 / 0.019** | 0.100 / 0.009 | **0.308 / 0.000** |
| no previous sweeps | 0.880 / 0.804 | 0.773 / 0.627 | 0.950 / 0.951 | 0.877 / 0.842 | **0.572 / 0.278** | **0.333 / 0.073** | **0.667 / 0.130** |
| range 20 m | 0.460 / 0.364 | 0.340 / 0.306 | 0.543 / 0.542 | 0.510 / 0.497 | 0.386 / 0.354 | 0.226 / 0.177 | **0.832 / 0.562** |

Small objects (traffic cones, motorcycles, bicycles) are almost invisible to sparse LiDAR alone; with 4 beams and no cameras, traffic cone AP is 0.007, and the cameras bring it back to 0.593. Large objects (buses, and cars at moderate degradation) are carried by LiDAR and gain little. At 1 beam even buses depend on the camera (0.046 → 0.642).

## Combined weather and hardware scenarios

Both sensors degraded at once. These are approximations, not calibrated weather models: fog lowers image contrast and shortens LiDAR range; rain blurs the image, dims it, and scatters LiDAR returns (random point loss).

| Scenario | Camera degradation | LiDAR degradation | mAP | ΔmAP | NDS | ΔNDS |
|---|---|---|---:|---:|---:|---:|
| rain, light | brightness 0.8, blur 2 px | 25% points dropped | 0.561 | −3.6% | 0.571 | −2.1% |
| fog, light | contrast 0.6, blur 2 px | range 40 m, 25% dropped | 0.534 | −8.2% | 0.559 | −4.3% |
| night + rain | brightness 0.2, blur 3 px | 40% points dropped | 0.517 | −11.1% | 0.552 | −5.4% |
| hardware fault | front + front-left cameras dead | 16 beams | 0.491 | −15.6% | 0.531 | −9.1% |
| rain, heavy | brightness 0.6, blur 5 px | range 35 m, 50% dropped | 0.472 | −18.9% | 0.529 | −9.4% |
| fog, heavy | contrast 0.3, blur 5 px | range 25 m, 50% dropped | 0.367 | −36.8% | 0.478 | −18.0% |

| Scenario | car | truck | bus | ped. | moto. | bicycle | cone | mATE | mAOE | mAVE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.926 | 0.815 | 0.994 | 0.926 | 0.734 | 0.534 | 0.888 | 0.404 | 0.472 | 0.422 |
| rain, light | 0.919 | 0.781 | 0.994 | 0.919 | 0.707 | 0.404 | 0.885 | 0.407 | 0.490 | 0.415 |
| fog, light | 0.873 | 0.681 | 0.889 | 0.917 | 0.708 | 0.381 | 0.892 | 0.406 | 0.486 | 0.423 |
| night + rain | 0.887 | 0.764 | 0.994 | 0.916 | 0.587 | 0.391 | 0.634 | 0.408 | 0.467 | 0.422 |
| hardware fault | 0.797 | 0.763 | 0.985 | 0.836 | 0.555 | 0.369 | 0.602 | 0.422 | 0.524 | 0.438 |
| rain, heavy | 0.789 | 0.593 | 0.797 | 0.847 | 0.553 | 0.337 | 0.799 | 0.407 | 0.479 | 0.431 |
| fog, heavy | 0.550 | 0.441 | 0.651 | 0.647 | 0.381 | 0.255 | 0.749 | 0.407 | 0.470 | 0.445 |

### Are the effects additive?

Expected drop = sum of the single-sensor drops for each component (from [degradation_fusion.md](degradation_fusion.md)). Where a component level was not tested on its own, the value is interpolated between neighboring levels and marked ~.

| Scenario | Components (single-sensor ΔmAP) | Sum | Observed |
|---|---|---:|---:|
| rain, light | 25% drop −1.1%, blur 2 px −1.8%, brightness 0.8 ~−1% | ~−4% | −3.6% |
| fog, light | range 40 m −4.7%, 25% drop −1.1%, blur 2 px −1.8%, contrast 0.6 (not tested alone) | ≤ −7.6% | −8.2% |
| hardware fault | 16 beams −8.6%, front camera −3.4%, front-left camera (not tested alone) | ≤ −12.0% | −15.6% |
| rain, heavy | range 35 m ~−10%, 50% drop −2.3%, blur 5 px −6.1%, brightness 0.6 ~−2% | ~−20% | −18.9% |
| fog, heavy | range 25 m ~−30%, 50% drop −2.3%, blur 5 px −6.1%, contrast 0.3 (not tested alone) | ≤ −38% | −36.8% |

Observed drops land within a few percent of the summed estimates. The hardware fault is somewhat worse than its known parts, but the front-left camera's own effect was not measured, so this does not show an interaction. The estimates are rough (interpolated levels, untested components, 81 frames), so read this as "no large interaction", not as an exact additivity result.

## Limitations

- **Small evaluation set:** 81 frames from 2 scenes; differences under about 2% mAP are within noise, and bicycle (44 objects), bus (41), and traffic cone (37) APs are noisy.
- **Synthetic weather:** real fog and rain also cause false LiDAR returns near the sensor, glare, water on the lens, and spatially uneven effects. None of those are modeled. The scenario levels are illustrative choices.
- **Cameras-off is a strong condition:** black frames remove all image content. It measures the camera's total contribution, not a realistic failure.
- **Range limit keeps far ground truth:** objects beyond the LiDAR range stay in the ground truth on purpose, so the score reflects what the vehicle would miss.
