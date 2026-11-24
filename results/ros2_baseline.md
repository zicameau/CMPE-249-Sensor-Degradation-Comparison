# BEVFusion in ROS 2 — Correctness and Real-Time Baseline

First results from running BEVFusion (camera + LiDAR) as a live ROS 2 node on replayed nuScenes-mini validation data. Pipeline description and terms: [../docs/ros2_pipeline.md](../docs/ros2_pipeline.md).

- **Date:** 2026-11-15
- **Code:** fork [christiankso/bevfusion](https://github.com/christiankso/bevfusion) @ [`395273f`](https://github.com/christiankso/bevfusion/commit/395273f)
- **Data:** nuScenes v1.0-mini val (scene-0103, scene-0916): 788 LiDAR sweeps, 81 labeled keyframes, ~40 s
- **Hardware:** RTX 4070 SUPER, 20 CPU threads, WSL2 + Docker; replay and node on the same machine
- **Raw data:** [ros2/correctness_rate0.1.json](ros2/correctness_rate0.1.json), [ros2/realtime_rate1.json](ros2/realtime_rate1.json)

## 1. Correctness: does the node compute the same thing as the offline test?

Replay at 0.1× speed with `--process-all`, so every sweep is processed and nothing is dropped. All 81 keyframes processed.

| Metric | Offline `test.py` | ROS 2 node | Difference |
|---|---:|---:|---:|
| mAP | 0.5818 | 0.5782 | −0.6% |
| NDS | 0.5837 | 0.5785 | −0.9% |
| mATE (position error, m) | 0.404 | 0.406 | |
| mAOE (heading error, rad) | 0.472 | 0.488 | |
| mAVE (velocity error, m/s) | 0.423 | 0.435 | |

| Class | Offline AP | ROS 2 AP |
|---|---:|---:|
| car | 0.926 | 0.924 |
| truck | 0.815 | 0.805 |
| bus | 0.994 | 0.994 |
| pedestrian | 0.926 | 0.924 |
| motorcycle | 0.734 | 0.726 |
| bicycle | 0.534 | 0.524 |
| traffic cone | 0.888 | 0.884 |

**Verdict: the node reproduces the offline model.** The differences are within run-to-run noise on 81 frames. The likely source of the small gap: the offline loader picks 9 of the 10 previous sweeps at random, while the node always uses the 9 most recent. Camera images were matched within 45 ms of the LiDAR timestamp at every keyframe.

## 2. Real time: replay at 1× (newest-frame mode)

**Correction (2026-11-15):** the first real-time run reported here (3.3 Hz throughput, 438 ms latency) was wrong. The replay publisher could not keep real time; each 19.5 s scene took ~26 s, so it was effectively 0.75× speed and the throughput was divided by the wrong duration. After fixing the publisher, three runs at true 1× speed give:

| Measure | Value |
|---|---:|
| LiDAR input | 20 Hz |
| Throughput | **2.5 Hz** (about 1 in 8 sweeps) |
| Latency p50 / p90 / p99 | **416 / 473 / 672 ms** |
| Real-time mAP / NDS | 0.378 / 0.470 |
| Real-time mAP / NDS, velocity-extrapolated | 0.483 / 0.526 |

Real-time NDS corrected on 2026-11-23 (previously 0.463 / 0.519): the evaluator had rotated moved boxes' headings the wrong way. mAP values are unaffected; details in [ros2_timing_stress.md](ros2_timing_stress.md).

**Latency per processed sweep** (median, ms): queue 30 · camera sync 33 · preprocessing 136 (decode 6 JPEGs, merge 10 sweeps of ~233k points, resize/normalize) · inference 218 · publish 3.

Real-time accuracy uses, at each keyframe, the newest boxes the node had already published, moved to cancel the car's own motion; counting only the few keyframes the node processed exactly would understate it. Full results, definitions, resource use, and the timing stress tests: [ros2_timing_stress.md](ros2_timing_stress.md).

## Limitations

- Replay and detector share one PC, so the replay's file reading and message sending take CPU time from the node.
- Latency starts when the node receives the LiDAR message; the scan itself takes 50 ms and the real sensor driver adds its own delay.
- Single GPU and single run; the p99 and max values come from a few frames.
