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

The node always processes the newest sweep and drops any that arrived while it was busy, like a deployed system.

| Measure | Value |
|---|---:|
| LiDAR sweeps received | 788 (20 Hz) |
| Sweeps processed | **131 (17%) ≈ 3.3 Hz** |
| Sweeps dropped (node busy) | 657 |
| Labeled keyframes processed exactly | 14 of 81 |

**Latency per processed sweep** (ms), from the LiDAR message arriving to boxes being published:

| Stage | What it is | p50 | p90 | p99 |
|---|---|---:|---:|---:|
| queue | Waiting for the worker to pick the sweep up | 43 | 78 | 90 |
| sync | Waiting for the matching camera images | 20 | 77 | 159 |
| preprocess | Decode 6 JPEGs, merge 10 sweeps (~233k points), resize/normalize | 135 | 145 | 175 |
| inference | BEVFusion on the GPU | 224 | 245 | 266 |
| publish | Build and send the ROS 2 message | 3 | 4 | 5 |
| **total latency** | | **438** | **494** | **637** |

(p50 = median; p90 / p99 = 90% / 99% of frames were faster than this.)

### What this means

- **BEVFusion keeps up with about 1 in 6 LiDAR sweeps.** One frame takes ~360 ms of compute (preprocess + inference), so the node can do about 3 frames per second, while the LiDAR produces 20.
- **Boxes describe the world as it was ~0.44 s earlier.** At 50 km/h (14 m/s) the car travels about 6 m in that time, so a planner using these boxes must account for the delay.
- **Preprocessing costs over half as much as the model itself** (135 ms vs 224 ms). It runs in Python on the CPU (JPEG decoding, point merging), so it is the easiest target for speedups, e.g. GPU image decoding or merging sweeps incrementally.
- **Camera sync adds 20 ms typically, up to ~160 ms.** A fusion model has to wait for all six cameras; a LiDAR-only model would not.

### Why the keyframe-only mAP is not reported as the real-time accuracy

Scoring only the 14 keyframes the node happened to process, and counting the other 67 as "nothing detected", gives mAP 0.044. That understates what a vehicle would have: at each keyframe the node had already published boxes from an earlier sweep. With ~440 ms latency and a new output every ~300 ms, those boxes are roughly 0.4–0.8 s old (an estimate from the timing above, not yet measured per keyframe). A fair real-time accuracy measure uses, at each keyframe, the most recent boxes available at that moment, shifted for the car's own movement since. That evaluation is the next step.

## Limitations

- Replay and detector share one PC, so the replay's file reading and message sending take CPU time from the node.
- Latency starts when the node receives the LiDAR message; the scan itself takes 50 ms and the real sensor driver adds its own delay.
- Single GPU and single run; the p99 and max values come from a few frames.
