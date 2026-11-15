# BEVFusion in ROS 2 — Timing Stress Tests

How BEVFusion (camera + LiDAR) behaves as a live ROS 2 node when data arrives faster than it can process, when the camera clock is wrong, when camera images arrive late, and when the inputs are degraded. Pipeline and terms: [../docs/ros2_pipeline.md](../docs/ros2_pipeline.md). Correctness check: [ros2_baseline.md](ros2_baseline.md).

- **Date:** 2026-11-15
- **Code:** fork [christiankso/bevfusion](https://github.com/christiankso/bevfusion) @ [`027f77f`](https://github.com/christiankso/bevfusion/commit/027f77f), script `ros2/run_timing_stress.sh`
- **Data:** nuScenes v1.0-mini val, 2 scenes: 788 LiDAR sweeps (20 Hz), 81 labeled keyframes, 39 s
- **Hardware:** RTX 4070 SUPER (12 GB), 20 CPU threads, WSL2 + Docker; replay and node on the same PC
- **Raw data:** [ros2/stress/](ros2/stress/) (one JSON per run)

## How to read the numbers

| Term | Meaning |
|---|---|
| **Throughput** | LiDAR sweeps the node finished per second (Hz). The sensor sends 20 Hz. |
| **Latency** | Time from a sweep reaching the node to its boxes being published. p50 = median, p90/p99 = 90%/99% of frames were faster. |
| **Model mAP** | Accuracy of the boxes computed from each keyframe's own sweep (what the model can do with no time pressure). |
| **Real-time mAP** | At the moment each keyframe arrives, take the **newest boxes the node had already published**, move them to cancel the car's own motion since their sweep, and score those. This is what a planner would actually have: it includes the effect of latency and dropped sweeps. |
| **Extrapolated** | Real-time, but each detected object is also moved forward by its predicted velocity for the elapsed time. |
| **Box age** | How old the newest available boxes are at a keyframe (data time). |

Reference points: the offline model scores mAP 0.582 / NDS 0.584; the node with no time pressure scores 0.578 / 0.579 ([ros2_baseline.md](ros2_baseline.md)).

## 1. Real-time baseline (1×, repeated 3 times)

| Run | Throughput | Latency p50 / p90 / p99 (ms) | Box age p50 | Real-time mAP / NDS | Extrapolated mAP / NDS |
|---|---:|---:|---:|---:|---:|
| 1 | 2.44 Hz | 423 / 485 / 765 | 625 ms | 0.374 / 0.461 | 0.476 / 0.512 |
| 2 | 2.52 Hz | 413 / 464 / 645 | 599 ms | 0.379 / 0.465 | 0.486 / 0.522 |
| 3 | 2.49 Hz | 413 / 469 / 607 | 601 ms | 0.380 / 0.464 | 0.486 / 0.522 |
| **mean** | **2.48 Hz** | **416 / 473 / 672** | **~610 ms** | **0.378 / 0.463** | **0.483 / 0.519** |

Run-to-run spread is under ±0.005 mAP, so differences larger than about 0.01 between conditions below are real.

**Latency breakdown** (median, ms): queue 30 · camera sync 33 · preprocessing 136 · inference 218 · publish 3.

**Findings**

- **The node finishes about 1 in 8 LiDAR sweeps** (2.5 of 20 per second). Each frame takes ~0.4 s of work.
- **Latency costs about a third of the model's accuracy.** Real-time mAP is 0.378 vs 0.578 with no time pressure (−35%). The boxes a planner has are ~0.6 s old, and other road users have moved in that time.
- **Using the predicted velocities recovers half of that loss** (0.378 → 0.483). BEVFusion's velocity estimates, which come from the merged LiDAR sweeps, are valuable for real-time use.
- **Preprocessing is 62% as long as inference.** It runs in Python on the CPU. Decoding the six JPEGs alone is ~50 ms (preprocessing drops from 136 to 87 ms when cameras are blacked out and the JPEGs are never decoded).

**Resources** (mean of the three runs):

| Resource | Mean | Max | Notes |
|---|---:|---:|---|
| CPU (node process) | 1.6 cores | 3.2 cores | 100% = one core; 20 available |
| RAM (node process) | 7.1 GB | 7.2 GB | Model, libraries, message buffers |
| GPU utilization | 37% | ~90% | GPU idles while the CPU prepares the next frame |
| GPU memory, PyTorch tensors | — | 2.5 GB | Peak memory actually used by the model |
| GPU memory, whole card | — | 11.6 / 12.3 GB | Includes PyTorch's reserved cache, the CUDA runtime, and the Windows desktop (~1 GB idle) |

## 2. Replay faster than real time

The replay sends data 2× and 4× faster, as if the computer were 2–4× slower than needed.

| Speed | LiDAR input | Throughput | Box age p50 | Keyframes with usable boxes (≤ 2 s old) | Real-time mAP | Extrapolated mAP |
|---|---:|---:|---:|---:|---:|---:|
| 1× | 20 Hz | 2.5 Hz | 0.6 s | 75 / 81 | 0.378 | 0.483 |
| 2× | 40 Hz | 2.4 Hz | 1.25 s | 69 / 81 | 0.289 | 0.375 |
| 4× | 80 Hz | 2.5 Hz | 1.8 s | 14 / 81 | 0.025 | 0.030 |

- **Throughput stays at ~2.5 Hz regardless of input rate**; the extra sweeps are simply dropped. Latency stays ~420 ms in wall time.
- **But the boxes get older in data time**, so accuracy falls: −24% at 2×, and at 4× almost every box is more than 2 s old by the time it is used. A detector that is 2–4× too slow for its sensors is not just slower; it describes a world that has already changed.

## 3. Camera clock offset

The camera's timestamps read later than the true capture time (an unsynchronized clock). The node pairs each LiDAR sweep with an image that is actually older than it believes, and places it using the car's pose at the wrong moment. These runs process only keyframes, so nothing is dropped and the change comes only from the misalignment.

Scored on the 69 keyframes every run processed (other keyframes count as empty for all runs equally, so absolute values are below the full 0.578; compare rows with each other):

| Camera clock offset | mAP | Change | NDS | mAOE (heading error) |
|---|---:|---:|---:|---:|
| 0 ms | 0.504 | — | 0.546 | 0.443 |
| 50 ms | 0.503 | 0% | 0.545 | 0.448 |
| 100 ms | 0.491 | −2.6% | 0.539 | 0.465 |
| 200 ms | 0.480 | −4.8% | 0.534 | 0.457 |
| 500 ms | 0.455 | −9.7% | 0.523 | 0.453 |
| **all cameras off** | **0.452** | −10.3% | 0.525 | 0.446 |

- **Up to ~50 ms of offset is harmless.** nuScenes cameras are already up to ~45 ms from the LiDAR timestamp, and the model was trained with that.
- **Beyond that, the camera's contribution fades steadily**, and at 500 ms the camera is worth about as much as no camera at all.
- **A misaligned camera did not make results worse than no camera** in this test. An earlier, uncorrected comparison suggested it did (0.460 vs 0.526); that came from the 500 ms run skipping 10 keyframes, not from the misalignment.
- The node also skips a few more frames as the offset grows (78 → 71 of 81 keyframes processed at 500 ms), because early in each scene there is no camera image close enough in time yet.

## 4. Camera transport delay

Camera timestamps are correct, but images arrive late (slow link or driver). The node waits up to its sync timeout (default 200 ms) for an image newer than the LiDAR sweep, then rejects any image more than 100 ms from the sweep.

| Camera delay | Sync timeout | Sweeps processed | Sync wait p50 | Latency p50 | Throughput | Real-time mAP | Extrapolated mAP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 ms | 200 ms | 98 | 33 ms | 416 ms | 2.5 Hz | 0.378 | 0.483 |
| 100 ms | 200 ms | 79 | 140 ms | 538 ms | 2.0 Hz | 0.357 | 0.457 |
| **300 ms** | **200 ms** | **0** | — | — | **0** | **0** | **0** |
| 300 ms | 500 ms | 55 | 343 ms | 747 ms | 1.4 Hz | 0.312 | 0.411 |

- **With the default timeout, a 300 ms camera delay stops the fusion node completely.** Every sweep times out waiting for its images and is rejected (237 skipped). The node is running but publishes nothing, which is a silent failure for downstream modules.
- **Raising the timeout restores output at the cost of latency.** The node then waits for the late images on every frame: latency +330 ms, throughput −44%, real-time mAP −17%.
- This trade-off only exists for a fusion model. A LiDAR-only detector would not wait for cameras; a camera-only one would not wait for LiDAR.

## 5. Speed under input degradation

Same 1× real-time setup, with an input degradation applied in the node.

| Condition | Preprocess p50 | Inference p50 | Latency p50 | Throughput | Real-time mAP | Extrapolated mAP |
|---|---:|---:|---:|---:|---:|---:|
| clean (mean of 3) | 136 ms | 218 ms | 416 ms | 2.5 Hz | 0.378 | 0.483 |
| no previous sweeps | 128 ms | 132 ms | 354 ms | 3.0 Hz | 0.344 | 0.356 |
| 75% of points dropped | 138 ms | 142 ms | 378 ms | 2.9 Hz | 0.366 | 0.458 |
| images darkened to 0.25× | 164 ms | 216 ms | 442 ms | 2.3 Hz | 0.351 | 0.450 |

- **Fewer LiDAR points make inference much faster** (218 → 132–142 ms). The LiDAR branch (sparse 3D convolutions) is a large share of the model's time and scales with point count.
- **The speed-up partly offsets the accuracy loss.** With 75% of points dropped, the model is less accurate (offline −8% mAP) but its boxes are fresher, so real-time mAP barely changes (0.378 → 0.366).
- **Without previous sweeps, velocity extrapolation stops helping** (0.344 → 0.356 vs 0.378 → 0.483 when clean). The velocity estimates come from the time offsets between sweeps; with one sweep they are poor, matching the offline finding (velocity error 0.42 → 1.02 m/s).
- **Darkening costs no model time.** The +28 ms preprocessing is the simulation itself (brightness adjustment on six full-size images in Python), not something a real dark camera would add.

## Limitations

- **Same machine:** the replay shares CPU with the node, so throughput on a dedicated vehicle computer could be somewhat higher.
- **Latency starts at the node.** Sensor capture (a LiDAR sweep takes 50 ms) and driver delays are not included.
- **One run per condition** except the 1× baseline (3 runs, ±0.005 mAP spread). Small differences between degradation rows (about 0.01–0.03) should be read with care.
- **Real-time mAP moves boxes to cancel the vehicle's own motion** (and optionally by predicted velocity), but nothing is tracked over time. A real system with an object tracker could do better.
- **Small evaluation set:** 81 keyframes from 2 scenes.
- **Unexplained:** the 75%-point-drop run reached a higher PyTorch GPU memory peak (3.6 GB vs 2.5 GB) despite fewer points; not investigated.
