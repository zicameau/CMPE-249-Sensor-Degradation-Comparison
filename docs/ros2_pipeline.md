# ROS 2 Pipeline for BEVFusion

How BEVFusion runs as a live ROS 2 node fed by replayed nuScenes data, so latency and throughput can be measured the way a vehicle would experience them. Results: [../results/ros2_baseline.md](../results/ros2_baseline.md).

## Key terms

| Term | Meaning |
|---|---|
| **ROS 2** | Robot Operating System 2: messaging software that lets separate robot programs exchange data. Not an operating system. |
| **Foxy** | The ROS 2 release (2020) built for Ubuntu 20.04 and Python 3.8, the only one matching BEVFusion's container, so ROS 2 and BEVFusion run in the same Python. |
| **Node** | One ROS 2 program (here: the replay publisher and the BEVFusion detector). |
| **Topic** | A named channel nodes send messages on, e.g. `/LIDAR_TOP/points`. |
| **Publish / subscribe** | Send messages on a topic / receive them. |
| **Message type** | The fixed format of a message, e.g. `sensor_msgs/PointCloud2` for a LiDAR sweep. |
| **tf** | ROS 2's standard way to say where things are relative to each other: the car on the map (`/tf`), each sensor on the car (`/tf_static`). |
| **Latched topic** | Keeps its last message for nodes that subscribe later (used for calibration, which rarely changes). |
| **Sweep** | One full 360° LiDAR rotation (20 per second in nuScenes). |
| **Keyframe** | A sweep with hand-labeled boxes (2 per second); only keyframes can be scored. |
| **Latency** | Time from a LiDAR sweep reaching the detector to its boxes being published. |
| **Throughput** | Frames processed per second. |
| **Dropped frame** | A sweep skipped because the detector was still busy with an earlier one. |

## Architecture

```
nuscenes_replay.py (publisher)                              bevfusion_node.py (detector)
 reads nuScenes files in time order        topics            callbacks: buffer messages only
 ─────────────────────────────────── ─────────────────────► worker thread:
 /CAM_FRONT … /CAM_BACK_RIGHT/image/compressed  (6 × ~12 Hz)   1. take newest LiDAR sweep (drop older)
 /<CAM>/camera_info  (lens intrinsics, latched)                2. sync: nearest image from each camera
 /LIDAR_TOP/points   (20 Hz)                                   3. merge 9 previous sweeps using ego poses
 /tf        map → base_link (car pose, every message)          4. same preprocessing as offline test
 /tf_static base_link → each sensor (mounting, latched)        5. BEVFusion inference on GPU
                                                               6. publish /bevfusion/detections
                                                                  (vision_msgs/Detection3DArray)
                                                               7. log timing per stage
```

Code is in the `bevfusion/` submodule, folder `ros2/` (fork commit [`395273f`](https://github.com/christiankso/bevfusion/commit/395273f)):

| File | Role |
|---|---|
| `ros2/nuscenes_replay.py` | Publishes camera JPEGs, LiDAR sweeps, car pose, and calibration with the original nuScenes timing. Uses only nuscenes-devkit and ROS 2, so **any detector node can subscribe**. `--rate 2` plays twice as fast. |
| `ros2/bevfusion_node.py` | The fusion detector node. |
| `ros2/evaluate_ros_results.py` | Scores the node's saved output with the official nuScenes metrics and summarizes timing. |

## How the node rebuilds the model input

Offline, BEVFusion's data loader reads files and precomputed calibration. Live, the node must rebuild the same input from messages:

- **Camera–LiDAR sync.** Cameras and LiDAR fire at different times. For each sweep, the node picks each camera's image with the closest timestamp, waiting until an image after the sweep has arrived (so it knows which is closest) or until `--sync-timeout`. Images further than `--sync-tolerance-ms` (100 ms) from the sweep are rejected.
- **Calibration matrices.** Camera-to-LiDAR transforms are computed from the car's pose at each sensor's timestamp (`/tf`) and each sensor's mounting (`/tf_static`), exactly as nuScenes' converter does. The car moves between the camera and LiDAR timestamps, so the pose at each one matters.
- **Sweep merging.** BEVFusion uses the current sweep plus 9 previous ones. The node keeps recent sweeps, shifts each into the current LiDAR position using the car's movement between the two timestamps, and adds a "time ago" value per point, which is how the model estimates velocity.
- **Same preprocessing.** Image resizing, normalization, point range filter, and degradation (optional `--degrade`, same options as `tools/test.py`) come from the model's own test pipeline.

## Running

Inside the container, in two terminals (`docker exec -it bevf bash`, then `cd /workspace/bevfusion && source /opt/ros/foxy/setup.bash` in each):

```bash
# terminal 1: detector (loads in ~30 s; wait for "BEVFusion node ready")
python ros2/bevfusion_node.py --idle-timeout 10 --save-results results/ros/run.pkl

# terminal 2: replay the 2 mini-val scenes in real time (~40 s)
python ros2/nuscenes_replay.py --rate 1
```

Then score it:

```bash
python ros2/evaluate_ros_results.py results/ros/run.pkl --out results/ros/run.json
```

Useful options:

| Option | Node / replay | Effect |
|---|---|---|
| `--rate 0.1` | replay | Play at a tenth of real time (no frames dropped) |
| `--process-all` | node | Queue every sweep instead of dropping; use with a slow `--rate` and `--sync-timeout 3` for accuracy checks |
| `--keyframes-only` | node | Process only the 81 labeled keyframes (all sweeps still buffered for merging); keeps up at 1× so accuracy can be measured without dropped frames |
| `--sweeps 0` | node | Number of previous LiDAR sweeps to merge (default 9) |
| `--sync-timeout 0.5` | node | Max seconds to wait for camera images (default 0.2) |
| `--degrade point_drop=0.5 brightness=0.25` | node | Same degradation options as the offline sweep, applied live |
| `--idle-timeout 10` | node | Exit and save results 10 s after the last LiDAR message |
| `--camera-clock-offset-ms 100` | replay | Camera timestamps read 100 ms later than the true capture time (unsynchronized clock) |
| `--camera-delay-ms 100` | replay | Correct timestamps, but images delivered 100 ms late (slow link) |

The evaluator reports two accuracies: **model mAP** (boxes computed from each keyframe's own sweep) and **real-time mAP** (the newest boxes already published when each keyframe arrived, moved to cancel the car's motion; plus a velocity-extrapolated version). It also reports throughput, per-stage latency percentiles, and CPU/GPU/memory use (`--gpu-log` takes an `nvidia-smi` CSV).

### Timing stress suite

```bash
bash ros2/run_timing_stress.sh            # all runs, ~25 min; finished runs are skipped
bash ros2/run_timing_stress.sh rate2 delay100   # selected runs
```

Runs: replay at 1× (×3), 2×, 4×; camera clock offset 0–500 ms and all cameras off (keyframes only); camera delay 100/300 ms (and 300 ms with a 500 ms sync timeout); no previous sweeps, 75% point drop, darkened images. Output in `results/ros/stress/`: per run a `.pkl` (raw), `.json` (scored), node/replay/eval logs, and a GPU CSV. `ros2/compare_common_keyframes.py` re-scores several runs on the keyframes all of them processed. Results: [../results/ros2_timing_stress.md](../results/ros2_timing_stress.md).

You can watch topics from a third terminal with `ros2 topic list`, `ros2 topic hz /bevfusion/detections` (output rate), or `ros2 topic echo /bevfusion/detections`.

## Notes

- Detections are published in the LiDAR frame (`frame_id: LIDAR_TOP`). `Detection3D` has no velocity field, so velocity is only kept in the saved results file.
- Latency starts when the node receives the LiDAR message, not when the laser fired. Transport between two processes on one machine adds about a millisecond, which is ignored.
- The replay and node run on the same PC, so they compete for CPU. A real vehicle has dedicated sensor drivers.
