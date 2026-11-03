# LiDAR lane (Trevor)

Owns: BEVFusion LiDAR-only checkpoint + own Docker env, reproduction of published LiDAR-only results,
LiDAR corruption sweeps, LiDAR-only ROS 2 node, Sonata feature probe, CenterPoint (stretch).
Shared piece: **LiDAR corruption node** (point drop, beam reduction, range limit, missing sweeps).

## Status

| Piece | State |
|---|---|
| Corruptions (`sdc_lidar/corruptions.py`) | Done, unit-tested |
| Multi-sweep accumulation with corruption on raw sweeps (`multisweep.py`) | Done, unit-tested |
| Shared results CSV writer + stats: scene bootstrap, paired bootstrap, RR/mRR, failure threshold, latency (`common/sdc_common`) | Done, unit-tested |
| **Bootstrap evaluation** (`common/sdc_common/nusc_eval.py`, `robustness.py`, `nusc_loader.py`, `common/tools/bootstrap_sweep.py`): nuScenes mAP / NDS / 5 TP errors / per-class AP with scene-bootstrap CIs, RR, mRR, failure threshold | Core verified against the official devkit on synthetic data (exact match). **Real-data check done on nuScenes mini**: the bootstrap's clean mAP/NDS (0.5424 / 0.5655) equal the devkit's `metrics_summary.json` (0.54236 / 0.56546). Not yet on the full val set |
| Feature-stability metric (`feature_stability.py`) | Done, unit-tested |
| ROS 2 corruption node (`ros2/lidar_corruption`) | **Run end to end** in `Dockerfile.ros` (ROS 2 Foxy): `colcon build`, `ros2 run`, live `ros2 param set`, real nuScenes sweep in; output identical to the offline corruption code (`scripts/ros_corruption_check.sh`, all checks pass) |
| LiDAR-only ROS 2 detector node (`ros2/lidar_detector`, `sdc_lidar/detector_core.py`) | **Run end to end** (GPU, fp16 like `tools/test.py`). Replaying real mini-val samples (sweeps + TF) matches the offline `results_nusc.json` for the same samples to ~1-4 mm / 0.002 score, clean and with `range_limit` set live (`scripts/ros_detector_check.sh`). ~0.3-0.5 s per frame on the laptop RTX 4050 (not real time; see risks) |
| BEVFusion plugin + sweep driver (`bevfusion_plugin.py`, `bevfusion/run_test.py`, `scripts/run_corruption_sweep.py`) | **Run on mini** (clean + `range_limit` s3: mAP 0.542 -> 0.454, NDS 0.565 -> 0.520). Full-val status below |
| nuScenes info files (`scripts/make_infos.py`) | Replaces `tools/create_data.py`, which crashes at the pinned commit (see Runbook). Output verified structurally identical to the converter's own |
| Distance-binned recall (`common/tools/distance_recall.py`) | Done, unit-tested; run on mini (`range_limit` s3: recall unchanged <30 m, 0.36 at 30-40 m, 0 beyond 40 m) |
| Dockerfile (`docker/Dockerfile`) | Built; smoke test 8/8 on the laptop RTX 4050; runs the real pipeline. Two upstream import bugs at the pinned commit are patched in the Dockerfile (`feature_decorator_ext`, `flash_attn`) |
| `Dockerfile.ros` | Built (ROS 2 Foxy on the BEVFusion image) |
| `Dockerfile.sonata` + Sonata probe (`scripts/sonata_probe.py`) | Built (CUDA 12.1: the laptop driver supports <= 12.3) and **run on mini** (2 scenes, 3 frames: cosine and coverage degrade monotonically with severity). Full-size run: see below |
| CenterPoint (stretch, `configs/centerpoint_voxelnet_0p075.yaml`, `scripts/convert_centerpoint_ckpt.py`) | **Runs through the same pipeline and sweep driver; weights converted from mmdet3d's v0.1.0 model zoo (504/504 tensors, no key warnings).** Mini clean mAP 0.518 / NDS 0.550 (TransFusion-L on the same 81 samples: 0.542 / 0.565), `range_limit` s3 -> 0.430 / 0.501. Full-val published-number check pending (see "CenterPoint") |

Run the tests on any machine: `pip install numpy scipy pytest && python -m pytest`. Two test modules compare
against the official devkit and skip themselves unless `nuscenes-devkit` is installed (use a venv; it pins an
old numpy): `pip install nuscenes-devkit scipy pytest`.

## Corruptions

Severity 0 = clean, 1-5 harsher. Tables live at the top of `corruptions.py` (single source of truth).

| Corruption | Params for severity 1..5 | Random? |
|---|---|---|
| `point_drop` | drop prob 0.10, 0.25, 0.40, 0.60, 0.80 | yes, seeds 0/1/2 |
| `beam_reduction` | beams kept of 32: 24, 16, 12, 8, 4 (evenly spaced rings) | no |
| `range_limit` | max horizontal range 50, 40, 30, 20, 10 m | no |
| `missing_sweeps` | past sweeps dropped (of 9): 20, 40, 60, 80, 100 %; current sweep always kept | yes, seeds 0/1/2 |

These severity values are my choices, not from the proposal - **agree on them with the team**, and keep
them identical in the report.

Seeding: `frame_rng(seed, frame_key, sweep_idx)` - the result for a frame does not depend on processing
order, so reruns are bit-identical.

## How the offline evaluation works

`bevfusion_plugin.py` re-registers `LoadPointsFromMultiSweeps` in BEVFusion's pipeline registry with a
subclass. **No upstream file or config is edited.** `SDC_CORRUPTION` / `SDC_SEVERITY` / `SDC_SEED`
environment variables select the corruption; with none set it runs the original code, so the clean run is
the published pipeline. Corruptions are applied to each *raw* sweep before ego-motion alignment, because
BEVFusion overwrites the `ring` column with the time lag (beam reduction needs it).

`run_corruption_sweep.py` runs clean + every (corruption, severity, seed), one `torchpack dist-run`
process each, resumable, and appends rows to the shared CSV. Each run directory also keeps
`results_nusc.json` (the detections in nuScenes format) - **that file is the input Ianna's 3D-to-2D
projection scorer needs for the LiDAR-only column of Tier 1.**

## Runbook

1. **Data.** nuScenes v1.0 trainval (LiDAR blobs + metadata are enough: `scripts/fetch_nuscenes_trainval.ps1`)
   or mini first. Make the info files with `python /workspace/sdc/lidar/scripts/make_infos.py --root
   /dataset/nuscenes --version v1.0-trainval --out-prefix nuscenes --val-only` (BEVFusion's
   `tools/create_data.py` crashes at the pinned commit; `make_infos.py` calls the same converter and rebases the
   paths; on mini its output is identical to the converter's own). The configs expect `data/nuscenes/` with
   the `*_infos_val.pkl` file inside it; `NUSCENES_DIR` should point at the folder that *contains*
   `nuscenes/`, then inside the container run `ln -sfn /dataset data` in `/workspace/bevfusion`.
   Without the camera / radar blobs, set `SDC_SKIP_CAMERA_RADAR=1` (or `--skip-camera-radar` on the sweep
   driver): the LiDAR-only model never reads them (mini metrics identical with and without).
2. **Image.** `docker build -f lidar/docker/Dockerfile -t sdc-lidar .`
3. **Checkpoint.** `bash tools/download_pretrained.sh` (Dropbox; only `lidar-only-det.pth` is needed).
4. **Reproduce the published number first** (gate for everything else). Expected on val: **mAP 64.68, NDS
   69.28** (BEVFusion README, "LiDAR-Only Baseline" = TransFusion-L):
   ```
   torchpack dist-run -np 1 python /workspace/sdc/lidar/bevfusion/run_test.py \
     configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml pretrained/lidar-only-det.pth \
     --eval bbox --eval-options jsonfile_prefix=/results/clean
   ```
   Report the gap to the published numbers honestly; do not proceed on a model that does not reproduce.
   **Protocol note.** This gate uses the stock loader, which draws a *random* 9 of the 10 sweeps at test time
   (`LoadPointsFromMultiSweeps` has no `test_mode`). The sweep driver always sets `SDC_DETERMINISTIC_SWEEPS=1`
   (nearest 9 sweeps, via `multisweep.accumulate`) for every run **including clean**, so corrupted runs are
   paired with a clean run that differs only by the corruption. The clean sweep run is therefore a second
   baseline; report both and the difference.
5. **Smoke-test one corruption**, e.g. `SDC_CORRUPTION=range_limit SDC_SEVERITY=3 ...` - mAP must drop.
6. **Full sweep** (preview with `--dry-run`):
   ```
   python /workspace/sdc/lidar/scripts/run_corruption_sweep.py --skip-camera-radar --out-dir /results/lidar_sweep
   ```
   (run it detached, e.g. `docker run -d --name sdc_sweep ...`; a finished run is skipped on restart).
   41 evaluations of the 6,019-sample val set (1 clean + 15 point_drop + 15 missing_sweeps + 5 + 5). The
   bootstrap needs **full-val** runs (it refuses runs on a different sample set).
7. **Bootstrap + CSV** (CPU only, same container, needs the nuScenes DB):
   ```
   python /workspace/sdc/common/tools/bootstrap_sweep.py --sweep-dir /results/lidar_sweep \
     --nuscenes-root data/nuscenes --model bevfusion_lidar --lane lidar --hardware hpc_a40 \
     --results-csv /results/results.csv
   ```
   Writes, with 95% CIs from 1,000 scene resamples shared by every run: `map`, `nds`, `mate`, `mase`, `maoe`,
   `mave`, `maae`, `ap_nusc/<class>`, `rr_map`, `rr_nds` per run; `mrr_map/nds` (corruption `all`) and
   `failure_threshold_map/nds` (per corruption; **6 = never below 0.8**) with `seed = -1` (pooled).
   Rerunning replaces that model/hardware's rows instead of duplicating them. Expect roughly 3-4 minutes of
   bootstrap per run (measured on synthetic data of the same size), so ~2-3 h for the sweep; match tables are
   cached per run directory.
8. **Sonata probe** in the second image (`Dockerfile.sonata`): `scripts/sonata_probe.py`.

## CenterPoint (stretch)

BEVFusion ships no CenterPoint detection config or checkpoint, so this uses mmdet3d's v0.1.0 model zoo checkpoint
`centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus` (published val **mAP 57.34 / NDS 65.23**, 10
sweeps, 0.075 m voxels, circle NMS) expressed in BEVFusion's config dialect. The two code bases have the same 504
parameters; only axis order differs, so `convert_centerpoint_ckpt.py` renames the prefixes, swaps the x/z axes of
the 3-D sparse kernels (coords are z,y,x vs x,y,z) and transposes the 2-D kernels (BEVFusion's BEV map is
rows=x). It is a convention swap, not retraining.

```
cd /workspace/bevfusion && bash /workspace/sdc/lidar/scripts/install_centerpoint.sh
python /workspace/sdc/lidar/scripts/run_corruption_sweep.py --skip-camera-radar --out-dir /results/centerpoint_sweep \
  --config configs/nuscenes/det/centerhead/secfpn/lidar/voxelnet_0p075.yaml \
  --checkpoint pretrained/centerpoint_0075voxel_bevfusion.pth
```

Honest status: on mini the converted model behaves like a real CenterPoint (per-class APs of 0.8-0.99 for car /
bus / pedestrian, mAP/NDS close to TransFusion-L's), which would not happen with a wrong axis convention, but
**the full-val number has to be compared with 57.34 / 65.23 before any CenterPoint row is reported.** Expect a
small gap: the published figure is from the original mmdet3d code and loader (not BEVFusion's deterministic
nearest-9-sweep loader), and the model is not DCN.

## Known risks / decisions to confirm

- **GPU architecture.** CUDA 11.3 (BEVFusion's pinned stack) cannot target Ada GPUs (RTX 40-series, sm_89 -
  this laptop's RTX 4050). The Dockerfile builds `8.6+PTX`. **Verified on the laptop**: the smoke test passes
  8/8, and the custom ops (voxelization, spconv, `bev_pool`, mmcv NMS) run on the sm_89 GPU using the native
  sm_86 binaries (an sm_86 cubin is binary-compatible with sm_89), not a JIT: the driver's JIT cache stayed
  empty during normal runs. The PTX fallback itself is **not** reliable - with `CUDA_FORCE_PTX_JIT=1` a kernel
  fails with "no kernel image is available" - so do not expect this image to work on an architecture newer than
  8.x. Smoke-test wall time is ~2 min per container start (the very first run after the build took ~12 min; I
  did not isolate why). The A40 (sm_86) remains the preferred place to run the sweep; the laptop is for the
  ROS 2 / hardware-comparison rows.
- **Evaluation cost (measured, not guessed).** The progress bar's "task/s" is misleading in offline mode; wall
  clock on the laptop RTX 4050 is ~0.3-0.45 s per sample (the deterministic voxelizer alone is ~200 ms), i.e.
  roughly 30-45 min per 6,019-sample run before data-loading overhead and ~20-30 h for 41 runs. The A40 was not
  reachable from the machine used so far; if you have it, run the sweep there (`--hardware hpc_a40`).
  Otherwise budget the time or evaluate a fixed scene subset (same subset for all models, state it).
- **How the bootstrap works.** The devkit's matching is sample-local, so each prediction's TP/FP status and
  TP errors are computed once per run, and each resample just reweights predictions by how many times their
  scene was drawn (`nusc_eval.py`). At unit weights it reproduces the devkit exactly (tested, 1e-9). A scene
  drawn k times counts as one weight-k step in the PR curve rather than k steps; measured difference from
  physically duplicating scenes is about 1e-4 mAP, far below the CI widths. Still do the real-data check:
  `bootstrap_sweep.py`'s clean `map`/`nds` must equal `metrics_summary.json` from the same run.
- **Aggregation conventions** (`robustness.py`, state them in the report): RR is averaged over seeds, then
  severities, then corruptions with equal weight per corruption. Failure threshold uses the seed-mean RR.
  Metric names follow `rr_<metric>`, `mrr_<metric>`, `failure_threshold_<metric>`; **Ianna and Christian should
  use the same names** for their Tier 1 metrics.
- **Seed-to-seed range** for random corruptions is not stored as a separate row; per-seed rows carry the
  `seed` column and the plotting code should report mean and min/max across them.
- **`missing_sweeps` in ROS 2** acts on the detector's sweep buffer, not the corruption node (see node
  docstring); it uses the same table and seeding as offline.
- **Sonata is out of domain** (indoor pre-training); the probe measures stability, not accuracy.

## Cross-lane items to settle with Christian and Ianna

- `common/sdc_common/results.py` is my implementation of the shared CSV schema - confirm everyone uses it
  (or move it to wherever you prefer).
- Shared detection message (Christian): the LiDAR ROS 2 detector node should publish it; until it exists, use
  `vision_msgs/Detection3DArray`.
- Hardware labels in the CSV (`laptop`, `hpc_a40`, ...) must be spelled identically across lanes.
- Class set for "common classes" in Tier 1 (Ianna's projection scorer).
