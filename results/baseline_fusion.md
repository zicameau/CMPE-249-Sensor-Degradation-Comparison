# BEVFusion Camera+LiDAR Baseline — nuScenes-mini

Clean-input baseline for the fusion model. All degradation experiments are compared against these numbers.

- **Date:** 2026-11-04
- **Model:** BEVFusion (mit-han-lab), camera+LiDAR detection
- **Checkpoint:** `pretrained/bevfusion-det.pth` (released weights, unmodified)
- **Config:** `configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml`
- **Data:** nuScenes **v1.0-mini**, val split — 2 scenes, **81 keyframes**
- **Hardware:** NVIDIA RTX 4070 SUPER (12 GB), Windows 11 → WSL2 Ubuntu 22.04 → Docker (CUDA 11.3, PyTorch 1.10.1)
- **Raw log:** [baseline_fusion.txt](baseline_fusion.txt)

## Command

Run inside the `bevf` container from `/workspace/bevfusion`:

```bash
torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth --eval bbox 2>&1 | tee results/baseline_fusion.txt
```

## Overall metrics

| Metric | Value |
|---|---|
| **mAP** | **0.5822** |
| **NDS** | **0.5838** |
| mAP over the 7 classes present in mini-val | 0.832 |
| mATE (translation error, m) | 0.4042 |
| mASE (scale error) | 0.4465 |
| mAOE (orientation error, rad) | 0.4719 |
| mAVE (velocity error, m/s) | 0.4241 |
| mAAE (attribute error) | 0.3264 |

## Per-class results

| Class | GT objects in mini-val | AP | ATE | ASE | AOE | AVE | AAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| car | 2056 | 0.928 | 0.173 | 0.157 | 0.100 | 0.117 | 0.074 |
| truck | 122 | 0.815 | 0.150 | 0.121 | 0.028 | 0.071 | 0.018 |
| bus | 41 | 0.994 | 0.164 | 0.077 | 0.026 | 0.432 | 0.394 |
| trailer | 0 | 0.000 | — | — | — | — | — |
| construction_vehicle | 0 | 0.000 | — | — | — | — | — |
| pedestrian | 1260 | 0.927 | 0.118 | 0.257 | 0.378 | 0.217 | 0.125 |
| motorcycle | 233 | 0.734 | 0.196 | 0.273 | 0.352 | 0.053 | 0.000 |
| bicycle | 44 | 0.536 | 0.157 | 0.238 | 0.363 | 0.502 | 0.000 |
| traffic_cone | 37 | 0.888 | 0.083 | 0.343 | — | — | — |
| barrier | 0 | 0.000 | — | — | — | — | — |

GT counts are valid annotations (at least one LiDAR or radar point) across the 81 val keyframes.

## Interpretation notes

- **The zero-AP classes are absent, not missed.** Trailer, construction vehicle, and barrier have no ground truth in mini-val, so they score AP 0 and pull the 10-class mAP down. The mean over the 7 classes that do appear is 0.832.
- **Not comparable to the paper.** The published 68.5 mAP / 71.4 NDS is on the full nuScenes val set (150 scenes, 6019 frames). Mini-val is 2 scenes, so the numbers are noisy and class coverage is incomplete. Use this baseline only as a reference point for the degradation runs, not to verify the paper.
- **Sparse classes are noisy.** Bicycle (44), bus (41), and traffic cone (37) have few objects, so their AP can swing a lot between conditions.
- **Runtime:** inference over the 81 frames took about a minute; NuScenes evaluation took 1.3 s.

## Config details relevant to degradation tests

| Setting | Value | Relevance |
|---|---|---|
| `sweeps_num` (test) | 9 | Each frame merges the current LiDAR scan with 9 previous sweeps. Lowering it simulates missing scans. |
| `reduce_beams` | 32 | nuScenes LiDAR is already 32-beam, so this is currently a no-op. Lowering it (16, 8) simulates a lower-resolution or failing LiDAR. |
| `point_cloud_range` | [-54, -54, -5, 54, 54, 3] m | Shrinking it simulates limited sensing range. |
| Camera input | 6 cameras, resized to 256×704 | Darken/blur/drop individual cameras for visibility stress. |

## Repo fixes required to reach this baseline

The upstream `mit-han-lab/bevfusion` repo no longer runs as-is. Fixes 1–5 are commit [`d553805`](https://github.com/christiankso/bevfusion/commit/d553805) on our fork's `cmpe249-fixes` branch, included in this repo as the `bevfusion/` submodule (setup: [../docs/bevfusion_setup.md](../docs/bevfusion_setup.md)):

1. **`docker/Dockerfile`** — Miniconda pinned to `py38_23.11.0-2`; PyTorch 1.10.1+cu113 via pip; prebuilt `mmcv-full` 1.4.0 for cu113; OpenMPI + linker workaround for `mpi4py` 3.1.5; `numpy==1.23.5` (numba 0.48 breaks on newer NumPy). See [../bevfusion/docker/Dockerfile](../bevfusion/docker/Dockerfile).
2. **`mmdet3d/ops/__init__.py`** (lines 12, 93) and **`mmdet3d/models/backbones/radar_encoder.py`** (line 15) — commented out the radar-only `feature_decorator` op, which `setup.py` never compiles.
3. **`mmdet3d/models/backbones/__init__.py`** (line 9) — commented out `from .radar_encoder import *` (requires `flash_attn`; only used by camera+radar configs).
4. **`tools/data_converter/nuscenes_converter.py`** (lines 87–100) — info files now save to `data/nuscenes/` as `nuscenes_infos_{train,val}.pkl` instead of `./nuscenes/…_radar.pkl`.
5. **`mmdet3d/models/vtransforms/depth_lss.py`** (lines 37–38) — passes `height_expand=False, add_depth_features=False` so the depth input is 1 channel, matching the released checkpoint. A later radar commit changed the base-class defaults to 6 channels.
6. **Weights** — `tools/download_pretrained.sh` returned Dropbox HTML pages for `bevfusion-det.pth`, `lidar-only-seg.pth`, and `swint-nuimages-pretrained.pth`; re-downloaded with `&dl=1` appended to the URLs.
