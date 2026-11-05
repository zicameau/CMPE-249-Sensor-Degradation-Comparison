# BEVFusion Setup and Replication Guide

How to reproduce our BEVFusion (camera + LiDAR) baseline and sensor-degradation results on nuScenes, starting from a Windows PC with an NVIDIA GPU.

**What you end up with:** a Docker container that runs the released BEVFusion fusion checkpoint on nuScenes-mini, the clean baseline (mAP 0.582 / NDS 0.584), and the 24-condition degradation sweep.

**Time:** about 1–2 hours, mostly downloads and the Docker build.

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Install WSL2 and Docker](#2-install-wsl2-and-docker)
3. [Get the code](#3-get-the-code)
4. [Build the Docker image](#4-build-the-docker-image)
5. [Download nuScenes](#5-download-nuscenes)
6. [Create the container and compile BEVFusion](#6-create-the-container-and-compile-bevfusion)
7. [Download pretrained weights](#7-download-pretrained-weights)
8. [Prepare the dataset](#8-prepare-the-dataset)
9. [Run the clean baseline](#9-run-the-clean-baseline)
10. [Run the degradation sweep](#10-run-the-degradation-sweep)
11. [Daily workflow](#11-daily-workflow)
12. [Repository layout](#12-repository-layout)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Windows 10/11 | Linux works too; skip step 2 and install Docker + NVIDIA Container Toolkit directly. |
| NVIDIA GPU, ≥ 8 GB VRAM | Tested on RTX 4070 SUPER (12 GB). Inference uses batch size 1. |
| Recent NVIDIA Windows driver | Do **not** install a separate driver inside WSL. |
| ~50 GB free disk | Docker image (~15–20 GB), nuScenes-mini extracted (~10 GB), weights (~0.9 GB). |
| nuScenes account | Free for non-commercial use: register at [nuscenes.org](https://www.nuscenes.org). |

**Why Linux/Docker?** BEVFusion compiles custom CUDA operators and depends on old pinned libraries (PyTorch 1.10, CUDA 11.3, mmcv-full 1.4.0). These do not build on native Windows, and Docker keeps the old versions isolated.

## 2. Install WSL2 and Docker

1. **WSL2 + Ubuntu 22.04.** In an administrator PowerShell:

   ```powershell
   wsl --install -d Ubuntu-22.04
   ```

   Reboot if asked. Ubuntu opens and asks for a Linux username and password. Check it is WSL **2**:

   ```powershell
   wsl -l -v
   ```

2. **Docker Desktop** from [docker.com](https://www.docker.com/products/docker-desktop/). Keep "Use WSL 2 instead of Hyper-V" checked. After installing: **Settings → Resources → WSL integration →** enable **Ubuntu-22.04** → **Apply & restart**.

3. **Check that Docker can use the GPU.** In the Ubuntu terminal (type `wsl` in PowerShell, or open Ubuntu from the Start menu):

   ```bash
   docker run --rm --gpus all nvidia/cuda:11.3.1-base-ubuntu20.04 nvidia-smi
   ```

   You should see a table listing your GPU.

> **All commands from here on run in the Ubuntu terminal** unless marked "inside the container". Keep code and data in the Linux home folder (`~`), not under `/mnt/c/...`: Windows files are very slow when accessed from Linux.

## 3. Get the code

We use our fork of BEVFusion, which contains the fixes needed to build and run it plus the degradation code. The team repo includes it as the `bevfusion/` submodule, but **run BEVFusion from its own clone in the Linux home folder:**

```bash
cd ~
git clone -b cmpe249-fixes https://github.com/christiankso/bevfusion.git
```

The fork is [mit-han-lab/bevfusion](https://github.com/mit-han-lab/bevfusion) at commit `326653d` plus our commits on branch `cmpe249-fixes`. [../results/baseline_fusion.md](../results/baseline_fusion.md#repo-fixes-required-to-reach-this-baseline) explains each fix.

## 4. Build the Docker image

```bash
cd ~/bevfusion/docker
docker build . -t bevfusion
```

Takes about 20–40 minutes. It ends with `naming to docker.io/library/bevfusion`. Two `LegacyKeyValueFormat` warnings are harmless.

## 5. Download nuScenes

Log in at [nuscenes.org](https://www.nuscenes.org) → **Download**, accept the terms, and download these three files (US or Asia mirror):

| Page section | File | Size |
|---|---|---|
| Full dataset (v1.0) → **Mini** | `v1.0-mini.tgz` | ~4 GB |
| **CAN bus expansion** | `can_bus.zip` | ~0.75 GB |
| **Map expansion** → **v1.3** | `nuScenes-map-expansion-v1.3.zip` | ~0.4 GB |

Skip trainval/test, nuImages, nuPlan, lidarseg, panoptic, and prediction.

Extract into `~/data/nuscenes` (adjust `<you>` to your Windows username if the files are in Downloads):

```bash
sudo apt install -y unzip
mkdir -p ~/data/nuscenes && cd ~/data/nuscenes
tar -xzf /mnt/c/Users/<you>/Downloads/v1.0-mini.tgz
unzip /mnt/c/Users/<you>/Downloads/can_bus.zip
unzip /mnt/c/Users/<you>/Downloads/nuScenes-map-expansion-v1.3.zip -d maps
ls
```

Expected: `LICENSE  can_bus  maps  samples  sweeps  v1.0-mini`.

## 6. Create the container and compile BEVFusion

Create a **named** container so you can come back to it (compiled code persists inside it):

```bash
docker run --gpus all -it --shm-size 16g --name bevf \
  -v ~/bevfusion:/workspace/bevfusion \
  -v ~/data/nuscenes:/dataset/nuscenes \
  bevfusion
```

The NVIDIA license banner is normal. The prompt changes to `root@<id>:/#`; you are now **inside the container**. Then:

```bash
cd /workspace/bevfusion
mkdir -p data && ln -s /dataset/nuscenes data/nuscenes
export TORCH_CUDA_ARCH_LIST="8.6+PTX"
python setup.py develop
```

Compiling takes 5–15 minutes and ends with `Finished processing dependencies for mmdet3d`.

`TORCH_CUDA_ARCH_LIST` must match your GPU. CUDA 11.3 does not know RTX 40-series (sm_89), so we compile for sm_86 with PTX, which runs on it:

| GPU | Value |
|---|---|
| RTX 40-series, RTX 30-series, A-series | `8.6+PTX` |
| RTX 20-series, T4 | `7.5` |
| V100 | `7.0` |

The two folders mounted with `-v` are the same files as `~/bevfusion` and `~/data/nuscenes`; nothing is copied.

## 7. Download pretrained weights

Inside the container, in `/workspace/bevfusion`. The upstream download script's Dropbox links return web pages instead of files unless `&dl=1` is added, so fix the links first:

```bash
sed -i 's/?rlkey=\([a-z0-9]*\)/?rlkey=\1\&dl=1/' tools/download_pretrained.sh
./tools/download_pretrained.sh
ls -lh pretrained
```

Check the sizes. Every file should be tens to hundreds of MB; a ~200 KB file is an HTML page, so rerun the download.

| File | Approx. size |
|---|---|
| `bevfusion-det.pth` (fusion, used here) | 157 MB |
| `lidar-only-det.pth` | 32 MB |
| `camera-only-det.pth` | 170 MB |
| `swint-nuimages-pretrained.pth` | 106 MB |

## 8. Prepare the dataset

Inside the container:

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini
```

Prints `train sample: 323, val sample: 81`, then builds a ground-truth database (a few minutes). It creates `nuscenes_infos_train.pkl`, `nuscenes_infos_val.pkl`, `nuscenes_dbinfos_train.pkl`, and `nuscenes_gt_database/` in `~/data/nuscenes`.

## 9. Run the clean baseline

Inside the container:

```bash
mkdir -p results
torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth --eval bbox \
  --metrics-out results/baseline_fusion.json 2>&1 | tee results/baseline_fusion.txt
```

The first run downloads Swin-T ImageNet weights (~109 MB). Inference on the 81 val frames takes under a minute. Expected:

| Metric | Expected |
|---|---|
| mAP | 0.5822 |
| NDS | 0.5838 |

Small differences (±0.005) are possible because the multi-sweep loader samples sweeps randomly. Trailer, construction vehicle, and barrier score AP 0 because mini-val contains none of them. Full results and interpretation: [../results/baseline_fusion.md](../results/baseline_fusion.md).

## 10. Run the degradation sweep

Inside the container:

```bash
python tools/run_degradation_sweep.py
```

Runs the clean baseline plus 23 degraded conditions (~45 s each, ~20 min total) and writes to `results/degradation/fusion/`:

- `<group>__<condition>.json` — metrics and timing for each condition
- `<group>__<condition>.log` — full test output for each condition
- `summary.csv` — one row per condition: mAP, NDS, per-class AP, fps

If interrupted, rerun the same command; finished conditions are skipped. To run only some groups, add `--groups lidar_point_drop cam_drop`. The options, what each one simulates, and the LiDAR-only comparison are documented in [degradation.md](degradation.md).

## 11. Daily workflow

1. Start **Docker Desktop** and wait for "Engine running".
2. Open the Ubuntu terminal.
3. Enter the container:

   ```bash
   docker start -ai bevf
   ```

   If the container is already running in the background, use `docker exec -it bevf bash` instead.
4. `cd /workspace/bevfusion` and work. Type `exit` to leave.

**Editing code:** edit files in `~/bevfusion` from Ubuntu (e.g. `cd ~/bevfusion && code .` with VS Code's WSL extension). Changes take effect in the container immediately. Python changes need no rebuild; rerun `python setup.py develop` only after changing CUDA/C++ code under `mmdet3d/ops/`.

**Saving code changes** (BEVFusion code lives in the fork, the team repo only points to a commit):

```bash
# 1. In ~/bevfusion (Ubuntu): commit and push to the fork
#    (first time: git remote add fork https://github.com/christiankso/bevfusion.git)
git add <files> && git commit -m "..."
git push fork cmpe249-fixes

# 2. In the team repo: point the submodule at the new commit
git submodule update --remote bevfusion
git add bevfusion && git commit -m "Bump bevfusion submodule" && git push
```

Push the fork **before** the team repo, or teammates get a "commit not found" error. Don't edit code inside the team repo's `bevfusion/` folder; treat it as read-only.

## 12. Repository layout

```
CMPE-249-Sensor-Degradation-Comparison/   (team repo, branch BEVFusion)
├── bevfusion/                → submodule: christiankso/bevfusion @ cmpe249-fixes
├── docs/
│   ├── bevfusion_setup.md    ← this guide
│   └── degradation.md        ← degradation options and sweep usage
└── results/
    ├── baseline_fusion.md    ← clean baseline results and repo fixes
    └── baseline_fusion.txt   ← raw test output

~/bevfusion/                  (WSL, where code runs; same as the fork)
~/data/nuscenes/              (WSL, dataset; not in git)
```

Clone the team repo with the submodule:

```bash
git clone --recursive -b BEVFusion https://github.com/zicameau/CMPE-249-Sensor-Degradation-Comparison.git
```

## 13. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `docker: command not found` or "Cannot connect to the Docker daemon" | Docker Desktop not running or WSL integration off | Start Docker Desktop; enable Ubuntu-22.04 under Settings → Resources → WSL integration. |
| `could not select device driver "" with capabilities: [[gpu]]` | GPU not visible to Docker | Update the NVIDIA Windows driver; restart Docker Desktop. |
| `Conflict. The container name "/bevf" is already in use` | Container already exists | Use `docker start -ai bevf` instead of `docker run`. |
| Build fails at `conda install` with `CondaToSNonInteractiveError` | Using upstream Dockerfile | Use the fork's Dockerfile (`cmpe249-fixes` branch). |
| `ImportError: cannot import name 'feature_decorator_ext'` or `No module named 'flash_attn'` | Upstream radar code | Use the fork; these imports are disabled there. |
| `AttributeError: module 'numpy' has no attribute 'long'` | NumPy ≥ 1.24 with numba 0.48 | Inside the container: `pip install numpy==1.23.5`. |
| `FileNotFoundError: ./data/nuscenes/nuscenes_infos_train.pkl` | Upstream converter saves to the wrong place | Use the fork; rerun `create_data.py`. Check the `data/nuscenes` symlink exists. |
| `expected input[6, 6, 256, 704] to have 1 channels, but got 6` | Upstream radar change to the depth transform | Use the fork (`depth_lss.py` fix). |
| `_pickle.UnpicklingError` / `invalid load key '<'` when loading a checkpoint | Weight file is a Dropbox HTML page | Redo step 7 and check file sizes. |
| `CUDA error: no kernel image is available` | Wrong `TORCH_CUDA_ARCH_LIST` | Set the value for your GPU (step 6), then `python setup.py develop` again. |
| CUDA out of memory | Other GPU programs, or < 8 GB VRAM | Close other GPU apps; test runs use batch size 1. |
| `Permission denied` deleting files in `~/bevfusion` | Files created by the container are owned by root | Delete them from inside the container, or with `sudo`. |
