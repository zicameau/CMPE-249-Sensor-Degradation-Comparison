# LiDAR lane environment: BEVFusion (LiDAR-only / TransFusion-L) + this repo's corruption plugin.
# Same stack as mit-han-lab/bevfusion's docker/Dockerfile (CUDA 11.3.1, Python 3.8, PyTorch 1.10.1), but:
#   * Ubuntu's own Python 3.8 + pip wheels instead of conda (current Miniconda needs an Anaconda ToS
#     acceptance step, and pip wheels are smaller and pinnable);
#   * BEVFusion pinned to a commit, for reproducibility;
#   * BEVFusion's setup.py patched to also emit PTX (see below);
#   * mmcv-full built from source so its ops get the same architecture list.
#
# GPU architectures: CUDA 11.3 cannot target Ada (RTX 40xx, sm_89). We build native code for Ampere
# (8.6: A40/3090) plus PTX, which the driver JIT-compiles for newer GPUs on first use (first run is slower).
#
#   docker build -f lidar/docker/Dockerfile -t sdc-lidar .
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ARG BEVFUSION_REF=326653dc06e0938edf1aae7d01efcd158ba83de5
ARG TORCH_CUDA_ARCH_LIST="8.6+PTX"
ARG MAX_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    FORCE_CUDA=1 \
    MMCV_WITH_OPS=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# NVIDIA rotated its apt signing keys in 2022; the 11.3.1 image still ships the old ones.
RUN apt-key del 7fa2af80 >/dev/null 2>&1 || true \
    && rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia-ml*.list \
    && apt-get update && apt-get install -yq --no-install-recommends \
        python3.8 python3.8-dev python3-pip python3-distutils python-is-python3 \
        wget git build-essential ninja-build ca-certificates \
        libgl1-mesa-glx libglib2.0-0 libgtk2.0-dev \
        openmpi-bin openmpi-common libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.8 /usr/local/bin/python \
    && python -m pip install "pip<25" "setuptools<66" wheel

# Keeps numpy where numba 0.48 needs it, no matter what later installs ask for.
RUN echo "numpy==1.19.5" > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 \
        -f https://download.pytorch.org/whl/cu113/torch_stable.html

# mmcv-full from source (slow, ~15-30 min) so its CUDA ops include PTX like everything else.
RUN pip install Pillow==8.4.0 tqdm opencv-python-headless \
    && pip install mmcv-full==1.4.0 mmdet==2.20.0

RUN pip install torchpack mpi4py==3.0.3 numba==0.48.0 nuscenes-devkit pytest scipy

# BEVFusion + its custom CUDA ops. Its setup.py hardcodes sm_70..sm_86 and no PTX, which would leave
# the ops unusable on newer GPUs; append a compute_86 PTX target after the sm_86 line.
RUN git clone https://github.com/mit-han-lab/bevfusion /workspace/bevfusion \
    && cd /workspace/bevfusion && git checkout ${BEVFUSION_REF} \
    && sed -i 's|^\(\s*\)"-gencode=arch=compute_86,code=sm_86",|&\n\1"-gencode=arch=compute_86,code=compute_86",|' setup.py \
    && grep -c "code=compute_86" setup.py \
    && python setup.py develop

# Upstream bug at this commit: mmdet3d/ops/__init__.py imports feature_decorator unconditionally, but
# setup.py never builds the `feature_decorator_ext` extension, so `import mmdet3d.ops` (and therefore the
# whole package) dies with ImportError. The op is only used by the camera+radar encoder and the PointPillars
# config, not the LiDAR-only voxelnet model we use. Make just that import tolerant (the op still raises if
# someone actually calls it). `develop` is an in-place install, so patching the source is enough.
RUN cd /workspace/bevfusion \
    && sed -i 's|^from \. import feature_decorator_ext$|try:\n    from . import feature_decorator_ext  # not built upstream; only PointPillars / radar need it\nexcept ImportError:\n    feature_decorator_ext = None|' mmdet3d/ops/feature_decorator/feature_decorator.py \
    && grep -c "feature_decorator_ext = None" mmdet3d/ops/feature_decorator/feature_decorator.py

# Same story for flash_attn: mmdet3d/models/backbones/radar_encoder.py imports FlashMHA at module level, and
# `mmdet3d.models` imports that module, so building ANY model fails without flash-attn (which needs an old
# 0.2.x build for this API). Only the camera+radar encoder uses it; make the import tolerant.
RUN cd /workspace/bevfusion \
    && sed -i 's|^from flash_attn\.flash_attention import FlashMHA$|try:\n    from flash_attn.flash_attention import FlashMHA\nexcept ImportError:  # only the camera+radar encoder needs it\n    FlashMHA = None|' mmdet3d/models/backbones/radar_encoder.py \
    && grep -c "FlashMHA = None" mmdet3d/models/backbones/radar_encoder.py
# Repo code (mounted at /workspace/sdc by docker-compose); checkpoints: tools/download_pretrained.sh
ENV PYTHONPATH=/workspace/sdc/common:/workspace/sdc/lidar
WORKDIR /workspace/bevfusion
