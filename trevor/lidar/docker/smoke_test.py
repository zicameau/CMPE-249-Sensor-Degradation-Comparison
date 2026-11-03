"""Smoke test for the sdc-lidar image. Run inside the container (needs the GPU):

    docker run --rm --gpus all -v <repo>:/workspace/sdc sdc-lidar python /workspace/sdc/lidar/docker/smoke_test.py

Each check prints PASS/FAIL; exit code = number of failures. It does not need nuScenes.
Checks that custom CUDA ops actually *execute* on this GPU (importing them is not enough: a missing
architecture only fails when a kernel launches).
"""
import sys
import traceback

results = []


def check(name):
    def deco(fn):
        try:
            detail = fn()
            print(f"PASS  {name}" + (f"  ({detail})" if detail else ""), flush=True)
            results.append(True)
        except Exception:
            print(f"FAIL  {name}\n{traceback.format_exc()}", flush=True)
            results.append(False)
        return fn
    return deco


@check("torch sees the GPU")
def _():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False (is --gpus all set?)"
    return f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}"


@check("plain torch CUDA kernel + conv run")
def _():
    import torch
    x = torch.randn(2, 8, 32, 32, device="cuda")
    y = torch.nn.Conv2d(8, 8, 3, padding=1).cuda()(x)
    torch.cuda.synchronize()
    assert torch.isfinite(y).all()


@check("mmcv-full CUDA op (nms) runs")
def _():
    import torch
    from mmcv.ops import nms
    boxes = torch.rand(100, 4, device="cuda") * 100
    boxes[:, 2:] += boxes[:, :2]
    kept, _ = nms(boxes, torch.rand(100, device="cuda"), 0.5)
    torch.cuda.synchronize()
    return f"{len(kept)} boxes kept"


@check("mmdet3d voxelization (CUDA) runs")
def _():
    import torch
    from mmdet3d.ops import Voxelization
    v = Voxelization(voxel_size=[0.075, 0.075, 0.2], point_cloud_range=[-54, -54, -5, 54, 54, 3],
                     max_num_points=10, max_voxels=[120000, 160000])
    pts = torch.rand(50000, 5, device="cuda") * torch.tensor([100, 100, 6, 1, 1], device="cuda") \
        - torch.tensor([50, 50, 4, 0, 0], device="cuda")
    out = v(pts)
    torch.cuda.synchronize()
    return f"{out[0].shape[0]} voxels"


@check("mmdet3d sparse 3D conv (spconv ext) runs")
def _():
    import torch
    from mmdet3d.ops import spconv
    idx = torch.unique(torch.randint(0, 40, (4000, 3), device="cuda"), dim=0).int()
    idx = torch.cat([torch.zeros(len(idx), 1, device="cuda", dtype=torch.int32), idx], 1)
    x = spconv.SparseConvTensor(torch.randn(len(idx), 5, device="cuda"), idx, [40, 40, 40], 1)
    y = spconv.SubMConv3d(5, 16, 3, padding=1, indice_key="t").cuda()(x)
    torch.cuda.synchronize()
    return f"{y.features.shape[0]} active sites"


@check("bev_pool ext runs")
def _():
    import torch
    from mmdet3d.ops import bev_pool
    n = 2000
    feats = torch.randn(n, 8, device="cuda")
    coords = torch.randint(0, 20, (n, 4), device="cuda")
    coords[:, 3] = 0
    out = bev_pool(feats, coords.int(), 1, 1, 20, 20)
    torch.cuda.synchronize()
    return f"out {tuple(out.shape)}"


@check("SDC corruption plugin registers over LoadPointsFromMultiSweeps")
def _():
    import sdc_lidar.bevfusion_plugin  # noqa: F401
    from mmdet.datasets.builder import PIPELINES
    cls = PIPELINES.get("LoadPointsFromMultiSweeps")
    assert cls.__name__ == "CorruptedLoadPointsFromMultiSweeps", cls
    obj = cls(sweeps_num=9, load_dim=5, use_dim=5)  # clean: no env set
    assert obj.corruption is None


@check("BEVFusion LiDAR-only config loads and the model builds")
def _():
    import torch
    from mmcv import Config
    from mmdet3d.models import build_model
    from mmdet3d.utils import recursive_eval
    from torchpack.utils.config import configs
    path = "configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml"
    configs.load(path, recursive=True)
    cfg = Config(recursive_eval(configs), filename=path)  # mmcv opens `filename`, so it must be a real file
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    n = sum(p.numel() for p in model.parameters())
    return f"{n / 1e6:.1f}M parameters"


print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(len(results) - sum(results))
