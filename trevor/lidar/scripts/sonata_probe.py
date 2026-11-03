#!/usr/bin/env python
"""Sonata feature-stability probe on nuScenes LiDAR (Tier 3 "feature stability", LiDAR backbone).

For each sampled keyframe: run Sonata on the clean cloud and on every corrupted cloud, then compare
features inside the GT boxes (see ``sdc_lidar.feature_stability``). Appends ``feat_cos`` (mean cosine)
and ``feat_cov`` (fraction of boxes still containing points) rows with scene-bootstrap CIs.

Runs in the *Sonata* environment (CUDA 12 / PyTorch 2.x - see lidar/docker/Dockerfile.sonata), NOT in
the BEVFusion image. Needs: sonata, torch, nuscenes-devkit, scipy.

Caveats to state in the report
  * Sonata is pre-trained on indoor/object data; nuScenes is out of domain. Features are used as a
    frozen backbone whose *stability under corruption* is measured, not its detection quality.
  * Single keyframe sweep (no multi-sweep accumulation), so ``missing_sweeps`` is not applicable.
  * nuScenes has no colour/normals: intensity is replicated as colour, normals are zeros.
  * The voxel ``--grid-size`` (default 0.1 m) is much larger than Sonata's indoor 0.02 m default.

Example:
    python lidar/scripts/sonata_probe.py --nuscenes-root /dataset/nuscenes --version v1.0-trainval \
        --hardware hpc_a40 --results-csv /results/results.csv --max-scenes 150 --frames-per-scene 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "lidar")]

from sdc_common import stats  # noqa: E402
from sdc_common.results import ResultRow, append_rows  # noqa: E402
from sdc_lidar import corruptions as C  # noqa: E402
from sdc_lidar.feature_stability import feature_stability  # noqa: E402

MODEL_NAME = "sonata"
LANE = "lidar"


def load_sonata(device="cuda", enable_flash=None):
    import sonata
    try:
        import flash_attn  # noqa: F401
        have_flash = True
    except ImportError:
        have_flash = False
    if enable_flash is None:
        enable_flash = have_flash
    if enable_flash:
        model = sonata.load("sonata", repo_id="facebook/sonata")
    else:
        model = sonata.load("sonata", repo_id="facebook/sonata", custom_config=dict(
            enc_patch_size=[1024 for _ in range(5)], enable_flash=False))
    return model.to(device).eval()


def sonata_features(model, transform, xyz: np.ndarray, intensity: np.ndarray, device="cuda"):
    """Per-point (original resolution) Sonata features for one cloud -> (N, D) float32 numpy."""
    import torch
    inten = np.clip(intensity / 255.0, 0, 1).astype(np.float32)
    point = {
        "coord": xyz.astype(np.float32),
        "color": np.repeat(inten[:, None], 3, axis=1),
        "normal": np.zeros_like(xyz, dtype=np.float32),
    }
    point = transform(point)
    with torch.inference_mode():
        for k in point.keys():
            if isinstance(point[k], torch.Tensor):
                point[k] = point[k].to(device, non_blocking=True)
        point = model(point)
        for _ in range(2):  # un-pool, exactly as in Sonata's demo/0_pca.py
            parent, inverse = point.pop("pooling_parent"), point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            parent, inverse = point.pop("pooling_parent"), point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
        return point.feat[point.inverse].float().cpu().numpy()


def gt_boxes_lidar(nusc, sample):
    """GT boxes of a keyframe in the LIDAR_TOP frame as (M, 7): cx cy cz l w h yaw."""
    sd = sample["data"]["LIDAR_TOP"]
    path, boxes, _ = nusc.get_sample_data(sd)
    out = [[*b.center, b.wlh[1], b.wlh[0], b.wlh[2], b.orientation.yaw_pitch_roll[0]] for b in boxes]
    return path, np.array(out, dtype=np.float64).reshape(-1, 7)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nuscenes-root", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--results-csv", required=True)
    ap.add_argument("--max-scenes", type=int, default=150)
    ap.add_argument("--frames-per-scene", type=int, default=4)
    ap.add_argument("--grid-size", type=float, default=0.1)
    ap.add_argument("--corruptions", nargs="+", default=["point_drop", "beam_reduction", "range_limit"],
                    choices=[n for n, c in C.REGISTRY.items() if not c.multi_sweep])
    ap.add_argument("--severities", nargs="+", type=int, default=list(C.SEVERITIES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-boot", type=int, default=stats.N_BOOT)
    args = ap.parse_args()

    import sonata
    from nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_root, verbose=False)
    val_names = set(create_splits_scenes()["val" if args.version == "v1.0-trainval" else "mini_val"])
    scenes = [s for s in nusc.scene if s["name"] in val_names][: args.max_scenes]

    # Sonata's default inference pipeline, with an outdoor-scale grid size (default is 0.02 m).
    transform = sonata.transform.Compose([
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=args.grid_size, hash_type="fnv", mode="train",
             return_grid_coord=True, return_inverse=True),
        dict(type="NormalizeColor"), dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "color", "inverse"),
             feat_keys=("coord", "color", "normal")),
    ])
    model = load_sonata(args.device)

    # runs: (corruption, severity, seed)
    runs = [(n, s, seed) for n in args.corruptions for s in args.severities for seed in C.seeds_for(n)]
    per_scene = {r: {"cos": [], "cov": []} for r in runs}   # list over scenes (nan if undefined)

    for si, scene in enumerate(scenes):
        # evenly spaced keyframes through the scene
        tokens, tok = [], scene["first_sample_token"]
        while tok:
            tokens.append(tok)
            tok = nusc.get("sample", tok)["next"]
        picks = np.linspace(0, len(tokens) - 1, min(args.frames_per_scene, len(tokens))).round().astype(int)
        acc = {r: ([], []) for r in runs}
        for pi in sorted(set(picks)):
            sample = nusc.get("sample", tokens[pi])
            path, boxes = gt_boxes_lidar(nusc, sample)
            if len(boxes) == 0:
                continue
            raw = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
            clean_f = sonata_features(model, transform, raw[:, :3], raw[:, 3], args.device)
            for (name, sev, seed) in runs:
                cor = C.apply(name, raw, sev, seed=seed, frame_key=sample["token"])
                if len(cor) < 8:
                    acc[(name, sev, seed)][1].append(0.0)
                    continue
                cor_f = sonata_features(model, transform, cor[:, :3], cor[:, 3], args.device)
                r = feature_stability(raw[:, :3], clean_f, cor[:, :3], cor_f, boxes)
                acc[(name, sev, seed)][1].append(r.coverage)
                if r.coverage > 0:
                    acc[(name, sev, seed)][0].append(r.mean_cosine)
        for r in runs:
            cs, cv = acc[r]
            per_scene[r]["cos"].append(float(np.mean(cs)) if cs else float("nan"))
            per_scene[r]["cov"].append(float(np.mean(cv)) if cv else float("nan"))
        print(f"scene {si + 1}/{len(scenes)} done", flush=True)

    n = len(scenes)
    W = stats.bootstrap_weights(n, args.n_boot)

    def weighted_mean(vals):
        v = np.asarray(vals, dtype=float)
        ok = np.isfinite(v)
        return lambda w: float((w[ok] * v[ok]).sum() / w[ok].sum()) if w[ok].sum() > 0 else float("nan")

    rows = []
    for (name, sev, seed), d in per_scene.items():
        for metric, key in (("feat_cos", "cos"), ("feat_cov", "cov")):
            est = stats.bootstrap_ci(weighted_mean(d[key]), n, weights=W)
            if est.value != est.value or est.ci_low != est.ci_low:
                continue  # undefined (e.g. every box uncovered) - leave it out rather than fake a number
            rows.append(ResultRow(MODEL_NAME, LANE, name, sev, seed, args.hardware, metric, "all",
                                  est.value, est.ci_low, est.ci_high))
    print(f"appended {append_rows(args.results_csv, rows)} rows to {args.results_csv}")


if __name__ == "__main__":
    main()
