#!/usr/bin/env python
"""3-D detection recall of a sweep on *the camera lane's ground truth*, in the camera lane's near/medium/far bins.

The camera lane (Camera/results/camera_shared_mini_val/ground_truth.csv) lists the CAM_FRONT-visible GT instances of
the 81 mini-val samples with their camera-frame distance and nuScenes annotation token. This tool takes exactly those
instances and asks, for every run of a LiDAR sweep: did a 3-D detection land within ``--dist-th`` metres of the object
(centre distance in the global frame, devkit-style greedy matching per sample and class, predictions with score >=
``--score-th``)? Matching is done against *all* same-class GT of the sample (so a prediction cannot be credited twice and
objects outside the camera view still compete for predictions); only the camera-visible instances are counted.

The two recalls are not the same metric (2-D IoU>=0.5 on image boxes vs 3-D centre distance < 2 m) but they share the
frames, the objects and the distance bins, which makes "which sensor loses which objects, and where" comparable.

    python common/tools/lane_recall.py --sweep-dir /results/lidar_sweep_mini --nuscenes-root data/nuscenes \\
        --version v1.0-mini --camera-gt Camera/results/camera_shared_mini_val/ground_truth.csv --out lane_recall.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distance_recall import matched_gt  # noqa: E402

BINS = {"near_0_20m": (0, 20), "medium_20_40m": (20, 40), "far_40m_plus": (40, float("inf"))}
# finer split of the far bin: the LiDAR detector only covers +-54 m, the camera does not
EXTRA_BINS = {"far_40_54m": (40, 54), "far_54m_plus": (54, float("inf"))}
COMMON = ("car", "truck", "bus", "pedestrian", "bicycle", "motorcycle")
RUN_RE = re.compile(r"^(?P<name>[a-z_]+)_s(?P<sev>[1-5])_seed(?P<seed>\d+)$")


def bins_of(d):
    return [n for n, (lo, hi) in {**BINS, **EXTRA_BINS}.items() if lo <= d < hi]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--nuscenes-root", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--camera-gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dist-th", type=float, default=2.0)
    ap.add_argument("--score-th", type=float, default=0.25, help="0.25 = the camera lane's confidence threshold")
    a = ap.parse_args()

    from nuscenes import NuScenes
    from nuscenes.eval.detection.utils import category_to_detection_name

    nusc = NuScenes(version=a.version, dataroot=a.nuscenes_root, verbose=False)

    cam = collections.defaultdict(dict)       # sample -> ann_token -> camera distance
    with open(a.camera_gt, newline="") as f:
        for r in csv.DictReader(f):
            cam[r["sample_token"]][r["annotation_token"]] = float(r["distance_m"])

    # per sample and class: all GT annotations (token, global xy); the camera lane's instances are a subset of them
    gt = {}
    skipped = 0
    for tok in cam:
        per = collections.defaultdict(list)
        for at in nusc.get("sample", tok)["anns"]:
            ann = nusc.get("sample_annotation", at)
            name = category_to_detection_name(ann["category_name"])
            if name in COMMON:
                per[name].append((at, ann["translation"][:2]))
        gt[tok] = per
        known = {t for v in per.values() for t, _ in v}
        skipped += sum(1 for at in cam[tok] if at not in known)
    n_cam = sum(len(v) for v in cam.values())
    print(f"camera GT: {n_cam} instances in {len(cam)} samples ({skipped} have no devkit detection class and are skipped)")

    rows = []
    runs = sorted(d for d in Path(a.sweep_dir).iterdir() if (d / "results_nusc.json").exists())
    for d in runs:
        m = RUN_RE.match(d.name)
        corruption, sev, seed = (m["name"], int(m["sev"]), int(m["seed"])) if m else ("clean", 0, 0)
        res = json.load(open(d / "results_nusc.json"))["results"]
        hit_by = collections.defaultdict(lambda: [0, 0])
        for tok, per in gt.items():
            preds = collections.defaultdict(list)
            for b in res.get(tok, []):
                if b["detection_name"] in COMMON and b["detection_score"] >= a.score_th:
                    preds[b["detection_name"]].append((b["translation"][:2], b["detection_score"]))
            for cls, items in per.items():
                gxy = np.array([xy for _, xy in items], float)
                pxy = np.array([p[0] for p in preds[cls]], float).reshape(-1, 2)
                ps = np.array([p[1] for p in preds[cls]], float)
                hit = matched_gt(gxy, pxy, ps, a.dist_th)
                for (at, _), h in zip(items, hit):
                    if at in cam[tok]:
                        for key in bins_of(cam[tok][at]) + ["all"]:
                            hit_by[key][0] += int(h)
                            hit_by[key][1] += 1
        for b, (nh, ng) in hit_by.items():
            rows.append((corruption, sev, seed, b, ng, nh / ng if ng else float("nan")))
        print(f"{d.name}: " + " ".join(f"{b}={nh / ng:.3f}" for b, (nh, ng) in sorted(hit_by.items())))

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corruption", "severity", "seed", "bin", "n_gt", "recall"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
