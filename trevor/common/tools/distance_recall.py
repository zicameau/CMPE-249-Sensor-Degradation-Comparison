#!/usr/bin/env python
"""Distance-binned recall for every run of a sweep (clean + corrupted) -> CSV.

Recall = fraction of GT boxes (binned by their ego range) that are matched by a prediction within a centre-distance
threshold (default 2.0 m, the devkit's mid threshold). Matching follows the nuScenes devkit's ``accumulate``: per class,
predictions in descending score order each claim the nearest still-unmatched GT of that class within the threshold.
Two operating points are reported: ``all`` (every prediction the detector emitted) and ``score>=T`` (default 0.3), so
a drop caused by the detector losing confidence is not mistaken for one caused by losing the object.

GT and predictions are loaded and filtered exactly as ``DetectionEval`` does (same helpers as ``sdc_common.nusc_loader``).

    python common/tools/distance_recall.py --sweep-dir /results/lidar_sweep --nuscenes-root /dataset/nuscenes \\
        --out /results/distance_recall.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

CLASSES = ("car", "truck", "construction_vehicle", "bus", "trailer",
           "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone")
DEFAULT_BINS = (0, 10, 20, 30, 40, 54.01)   # nuScenes eval range for every class is <= 50 m; 54 = detector grid edge
RUN_RE = re.compile(r"^(?P<name>[a-z_]+)_s(?P<sev>[1-5])_seed(?P<seed>\d+)$")


def _arrays(boxes_by_sample, token):
    """-> per class: (global xy (n,2), ego_dist (n,), score (n,))"""
    out = {c: ([], [], []) for c in CLASSES}
    for b in boxes_by_sample[token]:
        if b.detection_name in out:
            xs, ds, ss = out[b.detection_name]
            xs.append(b.translation[:2])
            ds.append(float(np.hypot(*b.ego_translation[:2])))
            ss.append(float(getattr(b, "detection_score", -1.0)))
    return {c: (np.asarray(v[0], float).reshape(-1, 2), np.asarray(v[1], float), np.asarray(v[2], float))
            for c, v in out.items()}


def matched_gt(gt_xy, pred_xy, pred_score, th):
    """Devkit-style greedy matching for one (sample, class). Returns bool mask over GT."""
    hit = np.zeros(len(gt_xy), bool)
    if len(gt_xy) == 0 or len(pred_xy) == 0:
        return hit
    for p in np.argsort(-pred_score, kind="stable"):
        d = np.linalg.norm(gt_xy - pred_xy[p], axis=1)
        d[hit] = np.inf
        j = int(np.argmin(d))
        if d[j] < th:
            hit[j] = True
    return hit


def run_recall(gt, pred, th, bins, score_th):
    """-> {(class, bin_idx, point): (n_matched, n_gt)}"""
    bins = np.asarray(bins, float)
    acc = {}
    for tok in gt.sample_tokens:
        g = _arrays(gt.boxes, tok)
        p = _arrays(pred.boxes, tok) if tok in pred.boxes else {c: (np.zeros((0, 2)), np.zeros(0), np.zeros(0)) for c in CLASSES}
        for c in CLASSES:
            gxy, gd, _ = g[c]
            if len(gxy) == 0:
                continue
            pxy, _, ps = p[c]
            b = np.clip(np.digitize(gd, bins) - 1, 0, len(bins) - 2)
            for point, sel in (("all", np.ones(len(ps), bool)), (f"score>={score_th:g}", ps >= score_th)):
                hit = matched_gt(gxy, pxy[sel], ps[sel], th)
                for bi in range(len(bins) - 1):
                    m = b == bi
                    if m.any():
                        k = (c, bi, point)
                        a = acc.get(k, (0, 0))
                        acc[k] = (a[0] + int(hit[m].sum()), a[1] + int(m.sum()))
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--nuscenes-root", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dist-th", type=float, default=2.0)
    ap.add_argument("--score-th", type=float, default=0.3)
    ap.add_argument("--bins", type=float, nargs="+", default=list(DEFAULT_BINS))
    args = ap.parse_args()

    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes, load_gt, load_prediction
    from nuscenes.eval.detection.data_classes import DetectionBox

    eval_set = {"v1.0-trainval": "val", "v1.0-mini": "mini_val"}[args.version]
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_root, verbose=False)
    cfg = config_factory("detection_cvpr_2019")
    gt = filter_eval_boxes(nusc, add_center_dist(nusc, load_gt(nusc, eval_set, DetectionBox, verbose=False)),
                           cfg.class_range, verbose=False)

    runs = sorted(d for d in Path(args.sweep_dir).iterdir() if (d / "results_nusc.json").exists())
    rows = []
    for d in runs:
        m = RUN_RE.match(d.name)
        corruption, sev, seed = (m["name"], int(m["sev"]), int(m["seed"])) if m else ("clean", 0, 0)
        pred, _ = load_prediction(str(d / "results_nusc.json"), cfg.max_boxes_per_sample, DetectionBox, verbose=False)
        pred = filter_eval_boxes(nusc, add_center_dist(nusc, pred), cfg.class_range, verbose=False)
        acc = run_recall(gt, pred, args.dist_th, args.bins, args.score_th)
        points = sorted({k[2] for k in acc})
        for point in points:
            for bi in range(len(args.bins) - 1):
                lo, hi = args.bins[bi], args.bins[bi + 1]
                tot = [0, 0]
                for c in CLASSES:
                    n_hit, n_gt = acc.get((c, bi, point), (0, 0))
                    tot[0] += n_hit
                    tot[1] += n_gt
                    if n_gt:
                        rows.append((corruption, sev, seed, point, c, lo, hi, n_gt, n_hit / n_gt))
                if tot[1]:
                    rows.append((corruption, sev, seed, point, "all", lo, hi, tot[1], tot[0] / tot[1]))
        print(f"{d.name}: done")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corruption", "severity", "seed", "operating_point", "class", "range_lo_m", "range_hi_m",
                    "n_gt", "recall"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
