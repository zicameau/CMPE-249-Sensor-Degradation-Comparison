"""(1) Dataset statistics per class (size / occlusion) for COCO-val and the KITTI subset.
(2) Duplicate analysis on KITTI at score >= THR: does NMS-free (RT-DETR) emit more duplicate boxes than NMS-based models?
    * same-class duplicate : a detection that is not a TP but overlaps (IoU>=0.5) a GT already claimed by a
                             higher-scoring detection of the same class.
    * cross-class duplicate: two detections of different classes with IoU>=0.9 (one object, two labels) -- possible for
                             RT-DETR because top-k runs over the flattened (query x class) matrix.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze import CAT, MAIN, THR, iou_matrix

HERE = Path(__file__).parent
R = HERE / "results"
out = {"stats": {}, "dups": {}}

# ---- (1) dataset statistics
k = json.loads((HERE / "data/kitti/kitti_subset_gt.json").read_text())
st = defaultdict(list)
for a in k["annotations"]:
    if not a["iscrowd"]:
        st[CAT[a["category_id"]]].append(a)
out["stats"]["kitti"] = {}
for c, v in st.items():
    h = np.array([a["bbox"][3] for a in v])
    out["stats"]["kitti"][c] = {"n": len(v), "median_h_px": float(np.median(h)), "pct_h_lt40": float(100 * (h < 40).mean()),
                                "pct_occluded": float(100 * np.mean([a["occluded"] >= 1 for a in v])),
                                "pct_truncated": float(100 * np.mean([a["truncated"] > 0.15 for a in v]))}
c = json.loads((HERE / "data/coco/annotations/instances_val2017.json").read_text())
out["stats"]["coco"] = {}
for cat in CAT:
    v = [a for a in c["annotations"] if a["category_id"] == cat and not a["iscrowd"]]
    ar = np.array([a["area"] for a in v]); h = np.array([a["bbox"][3] for a in v])
    out["stats"]["coco"][CAT[cat]] = {"n": len(v), "pct_small": float(100 * (ar < 32 ** 2).mean()),
                                      "pct_medium": float(100 * ((ar >= 32 ** 2) & (ar < 96 ** 2)).mean()),
                                      "pct_large": float(100 * (ar >= 96 ** 2).mean()), "median_h_px": float(np.median(h))}

# ---- (2) duplicates
by_img = defaultdict(list)
for a in k["annotations"]:
    if not a["iscrowd"]:
        by_img[a["image_id"]].append(a)
for tag in MAIN:
    dets = defaultdict(list)
    for d in json.loads((R / f"{tag}_kitti_dets.json").read_text()):
        if d["score"] >= THR and d["category_id"] in CAT:
            dets[d["image_id"]].append(d)
    n_det = tp = dup = xdup = 0
    for im in {a["image_id"] for v in by_img.values() for a in v} | set(dets):
        D = sorted(dets.get(im, []), key=lambda d: -d["score"])
        if not D:
            continue
        db = np.array([[d["bbox"][0], d["bbox"][1], d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3]] for d in D])
        G = by_img.get(im, [])
        gb = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in G]).reshape(-1, 4)
        M = iou_matrix(db, gb)
        claimed = set()
        for i, d in enumerate(D):
            n_det += 1
            best, bj = 0.0, -1
            for j, a in enumerate(G):
                if a["category_id"] == d["category_id"] and j not in claimed and M[i, j] >= 0.5 and M[i, j] > best:
                    best, bj = M[i, j], j
            if bj >= 0:
                claimed.add(bj); tp += 1
            elif any(a["category_id"] == d["category_id"] and j in claimed and M[i, j] >= 0.5 for j, a in enumerate(G)):
                dup += 1
        DD = iou_matrix(db, db)
        cats = np.array([d["category_id"] for d in D])
        iu = np.triu_indices(len(D), 1)
        xdup += int(((DD[iu] >= 0.9) & (cats[iu[0]] != cats[iu[1]])).sum())
    out["dups"][tag] = {"dets": n_det, "tp": tp, "same_class_dup": dup, "cross_class_dup_pairs": xdup,
                        "dup_rate_pct": 100 * dup / max(1, n_det)}
(R / "extra.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
