"""(a) For GT objects missed because of a wrong label (IoU>=0.5 box exists, wrong class, score>=THR): confusion
    (GT class -> predicted COCO label).
(b) For 'truck' detections (score>=THR) that do not match a KITTI Truck: which KITTI class do they sit on?
    Shows how much of the truck 'false-positive' mass is a taxonomy artefact (Van/Car/Tram/Misc)."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze import CAT, MAIN, THR, iou_matrix

HERE = Path(__file__).parent
R = HERE / "results"
COCO_NAMES = {c["id"]: c["name"] for c in json.loads((HERE / "data/coco/annotations/instances_val2017.json").read_text())["categories"]}
gt = json.loads((HERE / "data/kitti/kitti_subset_gt.json").read_text())
A = defaultdict(list)
for a in gt["annotations"]:
    A[a["image_id"]].append(a)

# raw KITTI objects (before taxonomy mapping): one entry per (kitti object); the mapping can duplicate boxes, so dedupe by bbox
def raw(anns):
    seen, out = set(), []
    for a in anns:
        key = (tuple(a["bbox"]), a["kitti_class"])
        if key not in seen:
            seen.add(key); out.append(a)
    return out


res = {}
for tag in MAIN:
    dets = defaultdict(list)
    for d in json.loads((R / f"{tag}_kitti_dets.json").read_text()):
        if d["score"] >= THR:
            dets[d["image_id"]].append(d)
    conf = Counter(); on = Counter(); n_truck_fp = 0
    for im, anns in A.items():
        D = dets.get(im, [])
        if not D:
            continue
        db = np.array([[d["bbox"][0], d["bbox"][1], d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3]] for d in D])
        g = [a for a in anns if not a["iscrowd"]]
        gb = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in g]).reshape(-1, 4)
        M = iou_matrix(gb, db)
        for k, a in enumerate(g):
            same = [j for j, d in enumerate(D) if d["category_id"] == a["category_id"] and M[k, j] >= 0.5]
            wrong = [j for j, d in enumerate(D) if d["category_id"] != a["category_id"] and M[k, j] >= 0.5]
            if not same and wrong:
                j = max(wrong, key=lambda j: M[k, j])
                conf[f"{CAT[a['category_id']]}->{COCO_NAMES[D[j]['category_id']]}"] += 1
        # truck detections not matching a GT truck
        rawg = raw(anns)
        rb = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in rawg]).reshape(-1, 4)
        MR = iou_matrix(db, rb)
        for j, d in enumerate(D):
            if d["category_id"] != 8:
                continue
            if any(rawg[k]["kitti_class"] == "Truck" and MR[j, k] >= 0.5 for k in range(len(rawg))):
                continue
            n_truck_fp += 1
            if len(rawg) and MR[j].max() >= 0.5:
                on[rawg[int(MR[j].argmax())]["kitti_class"]] += 1
            else:
                on["no GT (IoU<0.5)"] += 1
    res[tag] = {"label_confusions": conf.most_common(6), "truck_nonmatch": n_truck_fp, "truck_nonmatch_on": on.most_common()}
(R / "taxonomy.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
