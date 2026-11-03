"""Post-hoc analysis: tables (results/summary.md), figures (report/fig_*.png), failure analysis."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
R = HERE / "results"
FIG = HERE / "report"
FIG.mkdir(exist_ok=True)
MAIN = ["yolo11s", "rtdetr-l", "frcnn", "fcos"]
LABEL = {"yolo11s": "YOLO11s", "rtdetr-l": "RT-DETR-L", "frcnn": "Faster R-CNN R50-FPN", "fcos": "FCOS R50-FPN",
         "yolo11n": "YOLO11n", "yolo11m": "YOLO11m"}
CAT = {1: "person", 3: "car", 8: "truck"}
THR = 0.30  # operating threshold for the failure analysis


def load(tag):
    p = R / f"{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def iou_matrix(a, b):  # xyxy
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None] - inter + 1e-9)


def f(x, d=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}"


def tables(out):
    out.append("## Main results\n")
    out.append("| Model | Dataset | mAP | AP50 | AP75 | APs | APm | APl | AR100 | ms (median) | FPS | p95 ms | GPU MB | Params M |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag in MAIN + ["yolo11n", "yolo11m"]:
        for ds in ("coco", "kitti"):
            r = load(f"{tag}_{ds}")
            if not r:
                continue
            for key, nm in (("metrics_coco80", "COCO-80"), ("metrics_3cls", f"{ds.upper()} 3-cls")):
                m = r.get(key)
                if not m or (key == "metrics_3cls" and ds == "coco"):
                    nm = "COCO 3-cls" if key == "metrics_3cls" else nm
                if not m:
                    continue
                out.append(f"| {LABEL[tag]} | {nm} | {f(m['mAP'])} | {f(m['AP50'])} | {f(m['AP75'])} | {f(m['APs'])} | "
                           f"{f(m['APm'])} | {f(m['APl'])} | {f(m['AR100'])} | {r['latency_ms']['median']:.1f} | "
                           f"{r['fps_median']:.1f} | {r['latency_ms']['p95']:.1f} | {r['peak_gpu_mem_MB']:.0f} | {r['params_M']:.1f} |")
    out.append("\n## Per-class AP (3-class subset; AP / AP50)\n")
    out.append("| Model | Dataset | person | car | truck |")
    out.append("|---|---|---|---|---|")
    for tag in MAIN:
        for ds in ("coco", "kitti"):
            r = load(f"{tag}_{ds}")
            if r:
                pc = r["metrics_3cls"]["per_class"]
                out.append(f"| {LABEL[tag]} | {ds} | " + " | ".join(f"{f(pc[c]['AP'])} / {f(pc[c]['AP50'])}" for c in ("person", "car", "truck")) + " |")
    out.append("\n## Ablations on KITTI (3-class mAP / AP50 / AR100 / median ms)\n")
    out.append("| Config | mAP | AP50 | APs | APm | APl | AR100 | ms |")
    out.append("|---|---|---|---|---|---|---|---|")
    for tag, nm in [("yolo11s_kitti", "YOLO11s imgsz640 NMS0.7 (default)"), ("yolo11s_nms0.3_kitti", "YOLO11s NMS IoU 0.3"),
                    ("yolo11s_nms0.5_kitti", "YOLO11s NMS IoU 0.5"), ("yolo11s_nms0.9_kitti", "YOLO11s NMS IoU 0.9"),
                    ("yolo11s_imgsz1280_kitti", "YOLO11s imgsz1280"), ("rtdetr-l_kitti", "RT-DETR-L imgsz640"),
                    ("rtdetr-l_imgsz1280_kitti", "RT-DETR-L imgsz1280")]:
        r = load(tag)
        if r:
            m = r["metrics_3cls"]
            out.append(f"| {nm} | {f(m['mAP'])} | {f(m['AP50'])} | {f(m['APs'])} | {f(m['APm'])} | {f(m['APl'])} | {f(m['AR100'])} | {r['latency_ms']['median']:.1f} |")


def failure_analysis(out):
    gt = json.loads((HERE / "data/kitti/kitti_subset_gt.json").read_text())
    by_img = defaultdict(list)
    for a in gt["annotations"]:
        by_img[a["image_id"]].append(a)
    buckets = [("<25px", 0, 25), ("25-40px", 25, 40), ("40-80px", 40, 80), (">80px", 80, 1e9)]
    out.append(f"\n## KITTI failure analysis (operating threshold score >= {THR}, match IoU >= 0.5)\n")
    recall_curves = {}
    cause_tbl = {}
    for tag in MAIN:
        p = R / f"{tag}_kitti_dets.json"
        if not p.exists():
            continue
        dets = defaultdict(list)
        for d in json.loads(p.read_text()):
            dets[d["image_id"]].append(d)
        hit = defaultdict(lambda: [0, 0])  # key -> [found, total]
        causes = defaultdict(int)
        fp = defaultdict(int)
        n_det_hi = defaultdict(int)
        for im, anns in by_img.items():
            g = [a for a in anns if not a["iscrowd"]]
            ign = [a for a in anns if a["iscrowd"]]
            D = dets.get(im, [])
            db = np.array([[d["bbox"][0], d["bbox"][1], d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3]] for d in D]).reshape(-1, 4)
            ds = np.array([d["score"] for d in D]); dc = np.array([d["category_id"] for d in D])
            gb = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in g]).reshape(-1, 4)
            M = iou_matrix(gb, db)
            for k, a in enumerate(g):
                h = a["bbox"][3]
                hb = next(n for n, lo, hi in buckets if lo <= h < hi)
                cname = CAT[a["category_id"]]
                same = (dc == a["category_id"]) if len(D) else np.zeros(0, bool)
                hi_ = ds >= THR if len(D) else np.zeros(0, bool)
                found = bool(len(D)) and bool(((M[k] >= 0.5) & same & hi_).any())
                for key in (("all", hb), (cname, hb), (cname, "all"), ("all", "all"), ("occl", a["occluded"])):
                    hit[key][1] += 1; hit[key][0] += found
                if not found:
                    if not len(D):
                        c = "no detection near GT"
                    elif ((M[k] >= 0.5) & same & ~hi_).any():
                        c = "correct but score < thr"
                    elif ((M[k] >= 0.5) & ~same & hi_).any():
                        c = "wrong class (IoU>=0.5)"
                    elif ((M[k] >= 0.1) & same & hi_).any():
                        c = "poor localisation (0.1<=IoU<0.5)"
                    else:
                        c = "no detection near GT"
                    causes[c] += 1
            # false positives at THR: det with no same-class GT IoU>=0.5 and not inside ignore region
            ib = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in ign]).reshape(-1, 4)
            for j in np.where(ds >= THR)[0] if len(D) else []:
                if dc[j] not in CAT:
                    continue
                n_det_hi[CAT[dc[j]]] += 1
                ok = any(M[k, j] >= 0.5 and g[k]["category_id"] == dc[j] for k in range(len(g)))
                if not ok and len(ib):
                    # IoA with ignore regions
                    x1 = np.maximum(ib[:, 0], db[j, 0]); y1 = np.maximum(ib[:, 1], db[j, 1])
                    x2 = np.minimum(ib[:, 2], db[j, 2]); y2 = np.minimum(ib[:, 3], db[j, 3])
                    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
                    ok = bool((inter / ((db[j, 2] - db[j, 0]) * (db[j, 3] - db[j, 1]) + 1e-9) >= 0.5).any())
                if not ok:
                    fp[CAT[dc[j]]] += 1
        recall_curves[tag] = hit
        cause_tbl[tag] = (causes, fp, n_det_hi)
    out.append("### Recall by GT box height (all classes)\n")
    out.append("| Model | <25px | 25-40px | 40-80px | >80px | overall |")
    out.append("|---|---|---|---|---|---|")
    for tag, h in recall_curves.items():
        cells = [f"{100 * h[('all', b[0])][0] / max(1, h[('all', b[0])][1]):.1f} (n={h[('all', b[0])][1]})" for b in buckets]
        out.append(f"| {LABEL[tag]} | " + " | ".join(cells) + f" | {100 * h[('all', 'all')][0] / h[('all', 'all')][1]:.1f} |")
    out.append("\n### Recall by class\n")
    out.append("| Model | person | car | truck |")
    out.append("|---|---|---|---|")
    for tag, h in recall_curves.items():
        out.append(f"| {LABEL[tag]} | " + " | ".join(f"{100 * h[(c, 'all')][0] / max(1, h[(c, 'all')][1]):.1f} (n={h[(c, 'all')][1]})" for c in ("person", "car", "truck")) + " |")
    out.append("\n### Recall by KITTI occlusion level (0=visible, 1=partly, 2=largely, 3=unknown)\n")
    out.append("| Model | 0 | 1 | 2 | 3 |")
    out.append("|---|---|---|---|---|")
    for tag, h in recall_curves.items():
        out.append(f"| {LABEL[tag]} | " + " | ".join(f"{100 * h[('occl', o)][0] / max(1, h[('occl', o)][1]):.1f} (n={h[('occl', o)][1]})" for o in range(4)) + " |")
    out.append("\n### Why GT objects are missed (count of missed GT)\n")
    names = ["correct but score < thr", "wrong class (IoU>=0.5)", "poor localisation (0.1<=IoU<0.5)", "no detection near GT"]
    out.append("| Model | " + " | ".join(names) + " | total missed |")
    out.append("|---|" + "---|" * (len(names) + 1))
    for tag, (c, fp, nd) in cause_tbl.items():
        out.append(f"| {LABEL[tag]} | " + " | ".join(str(c[n]) for n in names) + f" | {sum(c.values())} |")
    out.append("\n### False positives at threshold (not matching GT or ignore region)\n")
    out.append("| Model | person FP / dets | car FP / dets | truck FP / dets |")
    out.append("|---|---|---|---|")
    for tag, (c, fp, nd) in cause_tbl.items():
        out.append(f"| {LABEL[tag]} | " + " | ".join(f"{fp[k]} / {nd[k]}" for k in ("person", "car", "truck")) + " |")

    dump = {tag: {"recall": {f"{k[0]}|{k[1]}": v for k, v in h.items()},
                  "causes": dict(cause_tbl[tag][0]), "fp": dict(cause_tbl[tag][1]), "dets_hi": dict(cause_tbl[tag][2])}
            for tag, h in recall_curves.items()}
    (R / "failure.json").write_text(json.dumps(dump, indent=1))

    # figure: recall vs height
    fig, ax = plt.subplots(figsize=(5, 3.3))
    for tag, h in recall_curves.items():
        ax.plot([b[0] for b in buckets], [100 * h[("all", b[0])][0] / max(1, h[("all", b[0])][1]) for b in buckets], marker="o", label=LABEL[tag])
    ax.set_xlabel("GT box height (KITTI px)"); ax.set_ylabel(f"Recall @ IoU 0.5, score>={THR} (%)"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig_recall_height.png", dpi=200); plt.close(fig)


def figures():
    # speed-accuracy
    bench = json.loads((R / "latency_bench.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.3))
    for ax, ds, key, ttl in ((axs[0], "coco", "metrics_3cls", "COCO val (3-class mAP)"), (axs[1], "kitti", "metrics_3cls", "KITTI subset (3-class mAP)")):
        for tag in MAIN + ["yolo11n", "yolo11m"]:
            r = load(f"{tag}_{ds}")
            b = bench.get(f"{tag}|{ds}|fp32")
            if r and b:
                ax.scatter(b["median_ms"], 100 * r[key]["mAP"], s=20 + b["peak_mem_MB"] / 6,
                           marker="o" if tag in MAIN else "s", label=LABEL[tag])
        ax.set_xlabel("median latency, fp32 batch 1 (ms)"); ax.set_ylabel("mAP@[.5:.95] (%)"); ax.set_title(ttl + " (marker area ~ GPU mem)", fontsize=8); ax.grid(alpha=.3)
    axs[1].legend(fontsize=6, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "fig_speed_acc.png", dpi=200); plt.close(fig)
    # cross-dataset per-class
    fig, axs = plt.subplots(1, 3, figsize=(8, 2.8), sharey=True)
    for ax, c in zip(axs, ("person", "car", "truck")):
        w = 0.38
        for i, ds in enumerate(("coco", "kitti")):
            ax.bar(np.arange(4) + (i - .5) * w, [100 * (load(f"{t}_{ds}") or {"metrics_3cls": {"per_class": {c: {"AP": 0}}}})["metrics_3cls"]["per_class"][c]["AP"] for t in MAIN], w, label=ds.upper())
        ax.set_xticks(range(4)); ax.set_xticklabels(["Y11s", "RTDETR", "FRCNN", "FCOS"], fontsize=7); ax.set_title(c, fontsize=9)
    axs[0].set_ylabel("AP@[.5:.95] (%)"); axs[0].legend(fontsize=7, loc="lower left")
    axs[0].set_ylim(0, 75)
    fig.tight_layout(); fig.savefig(FIG / "fig_per_class.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    out = []
    tables(out)
    failure_analysis(out)
    figures()
    (R / "summary.md").write_text("\n".join(out))
    print("\n".join(out))
