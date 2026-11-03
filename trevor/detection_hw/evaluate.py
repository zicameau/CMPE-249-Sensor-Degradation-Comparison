"""Unified 2D detection evaluation harness (Track A).

Every model is wrapped in an adapter returning post-processed detections
(xyxy in original-image pixels, score, COCO category id).  Latency is measured
per image (batch=1, fp32, CUDA-synchronised) and includes resize/letterbox,
forward pass, decoding and NMS (where the model has one) -- but NOT disk/JPEG decode.

Example:
  python evaluate.py --model yolo11s --dataset kitti
"""
import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"
EVAL_CATS = [1, 3, 8]  # person, car, truck
CAT_NAMES = {1: "person", 3: "car", 8: "truck"}

# Ultralytics' 80-class index -> COCO category id
COCO80_TO_91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


class UltralyticsAdapter:
    def __init__(self, weights, imgsz, conf, iou, max_det, half=False):
        from ultralytics import YOLO, RTDETR
        cls = RTDETR if "rtdetr" in weights else YOLO
        self.m = cls(weights)
        self.kw = dict(imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=0, verbose=False)
        if half:
            self.kw["quantize"] = "fp16"
        self.ckpt = weights

    def predict(self, bgr):
        r = self.m.predict(bgr, **self.kw)[0]
        b = r.boxes
        return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), [COCO80_TO_91[int(c)] for c in b.cls.cpu().numpy()]


class TorchvisionAdapter:
    def __init__(self, name, conf, iou, max_det, imgsz, half=False):
        import torchvision.models.detection as D
        self.half = half
        kw = {}
        if imgsz:  # shorter side; keep torchvision default ratio for max_size
            kw.update(min_size=imgsz, max_size=int(imgsz * 1333 / 800))
        if name == "frcnn":
            self.m = D.fasterrcnn_resnet50_fpn(weights="COCO_V1", box_score_thresh=conf, box_nms_thresh=iou,
                                               box_detections_per_img=max_det, **kw)
            self.ckpt = "torchvision fasterrcnn_resnet50_fpn COCO_V1"
        elif name == "fcos":
            self.m = D.fcos_resnet50_fpn(weights="COCO_V1", score_thresh=conf, nms_thresh=iou,
                                         detections_per_img=max_det, topk_candidates=1000, **kw)
            self.ckpt = "torchvision fcos_resnet50_fpn COCO_V1"
        self.m.eval().cuda()

    @torch.inference_mode()
    def predict(self, bgr):
        x = torch.from_numpy(bgr[:, :, ::-1].copy()).cuda().permute(2, 0, 1).float() / 255.0
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
            o = self.m([x])[0]
        return o["boxes"].float().cpu().numpy(), o["scores"].float().cpu().numpy(), o["labels"].cpu().tolist()


def build(a, half=False):
    if a.model in ("yolo11n", "yolo11s", "yolo11m", "rtdetr-l"):
        w = {"yolo11n": "yolo11n.pt", "yolo11s": "yolo11s.pt", "yolo11m": "yolo11m.pt", "rtdetr-l": "rtdetr-l.pt"}[a.model]
        return UltralyticsAdapter(w, a.imgsz or 640, a.conf, a.iou or 0.7, a.max_det, half)
    iou = a.iou or {"frcnn": 0.5, "fcos": 0.6}[a.model]
    return TorchvisionAdapter(a.model, a.conf, iou, a.max_det, a.imgsz, half)


def load_gt(dataset):
    if dataset == "coco":
        return DATA / "coco" / "annotations" / "instances_val2017.json", DATA / "coco" / "val2017"
    return DATA / "kitti" / "kitti_subset_gt.json", DATA / "kitti" / "images"


def summarize(ev, cat_ids):
    s = ev.stats
    out = {"mAP": s[0], "AP50": s[1], "AP75": s[2], "APs": s[3], "APm": s[4], "APl": s[5],
           "AR1": s[6], "AR10": s[7], "AR100": s[8], "ARs": s[9], "ARm": s[10], "ARl": s[11]}
    pr = ev.eval["precision"]  # [T,R,K,A,M]
    per_class = {}
    for k, c in enumerate(cat_ids):
        p = pr[:, :, k, 0, 2]
        p50 = pr[0, :, k, 0, 2]
        per_class[CAT_NAMES.get(c, str(c))] = {
            "AP": float(np.mean(p[p > -1])) if (p > -1).any() else float("nan"),
            "AP50": float(np.mean(p50[p50 > -1])) if (p50 > -1).any() else float("nan")}
    out["per_class"] = per_class
    return out


def coco_eval(gt, dets, cat_ids, img_ids):
    if not dets:
        return None
    dt = gt.loadRes(dets)
    ev = COCOeval(gt, dt, "bbox")
    ev.params.catIds = cat_ids
    ev.params.imgIds = img_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return summarize(ev, cat_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["yolo11n", "yolo11s", "yolo11m", "rtdetr-l", "frcnn", "fcos"])
    ap.add_argument("--dataset", required=True, choices=["coco", "kitti"])
    ap.add_argument("--imgsz", type=int, default=None, help="ultralytics: long side; torchvision: short side")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=None, help="NMS IoU (ignored by RT-DETR)")
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    gt_path, img_dir = load_gt(a.dataset)
    gt = COCO(str(gt_path))
    img_ids = sorted(gt.getImgIds())[: a.limit]
    ad = build(a)

    for i in img_ids[:10]:  # warm-up (cudnn autotune, lazy init) -- not counted
        ad.predict(cv2.imread(str(img_dir / gt.imgs[i]["file_name"])))
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()

    dets, lat = [], []
    for n, i in enumerate(img_ids):
        img = cv2.imread(str(img_dir / gt.imgs[i]["file_name"]))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        boxes, scores, labels = ad.predict(img)
        torch.cuda.synchronize(); lat.append((time.perf_counter() - t0) * 1000)
        for b, s, l in zip(boxes, scores, labels):
            dets.append({"image_id": i, "category_id": int(l), "score": float(s),
                         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
        if n % 500 == 0:
            print(f"[{n}/{len(img_ids)}] {np.mean(lat):.1f} ms/img", flush=True)

    lat = np.array(lat)
    tag = f"{a.model}{('_' + a.tag) if a.tag else ''}_{a.dataset}"
    (RESULTS / f"{tag}_dets.json").write_text(json.dumps(dets))

    res = {
        "model": a.model, "checkpoint": ad.ckpt, "dataset": a.dataset, "n_images": len(img_ids),
        "args": vars(a), "n_detections": len(dets),
        "latency_ms": {"mean": float(lat.mean()), "median": float(np.median(lat)), "p95": float(np.percentile(lat, 95))},
        "fps_median": float(1000 / np.median(lat)),
        "peak_gpu_mem_MB": torch.cuda.max_memory_allocated() / 2**20,
        "params_M": sum(p.numel() for p in (ad.m.model.parameters() if hasattr(ad.m, "model") else ad.m.parameters())) / 1e6,
        "env": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
                "python": platform.python_version(), "os": platform.platform()},
    }
    try:
        import ultralytics; res["env"]["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    import torchvision; res["env"]["torchvision"] = torchvision.__version__

    if a.dataset == "coco":
        res["metrics_coco80"] = coco_eval(gt, dets, sorted(gt.getCatIds()), img_ids)
        res["metrics_3cls"] = coco_eval(gt, dets, EVAL_CATS, img_ids)
    else:
        res["metrics_3cls"] = coco_eval(gt, dets, EVAL_CATS, img_ids)
    (RESULTS / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("latency_ms", "fps_median", "peak_gpu_mem_MB", "params_M")}, indent=1))


if __name__ == "__main__":
    main()
