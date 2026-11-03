"""Per-stage latency of the Ultralytics models (Results.speed: preprocess / inference / postprocess,
in ms, measured inside the library with CUDA sync) vs. our end-to-end wall clock.  Shows how much
of the end-to-end latency is the network forward pass vs. host-side work (letterbox, NMS, Python)."""
import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from evaluate import RESULTS, build, load_gt

out = {}
for ds in ("kitti", "coco"):
    _, img_dir = load_gt(ds)
    imgs = [cv2.imread(str(img_dir / n)) for n in sorted(p.name for p in img_dir.iterdir())[:100]]
    for m in ("yolo11n", "yolo11s", "yolo11m", "rtdetr-l"):
        ad = build(SimpleNamespace(model=m, imgsz=None, conf=0.25, iou=None, max_det=100))
        for i in range(40):
            ad.m.predict(imgs[i % 100], **ad.kw)
        pre, inf, post, wall = [], [], [], []
        for im in imgs:
            torch.cuda.synchronize(); t = time.perf_counter()
            r = ad.m.predict(im, **ad.kw)[0]
            torch.cuda.synchronize(); wall.append((time.perf_counter() - t) * 1000)
            pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
        out[f"{m}|{ds}"] = {k: float(np.median(v)) for k, v in
                           dict(preprocess=pre, inference=inf, postprocess=post, wall=wall).items()}
        print(m, ds, out[f"{m}|{ds}"], flush=True)
        del ad
(RESULTS / "latency_breakdown.json").write_text(json.dumps(out, indent=1))
