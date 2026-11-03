"""Controlled latency / memory benchmark (supersedes the single-pass latency inside evaluate.py).

Protocol: for each (dataset, precision) the first N images (sorted by filename) are timed with
batch=1, CUDA-synchronised, conf=0.25 (deployment threshold; NMS cost depends on #candidates).
Each model is (re)loaded, warmed up for WARMUP images (steady GPU clocks, cudnn, allocator), then
timed on N images.  This is repeated for ROUNDS interleaved rounds (models take turns, so slow
clock/thermal drift hits every model equally).  We report the median over rounds of the per-round
median latency, plus min/max round.  Peak memory = max_memory_allocated() minus the memory
allocated before the model was loaded (weights + activations, excludes the CUDA context).
"""
import argparse
import gc
import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from evaluate import RESULTS, build, load_gt

MODELS = ["yolo11n", "yolo11s", "yolo11m", "rtdetr-l", "frcnn", "fcos"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=40)
    a = ap.parse_args()
    out = {}
    for ds in ("kitti", "coco"):
        _, img_dir = load_gt(ds)
        names = sorted(p.name for p in img_dir.iterdir())[: a.n]
        imgs = [cv2.imread(str(img_dir / n)) for n in names]
        for half in (False, True):
            prec = "fp16" if half else "fp32"
            rounds = {m: [] for m in MODELS}
            mem = {}
            for r in range(a.rounds):
                for m in MODELS:
                    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
                    base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
                    ad = build(SimpleNamespace(model=m, imgsz=None, conf=0.25, iou=None, max_det=100), half)
                    for i in range(a.warmup):
                        ad.predict(imgs[i % len(imgs)])
                    lat = []
                    for im in imgs:
                        torch.cuda.synchronize(); t = time.perf_counter()
                        ad.predict(im)
                        torch.cuda.synchronize(); lat.append((time.perf_counter() - t) * 1000)
                    rounds[m].append(float(np.median(lat)))
                    mem[m] = (torch.cuda.max_memory_allocated() - base) / 2**20
                    del ad
                print(ds, prec, "round", r, {m: round(rounds[m][-1], 1) for m in MODELS}, flush=True)
            for m in MODELS:
                out[f"{m}|{ds}|{prec}"] = {"median_ms": float(np.median(rounds[m])), "min_round": min(rounds[m]),
                                          "max_round": max(rounds[m]), "fps": 1000 / float(np.median(rounds[m])),
                                          "peak_mem_MB": mem[m]}
    (RESULTS / "latency_bench.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
