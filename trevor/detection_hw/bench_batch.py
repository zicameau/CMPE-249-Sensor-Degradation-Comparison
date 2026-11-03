"""Is batch-1 latency launch-bound?  Time the raw (fused) network forward for batch 1..16 at 640x640.
If time(batch=B) ~ time(batch=1), the GPU is idle most of the time (CPU/driver launch-bound)."""
import json
import time

import torch
from ultralytics import YOLO, RTDETR

from evaluate import RESULTS

out = {}
for name, cls in (("yolo11n.pt", YOLO), ("yolo11s.pt", YOLO), ("yolo11m.pt", YOLO), ("rtdetr-l.pt", RTDETR)):
    net = cls(name).model.cuda().eval().fuse()
    for B in (1, 2, 4, 8, 16):
        x = torch.randn(B, 3, 640, 640, device="cuda")
        with torch.inference_mode():
            for _ in range(10):
                net(x)
            ts = []
            for _ in range(20):
                torch.cuda.synchronize(); t = time.perf_counter(); net(x); torch.cuda.synchronize()
                ts.append((time.perf_counter() - t) * 1000)
        ts.sort()
        out[f"{name}|{B}"] = {"batch_ms": ts[len(ts) // 2], "per_img_ms": ts[len(ts) // 2] / B}
        print(name, B, out[f"{name}|{B}"], flush=True)
        del x
    del net
    torch.cuda.empty_cache()
(RESULTS / "latency_batch.json").write_text(json.dumps(out, indent=1))
