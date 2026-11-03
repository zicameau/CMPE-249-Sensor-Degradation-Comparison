"""Time the fast nuScenes eval at full-val scale on synthetic data (6019 samples, 150 scenes, 200 preds each).

    python common/tools/bench_nusc_eval.py [--n-samples 6019] [--preds 200] [--boot 20]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sdc_common import nusc_eval as NE  # noqa: E402


def synth(n_samples, n_scenes, n_pred, rng):
    gt, pred, scene_of = {}, {}, {}
    for i in range(n_samples):
        tok = f"t{i}"
        scene_of[tok] = i * n_scenes // n_samples
        ng = int(rng.integers(5, 40))
        gxy = rng.uniform(-50, 50, (ng, 2))
        gcls = rng.integers(0, 10, ng)
        mk = lambda xy, cls, n, sc: NE.Boxes(xy, rng.uniform(0.5, 4, (n, 3)), rng.uniform(-3, 3, n),
                                             rng.normal(0, 2, (n, 2)), cls, np.full(n, -1), sc)
        gt[tok] = mk(gxy, gcls, ng, np.zeros(ng))
        det = rng.random(ng) < 0.7
        pxy = np.concatenate([gxy[det] + rng.normal(0, 0.5, (det.sum(), 2)),
                              rng.uniform(-50, 50, (n_pred - det.sum(), 2))])
        pcls = np.concatenate([gcls[det], rng.integers(0, 10, n_pred - det.sum())])
        psc = np.concatenate([rng.uniform(0.3, 1, det.sum()), rng.uniform(0, 0.5, n_pred - det.sum())])
        pred[tok] = mk(pxy, pcls, n_pred, psc)
    return gt, pred, scene_of


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=6019)
    ap.add_argument("--n-scenes", type=int, default=150)
    ap.add_argument("--preds", type=int, default=200)
    ap.add_argument("--boot", type=int, default=20)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    gt, pred, scene_of = synth(a.n_samples, a.n_scenes, a.preds, rng)
    t0 = time.time(); table = NE.build_match_table(gt, pred, scene_of, a.n_scenes); t1 = time.time()
    print(f"build_match_table: {t1 - t0:.1f}s for {a.n_samples * a.preds:,} predictions")
    t0 = time.time(); m = NE.evaluate(table); t1 = time.time()
    print(f"evaluate (1 resample): {t1 - t0:.2f}s  mAP={m.map:.3f}")
    t0 = time.time(); NE.bootstrap_draws(table, n_boot=a.boot); t1 = time.time()
    per = (t1 - t0) / a.boot
    print(f"bootstrap: {per:.2f}s/resample -> {per * 1000 / 60:.1f} min for 1,000 resamples")
