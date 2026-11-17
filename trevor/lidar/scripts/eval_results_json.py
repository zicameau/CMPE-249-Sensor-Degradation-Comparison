#!/usr/bin/env python
"""Run the official nuScenes detection devkit on an existing ``results_nusc.json`` (what BEVFusion's test run writes).

Use when the evaluation step of a test run died (the devkit needs several GB for 6,019 samples x up to 500 boxes) but
the detections were saved. Same call as ``NuScenesDataset.evaluate``: config ``detection_cvpr_2019``, eval_set val.

    python eval_results_json.py --run-dir /results/gate_orig --dataroot data/nuscenes --version v1.0-trainval
Writes <run-dir>/metrics_summary.json (+ plots/) like the test run would.
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-trainval")
    a = ap.parse_args()

    from nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    nusc = NuScenes(version=a.version, dataroot=a.dataroot, verbose=False)
    eval_set = {"v1.0-mini": "mini_val", "v1.0-trainval": "val"}[a.version]
    ev = NuScenesEval(nusc, config=config_factory("detection_cvpr_2019"),
                      result_path=os.path.join(a.run_dir, "results_nusc.json"),
                      eval_set=eval_set, output_dir=a.run_dir, verbose=True)
    ev.main(render_curves=False)
    m = json.load(open(os.path.join(a.run_dir, "metrics_summary.json")))
    print(f"mAP: {m['mean_ap']:.4f}\nNDS: {m['nd_score']:.4f}")


if __name__ == "__main__":
    main()
