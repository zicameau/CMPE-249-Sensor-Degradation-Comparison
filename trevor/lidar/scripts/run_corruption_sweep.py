#!/usr/bin/env python
"""Run the LiDAR corruption sweep with BEVFusion (LiDAR-only): one folder of detections per run.

Runs one clean evaluation, then every (corruption, severity, seed). Stochastic corruptions get
3 seeds, deterministic ones 1. Each run is a separate process (``torchpack dist-run``) in its own
directory; a finished run (has ``metrics_summary.json``) is skipped, so the sweep is resumable.
Each run directory keeps ``results_nusc.json`` (detections) and ``metrics_summary.json`` (devkit numbers).

Example (inside the Docker image, from /workspace/bevfusion):

    python /workspace/sdc/lidar/scripts/run_corruption_sweep.py --out-dir /results/lidar_sweep

Use --dry-run first to print the commands. Then compute CIs / RR / mRR / failure thresholds and write the
shared CSV with ``common/tools/bootstrap_sweep.py`` (the command is printed at the end).
"""
from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "lidar")]

from sdc_lidar import corruptions as C  # noqa: E402

DEFAULT_CONFIG = "configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml"
DEFAULT_CKPT = "pretrained/lidar-only-det.pth"


def plan(corruptions, severities, seeds_override):
    yield ("clean", 0, 0)
    for name in corruptions:
        seeds = seeds_override or C.seeds_for(name)
        if not C.REGISTRY[name].stochastic:
            seeds = seeds[:1]
        for sev, seed in itertools.product(severities, seeds):
            yield (name, sev, seed)


def run_name(name, sev, seed):
    return "clean" if name == "clean" else f"{name}_s{sev}_seed{seed}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bevfusion-dir", default=os.getcwd(), help="BEVFusion checkout (cwd for the runs)")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--corruptions", nargs="+", default=list(C.REGISTRY), choices=list(C.REGISTRY))
    ap.add_argument("--severities", nargs="+", type=int, default=list(C.SEVERITIES))
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="override the default seeds (0 1 2)")
    ap.add_argument("--np", type=int, default=1, help="GPUs for torchpack dist-run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    runner = ROOT / "lidar" / "bevfusion" / "run_test.py"
    todo = list(plan(args.corruptions, args.severities, args.seeds))
    print(f"{len(todo)} runs planned")
    failed = []

    for i, (name, sev, seed) in enumerate(todo, 1):
        rdir = out_dir / run_name(name, sev, seed)
        tag = f"[{i}/{len(todo)}] {run_name(name, sev, seed)}"
        if (rdir / "metrics_summary.json").exists():
            print(f"{tag}: done, skipping")
        else:
            env = dict(os.environ)
            env.pop("SDC_CORRUPTION", None)
            if name != "clean":
                env.update(SDC_CORRUPTION=name, SDC_SEVERITY=str(sev), SDC_SEED=str(seed))
            cmd = ["torchpack", "dist-run", "-np", str(args.np), "python", str(runner),
                   args.config, args.checkpoint, "--eval", "bbox",
                   "--eval-options", f"jsonfile_prefix={rdir}"]
            print(f"{tag}: {C.REGISTRY[name].describe(sev) if name != 'clean' else 'clean'}\n    "
                  + " ".join(f"{k}={env[k]}" for k in env if k.startswith("SDC_"))
                  + " " + " ".join(shlex.quote(c) for c in cmd))
            if args.dry_run:
                continue
            rdir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            with open(rdir / "run.log", "w", encoding="utf-8") as log:
                rc = subprocess.run(cmd, cwd=args.bevfusion_dir, env=env, stdout=log,
                                    stderr=subprocess.STDOUT).returncode
            print(f"    exit {rc} in {time.time() - t0:.0f}s (log: {rdir / 'run.log'})")
            if rc != 0 or not (rdir / "metrics_summary.json").exists():
                print("    FAILED - continuing with the next run (rerun the script to retry)", file=sys.stderr)
                failed.append(run_name(name, sev, seed))

    if args.dry_run:
        return
    if failed:
        print(f"\n{len(failed)} run(s) failed: {', '.join(failed)}")
    print("\nNext (CIs, RR, mRR, failure thresholds -> shared CSV):\n"
          f"  python {ROOT / 'common' / 'tools' / 'bootstrap_sweep.py'} --sweep-dir {out_dir} \\\n"
          "      --nuscenes-root data/nuscenes --model bevfusion_lidar --lane lidar \\\n"
          "      --hardware <laptop|hpc_a40> --results-csv /results/results.csv")


if __name__ == "__main__":
    main()
