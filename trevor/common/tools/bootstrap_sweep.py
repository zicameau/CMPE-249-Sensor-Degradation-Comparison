#!/usr/bin/env python
"""Turn a folder of BEVFusion evaluation runs into CSV rows with scene-bootstrap CIs.

Input: a sweep directory with one sub-folder per run, each holding ``results_nusc.json``:
    clean/   point_drop_s2_seed1/   range_limit_s4_seed0/ ...
(this is what ``lidar/scripts/run_corruption_sweep.py`` produces; any lane that evaluates on nuScenes can
use it). For every run it writes, with 95% CIs from 1,000 scene resamples (same resamples for all runs):

  per run          map, nds, mate, mase, maoe, mave, maae, ap_nusc/<class>, rr_map, rr_nds
  pooled (seed -1) mrr_map, mrr_nds                    (corruption = "all")
                   failure_threshold_map / _nds        (one row per corruption; 6 = never below 0.8)

Match tables are cached as ``<run>/match_table.npz``; delete it to force a rebuild.

    python common/tools/bootstrap_sweep.py --sweep-dir /results/lidar_sweep \\
        --nuscenes-root /workspace/bevfusion/data/nuscenes --model bevfusion_lidar --lane lidar \\
        --hardware hpc_a40 --results-csv /results/results.csv
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdc_common import nusc_eval as NE  # noqa: E402
from sdc_common import robustness as RB  # noqa: E402
from sdc_common import stats  # noqa: E402
from sdc_common.results import AGGREGATE, POOLED_SEED, ResultRow, append_rows  # noqa: E402

RUN_RE = re.compile(r"^(?P<name>[a-z_]+)_s(?P<sev>[1-5])_seed(?P<seed>\d+)$")
ROBUSTNESS_KEYS = ("map", "nds")


def discover_runs(sweep_dir: Path):
    """-> (clean_dir, {(corruption, severity, seed): dir}); only dirs that contain results_nusc.json."""
    clean, runs = None, {}
    for d in sorted(p for p in sweep_dir.iterdir() if p.is_dir() and (p / "results_nusc.json").exists()):
        if d.name == "clean":
            clean = d
        elif (m := RUN_RE.match(d.name)):
            runs[(m["name"], int(m["sev"]), int(m["seed"]))] = d
    return clean, runs


def split_key(flat_key: str):
    metric, _, cls = flat_key.partition("/")
    return metric, (cls or "all")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--nuscenes-root", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lane", required=True, choices=["camera", "lidar", "fusion"])
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--results-csv", required=True)
    ap.add_argument("--n-boot", type=int, default=stats.N_BOOT)
    ap.add_argument("--boot-seed", type=int, default=12345)
    args = ap.parse_args()

    from nuscenes import NuScenes
    from sdc_common import nusc_loader as NL

    sweep = Path(args.sweep_dir)
    clean_dir, run_dirs = discover_runs(sweep)
    if clean_dir is None:
        sys.exit(f"no clean/results_nusc.json in {sweep} - the clean run is required for RR")
    print(f"{len(run_dirs)} corrupted runs + clean")

    t0 = time.time()
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_root, verbose=False)
    gt = NL.load_ground_truth(nusc, args.version)
    print(f"ground truth: {len(gt.boxes)} samples, {len(gt.scene_names)} scenes ({time.time() - t0:.0f}s)")

    W = stats.bootstrap_weights(len(gt.scene_names), args.n_boot, args.boot_seed)   # shared by ALL runs

    def draws_for(d: Path) -> NE.Draws:
        cache = d / "match_table.npz"
        if cache.exists():
            table = NE.load_table(cache)
        else:
            table = NL.build_table_for_run(gt, d / "results_nusc.json")
            NE.save_table(table, cache)
        if ref_table:
            NE.check_paired(ref_table[0], table)       # same scenes, or paired CIs are meaningless
        else:
            ref_table.append(table)
        return NE.bootstrap_draws(table, weights=W)

    ref_table: list = []
    common = dict(model=args.model, lane=args.lane, hardware=args.hardware)
    rows = []

    def add(draws, corruption, severity, seed):
        for key in draws.point:
            est = draws.estimate(key)
            if est.value != est.value or est.ci_low != est.ci_low:
                continue
            metric, cls = split_key(key)
            rows.append(ResultRow(corruption=corruption, severity=severity, seed=seed, metric=metric,
                                  cls=cls, value=est.value, ci_low=est.ci_low, ci_high=est.ci_high, **common))

    t0 = time.time()
    clean = draws_for(clean_dir)
    add(clean, "clean", 0, 0)
    print(f"clean: mAP={clean.point['map']:.4f} NDS={clean.point['nds']:.4f} ({time.time() - t0:.0f}s)")

    all_draws = {}
    for i, (key, d) in enumerate(sorted(run_dirs.items()), 1):
        t0 = time.time()
        dr = draws_for(d)
        all_draws[key] = dr
        add(dr, *key)
        print(f"[{i}/{len(run_dirs)}] {d.name}: mAP={dr.point['map']:.4f} NDS={dr.point['nds']:.4f} "
              f"({time.time() - t0:.0f}s)")

    for k in ROBUSTNESS_KEYS:
        s = RB.summarize(clean, all_draws, k)
        for (c, sev, seed), est in s.rr.items():
            rows.append(ResultRow(corruption=c, severity=sev, seed=seed, metric=f"rr_{k}", cls="all",
                                  value=est.value, ci_low=est.ci_low, ci_high=est.ci_high, **common))
        if s.mrr.value == s.mrr.value:
            rows.append(ResultRow(corruption=AGGREGATE, severity=0, seed=POOLED_SEED, metric=f"mrr_{k}", cls="all",
                                  value=s.mrr.value, ci_low=s.mrr.ci_low, ci_high=s.mrr.ci_high, **common))
        for c, est in s.failure.items():
            rows.append(ResultRow(corruption=c, severity=0, seed=POOLED_SEED, metric=f"failure_threshold_{k}",
                                  cls="all", value=est.value, ci_low=est.ci_low, ci_high=est.ci_high, **common))
            print(f"  failure threshold ({k}) {c}: " + ("never" if est.value == RB.NEVER else f"severity {est.value:.0f}"))
        print(f"  mRR ({k}) = {s.mrr.value:.3f} [{s.mrr.ci_low:.3f}, {s.mrr.ci_high:.3f}]")

    n = append_rows(args.results_csv, rows, replace=True)   # rerunning overwrites this model/hardware's rows
    print(f"wrote {n} rows to {args.results_csv}")


if __name__ == "__main__":
    main()
