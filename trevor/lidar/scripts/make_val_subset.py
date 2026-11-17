#!/usr/bin/env python
"""Write a val-scenes-only copy of the nuScenes v1.0-trainval metadata tables.

Why: the devkit loads every table of v1.0-trainval (sample_data.json alone is 1.3 GB -> several GB of Python objects),
which gets killed on an 8 GB Docker/WSL VM - both for ``make_infos.py`` and for the evaluation (``NuScenesEval``) at the
end of every test run. The LiDAR study only evaluates the 150 val scenes (6,019 samples), so those are the only rows
we keep. Row content is untouched, so tokens, sweeps links and annotations are exactly the originals.

    python make_val_subset.py --src /dataset/nuscenes/v1.0-trainval --dst /dataset/nuscenes_val/v1.0-trainval

Tables are processed one at a time to keep the peak memory low.
"""
from __future__ import annotations

import argparse
import gc
import json
import os


def load(src, name):
    with open(os.path.join(src, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def dump(dst, name, rows):
    with open(os.path.join(dst, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(rows, f)
    print(f"  {name}: {len(rows)} rows", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)

    from nuscenes.utils.splits import val as val_names
    val_names = set(val_names)

    scenes = [s for s in load(a.src, "scene") if s["name"] in val_names]
    scene_tokens = {s["token"] for s in scenes}
    dump(a.dst, "scene", scenes)

    samples = [s for s in load(a.src, "sample") if s["scene_token"] in scene_tokens]
    sample_tokens = {s["token"] for s in samples}
    dump(a.dst, "sample", samples)

    sd = [d for d in load(a.src, "sample_data") if d["sample_token"] in sample_tokens]
    ego_tokens = {d["ego_pose_token"] for d in sd}
    calib_tokens = {d["calibrated_sensor_token"] for d in sd}
    dump(a.dst, "sample_data", sd)
    del sd
    gc.collect()

    ann = [x for x in load(a.src, "sample_annotation") if x["sample_token"] in sample_tokens]
    inst_tokens = {x["instance_token"] for x in ann}
    dump(a.dst, "sample_annotation", ann)
    del ann
    gc.collect()

    dump(a.dst, "instance", [x for x in load(a.src, "instance") if x["token"] in inst_tokens])
    dump(a.dst, "ego_pose", [x for x in load(a.src, "ego_pose") if x["token"] in ego_tokens])
    dump(a.dst, "calibrated_sensor", [x for x in load(a.src, "calibrated_sensor") if x["token"] in calib_tokens])
    for name in ("attribute", "category", "log", "map", "sensor", "visibility"):
        dump(a.dst, name, load(a.src, name))
    print(f"scenes={len(scenes)} samples={len(samples)} (expect 150 / 6019)")


if __name__ == "__main__":
    main()
