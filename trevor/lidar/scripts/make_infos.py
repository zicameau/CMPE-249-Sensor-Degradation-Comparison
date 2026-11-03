"""Generate BEVFusion info pickles for a nuScenes version WITHOUT the GT-database step.

Why not ``tools/create_data.py``?  At the pinned BEVFusion commit it (a) writes
``<prefix>_infos_{train,val}_radar.pkl`` into a path relative to the *current directory* instead of the dataset root,
and (b) then tries to build the GT database from ``nuscenes_infos_train.pkl`` (which it did not write) and crashes.
The GT database is only needed for training augmentation, not for evaluation.  This script calls the same
converter function and puts the results where the configs expect them.

Run inside the Docker image:
    python /workspace/sdc/lidar/scripts/make_infos.py --root /dataset/nuscenes --version v1.0-mini  --out-prefix mini
    python /workspace/sdc/lidar/scripts/make_infos.py --root /dataset/nuscenes --version v1.0-trainval --out-prefix nuscenes

Writes <root>/<out-prefix>_infos_train.pkl and <root>/<out-prefix>_infos_val.pkl (``--val-only``: train is empty).  (Use out-prefix ``nuscenes`` to get
the file names the stock configs expect; any other prefix needs ``--cfg-options data.test.ann_file=...``.)
"""
import argparse
import os
import shutil
import sys
import tempfile

import mmcv

sys.path.insert(0, "/workspace/bevfusion")
sys.path.insert(0, "/workspace/bevfusion/tools")

from data_converter import nuscenes_converter as nc  # noqa: E402


def rebase(obj, old, new):
    """Recursively replace the ``old`` path prefix in every string (the converter writes absolute paths)."""
    if isinstance(obj, str):
        return new + obj[len(old):] if obj.startswith(old) else obj
    if isinstance(obj, dict):
        return {k: rebase(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rebase(v, old, new) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--version", default="v1.0-trainval", choices=["v1.0-trainval", "v1.0-mini"])
    ap.add_argument("--out-prefix", default="nuscenes")
    ap.add_argument("--max-sweeps", type=int, default=10)
    ap.add_argument("--path-prefix", default="./data/nuscenes/",
                    help="prefix stored in the pkl instead of --root (what tools/create_data.py writes when run from "
                         "the BEVFusion dir with data/nuscenes); keeps the pkls valid wherever the data is mounted")
    ap.add_argument("--val-only", action="store_true",
                    help="skip the train split (an empty train pkl is written): evaluation needs only val, and the "
                         "train infos cost hours on a slow (e.g. Windows bind-mounted) filesystem")
    a = ap.parse_args()
    if a.val_only:
        from nuscenes.utils import splits
        splits.train = []          # the converter reads splits.train / splits.val at call time

    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    os.chdir(tmp)  # the converter writes to ./<info_prefix>/...
    try:
        nc.create_nuscenes_infos(a.root.rstrip("/") + "/", "nuscenes", version=a.version, max_sweeps=a.max_sweeps)
        for split in ("train", "val"):
            src = os.path.join(tmp, "nuscenes", f"nuscenes_infos_{split}_radar.pkl")
            dst = os.path.join(a.root, f"{a.out_prefix}_infos_{split}.pkl")
            data = rebase(mmcv.load(src), a.root.rstrip("/") + "/", a.path_prefix)
            mmcv.dump(data, dst)
            os.remove(src)
            print("wrote", dst, os.path.getsize(dst) // 2**20, "MB")
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
