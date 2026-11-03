"""Drop-in replacement for ``python tools/test.py`` that enables the SDC corruption plugin.

Run from the root of the BEVFusion checkout (arguments are forwarded unchanged):

    SDC_CORRUPTION=range_limit SDC_SEVERITY=3 \\
    torchpack dist-run -np 1 python /workspace/sdc/lidar/bevfusion/run_test.py \\
        configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml \\
        pretrained/lidar-only-det.pth --eval bbox --eval-options jsonfile_prefix=/results/run1

With no SDC_* variables set this reproduces the published number (clean run).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(ROOT, "common"), os.path.join(ROOT, "lidar")):
    if p not in sys.path:
        sys.path.insert(0, p)

import sdc_lidar.bevfusion_plugin  # noqa: E402,F401  (registers the corrupted loader)

if __name__ == "__main__":
    test_py = os.path.join(os.getcwd(), "tools", "test.py")
    if not os.path.exists(test_py):
        sys.exit("run this from the BEVFusion repo root (tools/test.py not found)")
    sys.path.insert(0, os.getcwd())
    sys.argv[0] = test_py
    runpy.run_path(test_py, run_name="__main__")
