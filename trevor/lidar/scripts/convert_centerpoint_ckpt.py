#!/usr/bin/env python
"""Convert an mmdet3d v0.x CenterPoint checkpoint (0.075 m voxels, SECOND+SECFPN) to BEVFusion's module layout.

Source:  centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus (mmdet3d v0.1.0 model zoo, val mAP 57.34 / NDS 65.23)
Target:  configs/nuscenes/det/centerhead/secfpn/lidar/voxelnet_0p075.yaml  (see lidar/configs/centerpoint_voxelnet_0p075.yaml)

The two code bases have the same 504 parameters with the same shapes except for axis order:

* prefixes:  pts_middle_encoder -> encoders.lidar.backbone,  pts_backbone -> decoder.backbone,
             pts_neck -> decoder.neck,  pts_bbox_head -> heads.object
* sparse 3-D kernels are stored (kz, ky, kx, in, out) in v0.x (coords are z,y,x) and (kx, ky, kz, in, out) in BEVFusion
  (coords are x,y,z)  ->  swap axes 0 and 2.   (conv_out: (3,1,1,..) -> (1,1,3,..), a visible sanity check.)
* BEVFusion's BEV map is (rows=x, cols=y) whereas v0.x's is (rows=y, cols=x)  ->  every 2-D conv / deconv kernel gets its two
  spatial axes swapped, which makes the transposed network exactly equivalent.  ``--no-transpose-2d`` disables this for
  diagnosis.

Usage (inside the sdc-lidar image, from /workspace/bevfusion):
    python /workspace/sdc/lidar/scripts/convert_centerpoint_ckpt.py \
        --src pretrained/centerpoint_0075voxel.pth --dst pretrained/centerpoint_0075voxel_bevfusion.pth
"""
from __future__ import annotations

import argparse

import torch

PREFIX = {
    "pts_middle_encoder.": "encoders.lidar.backbone.",
    "pts_backbone.": "decoder.backbone.",
    "pts_neck.": "decoder.neck.",
    "pts_bbox_head.": "heads.object.",
}


def convert_state_dict(sd: dict, transpose_2d: bool = True, swap_3d: bool = True) -> dict:
    out = {}
    for k, v in sd.items():
        for old, new in PREFIX.items():
            if k.startswith(old):
                k = new + k[len(old):]
                break
        else:
            raise KeyError(f"unexpected key {k}")
        if v.ndim == 5 and swap_3d:
            v = v.permute(2, 1, 0, 3, 4).contiguous()
        elif v.ndim == 4 and transpose_2d:
            v = v.transpose(2, 3).contiguous()
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--no-transpose-2d", action="store_true")
    ap.add_argument("--no-swap-3d", action="store_true")
    a = ap.parse_args()
    ck = torch.load(a.src, map_location="cpu")
    new = convert_state_dict(ck["state_dict"], not a.no_transpose_2d, not a.no_swap_3d)
    meta = {"CLASSES": ck.get("meta", {}).get("CLASSES"), "converted_from": a.src}
    torch.save({"meta": meta, "state_dict": new}, a.dst)
    print(f"wrote {a.dst}: {len(new)} tensors")


if __name__ == "__main__":
    main()
