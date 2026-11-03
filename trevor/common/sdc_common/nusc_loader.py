"""Load real nuScenes GT + a BEVFusion ``results_nusc.json`` into ``nusc_eval`` structures.

Uses the devkit's own loaders and filters in exactly the order ``DetectionEval.__init__`` does
(load -> add_center_dist -> filter_eval_boxes), so the boxes entering the match table are the boxes the
official evaluation scores. Needs ``nuscenes-devkit`` (present in the BEVFusion image).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from . import nusc_eval as NE

EVAL_SET = {"v1.0-trainval": "val", "v1.0-mini": "mini_val"}


@dataclass
class GroundTruth:
    boxes: Dict[str, NE.Boxes]        # sample token -> GT boxes (devkit-filtered)
    scene_of: Dict[str, int]          # sample token -> scene index (scenes sorted by name)
    scene_names: Tuple[str, ...]
    nusc: object
    cfg: object


def load_ground_truth(nusc, version: str = "v1.0-trainval", cfg_name: str = "detection_cvpr_2019") -> GroundTruth:
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes, load_gt
    from nuscenes.eval.detection.data_classes import DetectionBox

    cfg = config_factory(cfg_name)
    gt = load_gt(nusc, EVAL_SET[version], DetectionBox, verbose=False)
    gt = filter_eval_boxes(nusc, add_center_dist(nusc, gt), cfg.class_range, verbose=False)

    names = sorted({nusc.get("scene", nusc.get("sample", t)["scene_token"])["name"] for t in gt.sample_tokens})
    idx = {n: i for i, n in enumerate(names)}
    scene_of = {t: idx[nusc.get("scene", nusc.get("sample", t)["scene_token"])["name"]]
                for t in gt.sample_tokens}
    return GroundTruth(NE.from_eval_boxes(gt), scene_of, tuple(names), nusc, cfg)


def build_table_for_run(gt: GroundTruth, result_json: str | Path) -> NE.MatchTable:
    """Match one run's predictions (``results_nusc.json``) against the GT."""
    from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes, load_prediction
    from nuscenes.eval.detection.data_classes import DetectionBox

    pred, _ = load_prediction(str(result_json), gt.cfg.max_boxes_per_sample, DetectionBox, verbose=False)
    if set(pred.sample_tokens) != set(gt.boxes):
        raise ValueError(f"{result_json}: samples do not match the GT split "
                         f"({len(set(pred.sample_tokens))} vs {len(gt.boxes)}); was the run on the full split?")
    pred = filter_eval_boxes(gt.nusc, add_center_dist(gt.nusc, pred), gt.cfg.class_range, verbose=False)
    return NE.build_match_table(gt.boxes, NE.from_eval_boxes(pred), gt.scene_of,
                                n_scenes=len(gt.scene_names), scene_names=gt.scene_names)
