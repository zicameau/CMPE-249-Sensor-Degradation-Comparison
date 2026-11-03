import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from distance_recall import matched_gt  # noqa: E402


def test_each_gt_matched_at_most_once_and_highest_score_wins():
    gt = np.array([[0.0, 0.0], [10.0, 0.0]])
    pred = np.array([[0.5, 0.0], [0.4, 0.0], [10.2, 0.0]])
    score = np.array([0.5, 0.9, 0.1])
    hit = matched_gt(gt, pred, score, th=2.0)
    assert hit.tolist() == [True, True]          # two preds near GT 0 -> only one GT consumed, third pred takes GT 1


def test_threshold_is_strict_and_empty_inputs():
    gt = np.array([[0.0, 0.0]])
    assert matched_gt(gt, np.array([[2.0, 0.0]]), np.array([1.0]), th=2.0).tolist() == [False]
    assert matched_gt(gt, np.zeros((0, 2)), np.zeros(0), th=2.0).tolist() == [False]
    assert matched_gt(np.zeros((0, 2)), np.array([[0.0, 0.0]]), np.array([1.0]), th=2.0).size == 0
