import numpy as np
import pytest

from sdc_lidar.feature_stability import feature_stability, points_in_box

BOX = [10.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0]          # car-ish box at x=10


def cloud(rng, n=400):
    xyz = np.concatenate([rng.uniform([8, -1, -1], [12, 1, 1], (n, 3)),      # inside BOX
                          rng.uniform([30, -5, -1], [40, 5, 1], (n, 3))])    # elsewhere
    feat = rng.normal(size=(len(xyz), 16))
    return xyz, feat


def test_points_in_box_with_yaw():
    pts = np.array([[10.0, 0, 0], [11.9, 0, 0], [10, 0.9, 0], [10, 1.1, 0]])
    assert points_in_box(pts, BOX).tolist() == [True, True, True, False]
    rot = [10, 0, 0, 4, 2, 2, np.pi / 2]            # rotated 90 deg: long axis now along y
    assert points_in_box(np.array([[10, 1.9, 0], [11.9, 0, 0]]), rot).tolist() == [True, False]


def test_identical_clouds_give_cosine_one_and_full_coverage():
    xyz, f = cloud(np.random.default_rng(0))
    r = feature_stability(xyz, f, xyz, f, [BOX])
    assert r.mean_cosine == pytest.approx(1.0) and r.coverage == 1.0


def test_subset_keeps_cosine_one_for_surviving_points():
    xyz, f = cloud(np.random.default_rng(1))
    keep = np.random.default_rng(2).random(len(xyz)) > 0.5
    r = feature_stability(xyz, f, xyz[keep], f[keep], [BOX])
    # dropped points are matched to a neighbour with a different (random) feature -> < 1 but covered
    assert r.coverage == 1.0 and 0.0 < r.n_boxes and r.mean_cosine < 1.0


def test_uncovered_box_is_reported_not_scored_zero():
    xyz, f = cloud(np.random.default_rng(3))
    far = xyz[:, 0] > 20                                # corruption removed everything near the box
    r = feature_stability(xyz, f, xyz[far], f[far], [BOX])
    assert r.coverage == 0.0 and np.isnan(r.mean_cosine) and np.isnan(r.per_box[0])


def test_box_without_clean_points_is_ignored_in_denominator():
    xyz, f = cloud(np.random.default_rng(4))
    empty_box = [-50, 0, 0, 2, 2, 2, 0]
    r = feature_stability(xyz, f, xyz, f, [BOX, empty_box])
    assert r.n_boxes == 1 and r.coverage == 1.0


def test_unrelated_features_have_near_zero_cosine():
    rng = np.random.default_rng(5)
    xyz, f = cloud(rng, 2000)
    r = feature_stability(xyz, f, xyz, rng.normal(size=f.shape), [BOX])
    assert abs(r.mean_cosine) < 0.1
