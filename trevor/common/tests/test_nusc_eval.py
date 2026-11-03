"""Validate sdc_common.nusc_eval against the official nuScenes devkit on synthetic data.

Skipped automatically when ``nuscenes-devkit`` is not installed (it needs an old numpy stack, so use
a venv: ``pip install nuscenes-devkit scipy pytest``).
"""
import numpy as np
import pytest

pytest.importorskip("nuscenes")
from pyquaternion import Quaternion  # noqa: E402
from nuscenes.eval.common.data_classes import EvalBoxes  # noqa: E402
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp  # noqa: E402
from nuscenes.eval.detection.config import config_factory  # noqa: E402
from nuscenes.eval.detection.constants import TP_METRICS, ATTRIBUTE_NAMES  # noqa: E402
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionMetrics, DetectionMetricDataList  # noqa: E402

from sdc_common import nusc_eval as NE  # noqa: E402
from sdc_common import stats  # noqa: E402

CFG = config_factory("detection_cvpr_2019")
ATTRS = {"car": ["vehicle.moving", "vehicle.parked", "vehicle.stopped"],
         "truck": ["vehicle.moving", "vehicle.parked"], "bus": ["vehicle.moving"],
         "trailer": ["vehicle.parked"], "construction_vehicle": ["vehicle.parked"],
         "pedestrian": ["pedestrian.moving", "pedestrian.standing"],
         "motorcycle": ["cycle.with_rider"], "bicycle": ["cycle.without_rider"],
         "traffic_cone": [""], "barrier": [""]}
assert all(a in ATTRIBUTE_NAMES or a == "" for v in ATTRS.values() for a in v)


def box(tok, name, xy, size, yaw, vel, score=-1.0, attr=""):
    q = Quaternion(axis=[0, 0, 1], angle=yaw)
    return DetectionBox(sample_token=tok, translation=(float(xy[0]), float(xy[1]), 0.0),
                        size=tuple(float(s) for s in size), rotation=tuple(float(v) for v in q.elements),
                        velocity=(float(vel[0]), float(vel[1])), detection_name=name,
                        detection_score=float(score), attribute_name=attr)


def make_data(seed=0, n_scenes=14, per_scene=3, tie_scores=True):
    rng = np.random.default_rng(seed)
    gt, pred, scene_of = EvalBoxes(), EvalBoxes(), {}
    names = list(NE.CLASS_NAMES)
    for s in range(n_scenes):
        for f in range(per_scene):
            tok = f"s{s}f{f}"
            scene_of[tok] = s
            g, p = [], []
            for _ in range(rng.integers(3, 12)):
                name = names[rng.integers(0, 10)]
                xy = rng.uniform(-25, 25, 2)
                size = rng.uniform(0.5, 4, 3)
                yaw = rng.uniform(-np.pi, np.pi)
                vel = rng.normal(0, 3, 2)
                attr = rng.choice(ATTRS[name])
                if rng.random() < 0.05:
                    attr = ""            # GT with a missing attribute -> NaN attr error
                g.append(box(tok, name, xy, size, yaw, vel, attr=attr))
                if rng.random() < 0.8:   # detected, with noise (some beyond the 4 m threshold)
                    nm = name if rng.random() < 0.9 else names[rng.integers(0, 10)]
                    sc = rng.uniform(0.3, 1.0)
                    p.append(box(tok, nm, xy + rng.normal(0, 1.0, 2), size * rng.uniform(0.7, 1.3, 3),
                                 yaw + rng.normal(0, 0.4), vel + rng.normal(0, 1, 2),
                                 np.round(sc, 1) if tie_scores else sc,
                                 rng.choice(ATTRS[nm])))
            for _ in range(rng.integers(2, 8)):    # false positives
                nm = names[rng.integers(0, 10)]
                sc = rng.uniform(0.0, 0.7)
                p.append(box(tok, nm, rng.uniform(-25, 25, 2), rng.uniform(0.5, 4, 3), rng.uniform(-3, 3),
                             rng.normal(0, 2, 2), np.round(sc, 1) if tie_scores else sc,
                             rng.choice(ATTRS[nm])))
            gt.add_boxes(tok, g)
            pred.add_boxes(tok, p)
    return gt, pred, scene_of


def devkit_metrics(gt, pred):
    mdl = DetectionMetricDataList()
    for c in CFG.class_names:
        for th in CFG.dist_ths:
            mdl.set(c, th, accumulate(gt, pred, c, CFG.dist_fcn_callable, th))
    m = DetectionMetrics(CFG)
    for c in CFG.class_names:
        for th in CFG.dist_ths:
            m.add_label_ap(c, th, calc_ap(mdl[(c, th)], CFG.min_recall, CFG.min_precision))
        for tp_name in TP_METRICS:
            md = mdl[(c, CFG.dist_th_tp)]
            if c == "traffic_cone" and tp_name in ("attr_err", "vel_err", "orient_err"):
                v = np.nan
            elif c == "barrier" and tp_name in ("attr_err", "vel_err"):
                v = np.nan
            else:
                v = calc_tp(md, CFG.min_recall, tp_name)
            m.add_label_tp(c, tp_name, v)
    return m


def my_table(gt, pred, scene_of):
    return NE.build_match_table(NE.from_eval_boxes(gt), NE.from_eval_boxes(pred), scene_of)


@pytest.mark.parametrize("seed,ties", [(0, True), (1, True), (2, False)])
def test_matches_devkit_exactly(seed, ties):
    gt, pred, scene_of = make_data(seed, tie_scores=ties)
    ref = devkit_metrics(gt, pred)
    mine = NE.evaluate(my_table(gt, pred, scene_of))
    assert mine.map == pytest.approx(ref.mean_ap, abs=1e-9)
    assert mine.nds == pytest.approx(ref.nd_score, abs=1e-9)
    for c in NE.CLASS_NAMES:
        assert mine.ap[c] == pytest.approx(ref.mean_dist_aps[c], abs=1e-9)
        for m in TP_METRICS:
            a, b = mine.label_tp[c][m], ref.get_label_tp(c, m)
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, abs=1e-9), (c, m)
    for m, v in ref.tp_errors.items():
        assert mine.tp_errors[m] == pytest.approx(v, abs=1e-9)


def test_weights_approximate_physically_duplicated_scenes():
    gt, pred, scene_of = make_data(3, n_scenes=12)
    table = my_table(gt, pred, scene_of)
    rng = np.random.default_rng(7)
    for _ in range(3):
        w = rng.multinomial(12, np.full(12, 1 / 12)).astype(float)
        # physically duplicate scenes: copy every sample of scene s, w[s] times
        gt2, pred2 = EvalBoxes(), EvalBoxes()
        for tok in pred.sample_tokens:
            s = scene_of[tok]
            for k in range(int(w[s])):
                nt = f"{tok}_d{k}"
                gt2.add_boxes(nt, [box(nt, b.detection_name, b.translation, b.size,
                                       Quaternion(b.rotation).yaw_pitch_roll[0], b.velocity,
                                       attr=b.attribute_name) for b in gt[tok]])
                pred2.add_boxes(nt, [box(nt, b.detection_name, b.translation, b.size,
                                         Quaternion(b.rotation).yaw_pitch_roll[0], b.velocity,
                                         b.detection_score, b.attribute_name) for b in pred[tok]])
        ref = devkit_metrics(gt2, pred2)
        mine = NE.evaluate(table, w)
        assert mine.map == pytest.approx(ref.mean_ap, abs=5e-3)
        assert mine.nds == pytest.approx(ref.nd_score, abs=5e-3)


def test_zero_weight_scenes_equal_dropping_them():
    gt, pred, scene_of = make_data(4, n_scenes=10)
    w = np.ones(10)
    w[[2, 5]] = 0
    drop = {t for t in pred.sample_tokens if scene_of[t] in (2, 5)}
    gt2, pred2 = EvalBoxes(), EvalBoxes()
    for t in pred.sample_tokens:
        if t not in drop:
            gt2.add_boxes(t, gt[t]); pred2.add_boxes(t, pred[t])
    ref = devkit_metrics(gt2, pred2)
    mine = NE.evaluate(my_table(gt, pred, scene_of), w)
    assert mine.map == pytest.approx(ref.mean_ap, abs=1e-9)
    assert mine.nds == pytest.approx(ref.nd_score, abs=1e-9)


def test_table_roundtrip_and_bootstrap(tmp_path):
    gt, pred, scene_of = make_data(5, n_scenes=10)
    t = my_table(gt, pred, scene_of)
    NE.save_table(t, tmp_path / "t.npz")
    t2 = NE.load_table(tmp_path / "t.npz")
    assert NE.evaluate(t2).flat() == NE.evaluate(t).flat()
    d = NE.bootstrap_draws(t, n_boot=60, seed=1)
    est = d.estimate("map")
    assert est.value == pytest.approx(NE.evaluate(t).map)
    assert est.ci_low < est.value < est.ci_high or est.ci_low <= est.value <= est.ci_high
    assert set(d.boot) == set(d.point) and all(len(v) == 60 for v in d.boot.values())


def test_rr_and_paired_diff_use_same_resamples():
    gt, pred, scene_of = make_data(6, n_scenes=10)
    t = my_table(gt, pred, scene_of)
    W = stats.bootstrap_weights(10, 40, seed=3)
    a = NE.bootstrap_draws(t, weights=W)
    est = NE.rr_estimate(a, a, "map")
    assert est.value == pytest.approx(1.0) and est.ci_low == pytest.approx(1.0) and est.ci_high == pytest.approx(1.0)
    diff = NE.diff_estimate(a, a, "nds")
    assert diff.value == 0 and diff.ci_low == 0 and diff.ci_high == 0 and not diff.excludes_zero()


def test_no_predictions_for_a_class_matches_devkit():
    gt, pred, scene_of = make_data(8)
    for tok in pred.sample_tokens:
        pred.boxes[tok] = [b for b in pred.boxes[tok] if b.detection_name != "bus"]
    ref = devkit_metrics(gt, pred)
    mine = NE.evaluate(my_table(gt, pred, scene_of))
    assert mine.map == pytest.approx(ref.mean_ap, abs=1e-9)
    assert mine.nds == pytest.approx(ref.nd_score, abs=1e-9)
