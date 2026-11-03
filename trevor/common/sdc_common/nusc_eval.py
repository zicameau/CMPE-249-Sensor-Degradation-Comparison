"""Fast, bootstrappable nuScenes detection metrics (mAP, NDS, 5 TP errors, per-class AP).

Why this exists
---------------
The proposal needs 95% CIs from 1,000 scene-level bootstrap resamples. Running the official devkit
1,000 times per evaluation is not feasible. The devkit's matching, however, is *sample-local*: a
prediction is matched only against GT boxes of its own sample, and the "taken" bookkeeping is
per (sample, gt). So whether each prediction is a TP/FP (and its TP errors) does not depend on which
other samples are in the evaluation. We therefore

  1. match once  -> ``build_match_table`` (per class, per distance threshold: TP flag per prediction,
     plus the five TP errors at the 2 m threshold), and
  2. evaluate any scene resample cheaply by *weighting* predictions with the multiplicity of their
     scene -> ``evaluate(table, w)``.

``evaluate(table)`` with unit weights reproduces the devkit (verified against the devkit's own
``accumulate`` / ``calc_ap`` / ``calc_tp`` / ``DetectionMetrics`` in ``tests/test_nusc_eval.py``).

Resampling approximation
------------------------
A scene drawn k times contributes its predictions with weight k, as one jump of k in the cumulative
TP/FP counts instead of k separate steps. Linear interpolation of the PR curve across that jump is the
only difference from physically duplicating the scenes (tested: < 1e-3 on mAP, far below CI widths).

Conventions follow the devkit exactly: detection_cvpr_2019 config, center-distance matching with strict
``<`` thresholds, ties in confidence broken by prediction order, 101-point recall interpolation,
min_recall = min_precision = 0.1, TP errors at 2 m, NDS = (5 mAP + sum(1 - min(1, TP err))) / 10.
This module needs only numpy; building ``Boxes`` from devkit ``EvalBoxes`` is in ``from_eval_boxes``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

from .stats import Estimate, N_BOOT, bootstrap_weights

CLASS_NAMES = ("car", "truck", "bus", "trailer", "construction_vehicle",
               "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier")
DIST_THS = (0.5, 1.0, 2.0, 4.0)
DIST_TH_TP = 2.0
MIN_RECALL = 0.1
MIN_PRECISION = 0.1
MEAN_AP_WEIGHT = 5
TP_METRICS = ("trans_err", "scale_err", "orient_err", "vel_err", "attr_err")
# devkit: these TP errors are undefined for these classes and ignored in the class mean
NAN_TP = {"traffic_cone": {"attr_err", "vel_err", "orient_err"}, "barrier": {"attr_err", "vel_err"}}
NELEM = 101

_ATTR_IDS: Dict[str, int] = {}


def attr_id(name: str) -> int:
    """'' (no attribute) -> -1, otherwise a stable integer id."""
    if not name:
        return -1
    return _ATTR_IDS.setdefault(name, len(_ATTR_IDS))


@dataclass
class Boxes:
    """Boxes of ONE sample as arrays (all length n). ``size`` is the devkit order (w, l, h)."""
    xy: np.ndarray
    size: np.ndarray
    yaw: np.ndarray
    vel: np.ndarray
    cls: np.ndarray             # int index into CLASS_NAMES
    attr: np.ndarray            # int, -1 = none
    score: np.ndarray           # GT: ignored

    def __len__(self):
        return len(self.cls)


def quat_yaw(q) -> float:
    """Yaw of a (w, x, y, z) quaternion = angle of the rotated x-axis in the xy plane (devkit's quaternion_yaw)."""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def from_eval_boxes(eval_boxes, class_names: Sequence[str] = CLASS_NAMES) -> Dict[str, Boxes]:
    """devkit ``EvalBoxes`` (already loaded + range/num_pts filtered) -> {sample_token: Boxes}.

    Dict order = ``eval_boxes.sample_tokens`` order, which defines the devkit's tie-breaking order.
    """
    cidx = {c: i for i, c in enumerate(class_names)}
    out = {}
    for tok in eval_boxes.sample_tokens:
        bl = [b for b in eval_boxes[tok] if b.detection_name in cidx]
        n = len(bl)
        out[tok] = Boxes(
            xy=np.array([b.translation[:2] for b in bl], dtype=np.float64).reshape(n, 2),
            size=np.array([b.size for b in bl], dtype=np.float64).reshape(n, 3),
            yaw=np.array([quat_yaw(b.rotation) for b in bl], dtype=np.float64),
            vel=np.array([b.velocity for b in bl], dtype=np.float64).reshape(n, 2),
            cls=np.array([cidx[b.detection_name] for b in bl], dtype=np.int64),
            attr=np.array([attr_id(b.attribute_name) for b in bl], dtype=np.int64),
            score=np.array([b.detection_score for b in bl], dtype=np.float64),
        )
    return out


# ---------------------------------------------------------------------------
# Step 1: matching (once per run)
# ---------------------------------------------------------------------------
def _angle_diff(x, y, period):
    d = (x - y + period / 2) % period - period / 2
    return np.where(d > np.pi, d - 2 * np.pi, d)


def _tp_errors(g: Boxes, gi: np.ndarray, p: Boxes, pi: np.ndarray, period: float) -> np.ndarray:
    """(m, 5) errors in TP_METRICS order for matched pairs (gt idx gi, pred idx pi)."""
    trans = np.linalg.norm(p.xy[pi] - g.xy[gi], axis=1)
    vel = np.linalg.norm(p.vel[pi] - g.vel[gi], axis=1)
    a, b = g.size[gi], p.size[pi]
    inter = np.prod(np.minimum(a, b), axis=1)
    scale = 1.0 - inter / (np.prod(a, axis=1) + np.prod(b, axis=1) - inter)
    orient = np.abs(_angle_diff(g.yaw[gi], p.yaw[pi], period))
    ga = g.attr[gi]
    attr = np.where(ga < 0, np.nan, 1.0 - (ga == p.attr[pi]).astype(float))
    return np.stack([trans, scale, orient, vel, attr], axis=1)


def _greedy_match(dist: np.ndarray, order: np.ndarray, th: float) -> np.ndarray:
    """dist (P, G); order = pred indices by descending confidence. Returns matched gt idx per pred (-1 = FP)."""
    P, G = dist.shape
    match = np.full(P, -1, dtype=np.int64)
    if G == 0:
        return match
    possible = dist.min(axis=1) < th
    taken = np.zeros(G, dtype=bool)
    for p in order:
        if not possible[p]:
            continue
        row = np.where(taken, np.inf, dist[p])
        j = int(np.argmin(row))
        if row[j] < th:
            match[p] = j
            taken[j] = True
    return match


@dataclass
class ClassData:
    conf: np.ndarray            # (n,) sorted by confidence desc (devkit tie order)
    scene: np.ndarray           # (n,) scene index of each prediction
    tp: np.ndarray              # (n_th, n) bool
    err: np.ndarray             # (n, 5) TP errors at DIST_TH_TP (NaN where unmatched / undefined)
    npos_by_scene: np.ndarray   # (n_scenes,) GT count of this class per scene


@dataclass
class MatchTable:
    n_scenes: int
    classes: Dict[str, ClassData]
    class_names: Sequence[str] = CLASS_NAMES
    dist_ths: Sequence[float] = DIST_THS
    # Scene names in index order. Tables compared with the same bootstrap weights (paired differences,
    # RR against the clean run) MUST have identical scene_names - see ``check_paired``.
    scene_names: Sequence[str] = ()


def check_paired(*tables: MatchTable) -> None:
    first = tables[0]
    for t in tables[1:]:
        if t.n_scenes != first.n_scenes or (first.scene_names and t.scene_names
                                            and tuple(t.scene_names) != tuple(first.scene_names)):
            raise ValueError("tables cover different scenes; paired bootstrap would be invalid")


def build_match_table(gt: Mapping[str, Boxes], pred: Mapping[str, Boxes], scene_of: Mapping[str, int],
                      n_scenes: int | None = None, class_names: Sequence[str] = CLASS_NAMES,
                      dist_ths: Sequence[float] = DIST_THS, dist_th_tp: float = DIST_TH_TP,
                      scene_names: Sequence[str] = ()) -> MatchTable:
    """Match every prediction to GT (devkit rules). ``gt`` and ``pred`` need identical sample tokens."""
    if set(gt) != set(pred):
        raise ValueError("gt and pred must cover the same samples")
    if n_scenes is None:
        n_scenes = max(scene_of.values()) + 1
    th_tp_idx = list(dist_ths).index(dist_th_tp)
    n_th = len(dist_ths)

    acc = {c: dict(conf=[], scene=[], tp=[], err=[]) for c in range(len(class_names))}
    npos = {c: np.zeros(n_scenes) for c in range(len(class_names))}

    for tok, P in pred.items():
        G = gt[tok]
        s = scene_of[tok]
        for c, name in enumerate(class_names):
            gi = np.flatnonzero(G.cls == c)
            npos[c][s] += len(gi)
            pi = np.flatnonzero(P.cls == c)
            if len(pi) == 0:
                continue
            score = P.score[pi]
            order = np.lexsort((np.arange(len(pi)), score))[::-1]   # devkit: sorted by (score, index), reversed
            dist = (np.linalg.norm(P.xy[pi][:, None, :] - G.xy[gi][None, :, :], axis=2)
                    if len(gi) else np.zeros((len(pi), 0)))
            tp = np.zeros((n_th, len(pi)), dtype=bool)
            err = np.full((len(pi), len(TP_METRICS)), np.nan)
            period = np.pi if name == "barrier" else 2 * np.pi
            for k, th in enumerate(dist_ths):
                m = _greedy_match(dist, order, th)
                tp[k] = m >= 0
                if k == th_tp_idx and tp[k].any():
                    sel = np.flatnonzero(tp[k])
                    err[sel] = _tp_errors(G, gi[m[sel]], P, pi[sel], period)
            a = acc[c]
            a["conf"].append(score); a["scene"].append(np.full(len(pi), s)); a["tp"].append(tp); a["err"].append(err)

    classes = {}
    for c, name in enumerate(class_names):
        a = acc[c]
        if a["conf"]:
            conf = np.concatenate(a["conf"]); scene = np.concatenate(a["scene"])
            tp = np.concatenate(a["tp"], axis=1); err = np.concatenate(a["err"], axis=0)
        else:
            conf = np.zeros(0); scene = np.zeros(0, dtype=np.int64)
            tp = np.zeros((n_th, 0), dtype=bool); err = np.zeros((0, len(TP_METRICS)))
        order = np.lexsort((np.arange(len(conf)), conf))[::-1]       # global devkit order
        classes[name] = ClassData(conf[order], scene[order].astype(np.int64), tp[:, order], err[order], npos[c])
    return MatchTable(n_scenes, classes, tuple(class_names), tuple(dist_ths), tuple(scene_names))


def save_table(table: MatchTable, path) -> None:
    flat = {"n_scenes": np.array(table.n_scenes), "class_names": np.array(table.class_names),
            "dist_ths": np.array(table.dist_ths), "scene_names": np.array(list(table.scene_names), dtype=str)}
    for name, d in table.classes.items():
        for f in ("conf", "scene", "tp", "err", "npos_by_scene"):
            flat[f"{name}/{f}"] = getattr(d, f)
    np.savez_compressed(path, **flat)


def load_table(path) -> MatchTable:
    z = np.load(path, allow_pickle=False)
    names = tuple(str(x) for x in z["class_names"])
    classes = {n: ClassData(*(z[f"{n}/{f}"] for f in ("conf", "scene", "tp", "err", "npos_by_scene")))
               for n in names}
    return MatchTable(int(z["n_scenes"]), classes, names, tuple(float(x) for x in z["dist_ths"]),
                      tuple(str(x) for x in z["scene_names"]))


# ---------------------------------------------------------------------------
# Step 2: weighted evaluation (once per resample)
# ---------------------------------------------------------------------------
def _pr_curve(conf, tp, w, npos):
    """Interpolated precision/conf at 101 recall points, from weighted cumulative TP/FP.

    Returns (prec_i, conf_i, w_match_conf, keep, matched) or None for the devkit's 'no predictions' case.
    """
    keep = w > 0
    t = tp[keep]
    if npos <= 0 or not t.any():
        return None
    ww, c = w[keep], conf[keep]
    tpc = np.cumsum(ww * t)
    fpc = np.cumsum(ww * ~t)
    prec = tpc / (tpc + fpc)
    rec = tpc / npos
    rec_i = np.linspace(0, 1, NELEM)
    prec_i = np.interp(rec_i, rec, prec, right=0)
    conf_i = np.interp(rec_i, rec, c, right=0)
    return prec_i, conf_i, keep, t


def _ap(prec_i):
    p = prec_i[round(100 * MIN_RECALL) + 1:] - MIN_PRECISION
    p[p < 0] = 0
    return float(np.mean(p)) / (1.0 - MIN_PRECISION)


def _tp_score(conf_i, conf_m, err_m, w_m):
    """Devkit calc_tp for one error type, with weighted nan-aware cummean."""
    valid = ~np.isnan(err_m)
    if not valid.any():
        tmp = np.ones(len(err_m))
    else:
        s = np.cumsum(np.where(valid, w_m * err_m, 0.0))
        n = np.cumsum(w_m * valid)
        tmp = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
    res = np.interp(conf_i[::-1], conf_m[::-1], tmp[::-1])[::-1]
    nz = np.nonzero(conf_i)[0]
    last = nz[-1] if len(nz) else 0
    first = round(100 * MIN_RECALL) + 1
    return 1.0 if last < first else float(np.mean(res[first:last + 1]))


@dataclass
class NuscMetrics:
    map: float
    nds: float
    tp_errors: Dict[str, float]            # class-mean per TP metric (devkit "tp_errors")
    ap: Dict[str, float]                   # per class, mean over distance thresholds
    label_tp: Dict[str, Dict[str, float]]  # per class per TP metric (NaN where undefined)

    def flat(self) -> Dict[str, float]:
        """Flat dict keyed like the shared CSV: map, nds, mate/mase/maoe/mave/maae, ap_nusc/<class>."""
        names = dict(zip(TP_METRICS, ("mate", "mase", "maoe", "mave", "maae")))
        out = {"map": self.map, "nds": self.nds}
        out.update({names[k]: v for k, v in self.tp_errors.items()})
        out.update({f"ap_nusc/{c}": v for c, v in self.ap.items()})
        return out


def evaluate(table: MatchTable, w: np.ndarray | None = None) -> NuscMetrics:
    """Metrics for scene weights ``w`` (length n_scenes). ``None`` = every scene once = the devkit's numbers."""
    w = np.ones(table.n_scenes) if w is None else np.asarray(w, dtype=float)
    ap, label_tp = {}, {}
    k_tp = list(table.dist_ths).index(DIST_TH_TP)
    for name in table.class_names:
        d = table.classes[name]
        wp = w[d.scene]
        npos = float(w @ d.npos_by_scene)
        aps = []
        label_tp[name] = {m: float("nan") if m in NAN_TP.get(name, ()) else 1.0 for m in TP_METRICS}
        for k, th in enumerate(table.dist_ths):
            r = _pr_curve(d.conf, d.tp[k], wp, npos)
            if r is None:
                aps.append(0.0)
                continue
            prec_i, conf_i, keep, t = r
            aps.append(_ap(prec_i))
            if k == k_tp:
                conf_m, w_m = d.conf[keep][t], wp[keep][t]
                for j, m in enumerate(TP_METRICS):
                    if m not in NAN_TP.get(name, ()):
                        label_tp[name][m] = _tp_score(conf_i, conf_m, d.err[keep][t][:, j], w_m)
        ap[name] = float(np.mean(aps))
    mean_ap = float(np.mean(list(ap.values())))
    tp_errors = {m: float(np.nanmean([label_tp[c][m] for c in table.class_names])) for m in TP_METRICS}
    scores = [max(0.0, 1.0 - tp_errors[m]) for m in TP_METRICS]
    nds = (MEAN_AP_WEIGHT * mean_ap + sum(scores)) / (MEAN_AP_WEIGHT + len(TP_METRICS))
    return NuscMetrics(mean_ap, nds, tp_errors, ap, label_tp)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
@dataclass
class Draws:
    """Point estimate + bootstrap draws for every flat metric key of one run."""
    point: Dict[str, float]
    boot: Dict[str, np.ndarray]            # key -> (n_boot,)

    def estimate(self, key: str) -> Estimate:
        b = self.boot[key]
        b = b[np.isfinite(b)]
        lo, hi = (np.percentile(b, 2.5), np.percentile(b, 97.5)) if b.size else (float("nan"),) * 2
        return Estimate(self.point[key], float(lo), float(hi))


def bootstrap_draws(table: MatchTable, weights: np.ndarray | None = None, n_boot: int = N_BOOT,
                    seed: int = 12345) -> Draws:
    """Evaluate every scene resample. Pass the *same* ``weights`` for runs you want paired."""
    W = bootstrap_weights(table.n_scenes, n_boot, seed) if weights is None else weights
    point = evaluate(table).flat()
    rows = [evaluate(table, w).flat() for w in W]
    return Draws(point, {k: np.array([r[k] for r in rows]) for k in point})


def diff_estimate(a: Draws, b: Draws, key: str) -> Estimate:
    """Paired difference a - b (same resamples required)."""
    d = a.boot[key] - b.boot[key]
    return Estimate(a.point[key] - b.point[key], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def rr_estimate(clean: Draws, corrupted: Draws, key: str) -> Estimate:
    """Resilience rate corrupted/clean with the CI from the same resamples (needs the same weights)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = corrupted.boot[key] / clean.boot[key]
    r = r[np.isfinite(r)]
    point = corrupted.point[key] / clean.point[key] if clean.point[key] else float("nan")
    lo, hi = (np.percentile(r, 2.5), np.percentile(r, 97.5)) if r.size else (float("nan"),) * 2
    return Estimate(float(point), float(lo), float(hi))
