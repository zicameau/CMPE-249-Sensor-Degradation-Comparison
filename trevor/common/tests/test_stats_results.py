import csv

import numpy as np
import pytest

from sdc_common import results as R
from sdc_common import stats as S


def test_bootstrap_weights_are_resamples():
    W = S.bootstrap_weights(50, 200, seed=1)
    assert W.shape == (200, 50) and np.all(W.sum(1) == 50) and W.min() >= 0


def test_bootstrap_ci_covers_mean_of_scene_values():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.5, 0.1, 100)
    est = S.bootstrap_ci(lambda w: float((w * vals).sum() / w.sum()), 100)
    assert est.value == pytest.approx(vals.mean())
    assert est.ci_low < est.value < est.ci_high
    # CI width ~ 2*1.96*sd/sqrt(n) = ~0.039
    assert 0.02 < est.ci_high - est.ci_low < 0.06


def test_paired_diff_detects_real_gap_and_not_noise():
    rng = np.random.default_rng(1)
    base = rng.normal(0.5, 0.1, 80)
    a = base + 0.1
    b = base
    f = lambda v: (lambda w: float((w * v).sum() / w.sum()))
    assert S.paired_bootstrap_diff(f(a), f(b), 80).excludes_zero()
    noise = base + rng.normal(0, 0.1, 80)
    assert not S.paired_bootstrap_diff(f(noise), f(base), 80).excludes_zero()


def test_rr_mrr_failure_threshold():
    assert S.resilience_rate(0.3, 0.6) == pytest.approx(0.5)
    assert np.isnan(S.resilience_rate(0.3, 0))
    assert S.mrr([1.0, 0.5, float("nan")]) == pytest.approx(0.75)
    assert S.failure_threshold({1: 0.95, 2: 0.85, 3: 0.79, 4: 0.5}) == 3
    assert S.failure_threshold({1: 0.95, 2: 0.9}) is None


def test_rr_bootstrap_point():
    v = np.linspace(0.4, 0.6, 40)
    clean = lambda w: float((w * v).sum() / w.sum())
    corr = lambda w: 0.5 * clean(w)
    est = S.rr_bootstrap(clean, corr, 40)
    assert est.value == pytest.approx(0.5) and est.ci_low == pytest.approx(0.5)


def test_seed_summary():
    assert S.seed_summary({0: 0.4, 1: 0.5, 2: 0.6}) == pytest.approx((0.5, 0.4, 0.6))


def test_latency_summary_drops_warmup_and_reports_run_range():
    runs = [[1000.0] * 50 + [10.0] * 100, [1000.0] * 50 + [12.0] * 100, [1000.0] * 50 + [14.0] * 100]
    out = S.latency_summary(runs)
    assert out["p50"].value == pytest.approx(12.0)
    assert (out["p50"].ci_low, out["p50"].ci_high) == (10.0, 14.0)
    with pytest.raises(ValueError):
        S.latency_summary([[1.0] * 50])


def row(**kw):
    d = dict(model="bevfusion_lidar", lane="lidar", corruption="point_drop", severity=2, seed=1,
             hardware="hpc_a40", metric="nds", cls="all", value=0.611, ci_low=0.598, ci_high=0.624)
    d.update(kw)
    return R.ResultRow(**d)


def test_append_rows_writes_header_once_and_proposal_columns(tmp_path):
    p = tmp_path / "r.csv"
    R.append_rows(p, [row()])
    R.append_rows(p, [row(metric="map")])
    rows = list(csv.DictReader(open(p)))
    assert list(rows[0].keys()) == ["model", "lane", "corruption", "severity", "seed", "hardware",
                                    "metric", "class", "value", "ci_low", "ci_high"]
    assert len(rows) == 2 and rows[0]["class"] == "all"


def test_replace_is_idempotent_and_scoped(tmp_path):
    p = tmp_path / "r.csv"
    R.append_rows(p, [row(metric="map", value=0.5), row(metric="nds"), row(model="other", metric="map")])
    R.append_rows(p, [row(metric="map", value=0.7)], replace=True)
    rows = list(csv.DictReader(open(p)))
    maps = [r for r in rows if r["metric"] == "map"]
    assert len(rows) == 3 and {(r["model"], r["value"]) for r in maps} == {("bevfusion_lidar", "0.7"), ("other", "0.611")}


def test_pooled_rows_allowed_with_severity_zero(tmp_path):
    R.append_rows(tmp_path / "r.csv", [row(corruption="all", severity=0, seed=-1, metric="mrr_map"),
                                       row(corruption="point_drop", severity=0, seed=-1,
                                           metric="failure_threshold_map")])


@pytest.mark.parametrize("bad", [dict(lane="radar"), dict(severity=6), dict(severity=0),
                                 dict(ci_low=0.9, ci_high=0.1), dict(value=float("nan")),
                                 dict(ci_low=0.5, ci_high=None), dict(ci_low=float("nan"), ci_high=0.5)])
def test_validation_rejects(bad, tmp_path):
    with pytest.raises(ValueError):
        R.append_rows(tmp_path / "r.csv", [row(**bad)])
    assert not (tmp_path / "r.csv").exists()
