import numpy as np
import pytest

from sdc_common.nusc_eval import Draws
from sdc_common.robustness import NEVER, summarize

N = 200


def draws(value, noise=0.0, seed=0, key="map"):
    rng = np.random.default_rng(seed)
    return Draws({key: value}, {key: value + noise * rng.normal(size=N)})


def test_rr_mrr_and_failure_deterministic():
    clean = draws(0.60)
    runs = {("range_limit", s, 0): draws(0.60 * rr) for s, rr in zip(range(1, 6), (0.95, 0.9, 0.7, 0.5, 0.2))}
    out = summarize(clean, runs, "map")
    assert out.rr[("range_limit", 3, 0)].value == pytest.approx(0.7)
    assert out.failure["range_limit"].value == 3 and out.failure["range_limit"].ci_low == 3
    assert out.mrr.value == pytest.approx(np.mean([0.95, 0.9, 0.7, 0.5, 0.2]))


def test_never_fails_gets_sentinel():
    clean = draws(0.5)
    runs = {("point_drop", s, 0): draws(0.5 * 0.95) for s in range(1, 6)}
    assert summarize(clean, runs, "map").failure["point_drop"].value == NEVER


def test_seed_mean_and_equal_weight_per_corruption():
    clean = draws(1.0)
    runs = {}
    for s in range(1, 6):
        runs[("range_limit", s, 0)] = draws(0.5)                     # deterministic, RR 0.5
        for seed, v in enumerate((0.95, 0.85, 0.75)):                  # stochastic, 3 seeds -> mean 0.85
            runs[("point_drop", s, seed)] = draws(v)
    out = summarize(clean, runs, "map")
    assert out.rr_cs[("point_drop", 2)].value == pytest.approx(0.85)
    assert out.mrr.value == pytest.approx((0.5 + 0.85) / 2)          # not pooled by run count
    assert out.failure["point_drop"].value == NEVER                  # seed-mean never drops below 0.8
    assert out.failure["range_limit"].value == 1


def test_failure_threshold_ci_reflects_uncertainty():
    clean = draws(1.0, noise=0.0)
    # RR at severity 2 sits right at 0.8 with noise -> threshold flips between 2 and 3 across resamples
    runs = {("c", 1, 0): draws(0.95), ("c", 2, 0): draws(0.8, noise=0.05, seed=1), ("c", 3, 0): draws(0.5)}
    f = summarize(clean, runs, "map").failure["c"]
    assert f.ci_low == 2 and f.ci_high == 3


def test_paired_clean_noise_cancels_in_rr():
    rng = np.random.default_rng(0)
    shared = 0.05 * rng.normal(size=N)
    clean = Draws({"map": 0.6}, {"map": 0.6 + shared})
    corr = Draws({"map": 0.3}, {"map": 0.5 * (0.6 + shared)})        # exactly half of clean on every resample
    est = summarize(clean, {("c", 1, 0): corr}, "map").rr[("c", 1, 0)]
    assert est.ci_low == pytest.approx(0.5) and est.ci_high == pytest.approx(0.5)
