import numpy as np
import pytest

from sdc_lidar.multisweep import RawSweep, accumulate, remove_close
from sdc_lidar import corruptions as C
from test_corruptions import make_cloud


def make_sweeps(n=9, pts=500):
    sweeps = []
    for i in range(n):
        R = np.eye(3)
        T = np.array([0.1 * (i + 1), 0.0, 0.0])
        sweeps.append(RawSweep(make_cloud(pts, seed=i + 1), R, T, timestamp_s=100.0 - 0.05 * (i + 1)))
    return sweeps


def test_clean_matches_bevfusion_semantics():
    key = make_cloud(1000)
    sweeps = make_sweeps()
    out = accumulate(key, 100.0, sweeps)
    n_expected = len(key) + sum(len(remove_close(s.points)) for s in sweeps)
    assert out.shape == (n_expected, 5)
    assert np.all(out[:len(key), 4] == 0)                       # keyframe time lag 0
    lag = out[len(key):, 4]
    assert lag.min() == pytest.approx(0.05) and lag.max() == pytest.approx(0.45)
    # translation applied: first sweep x shifted by +0.1 relative to its (close-filtered) raw x
    s0 = remove_close(sweeps[0].points)
    assert np.allclose(out[len(key):len(key) + len(s0), 0], s0[:, 0] + 0.1, atol=1e-5)


def test_pad_empty_sweeps_repeats_keyframe():
    key = make_cloud(300)
    out = accumulate(key, 1.0, [], sweeps_num=9)
    assert len(out) == len(key) + 9 * len(remove_close(key[:, :5]))


def test_sweeps_num_cap():
    key = make_cloud(100)
    assert len(accumulate(key, 100.0, make_sweeps(9), sweeps_num=3)) < len(accumulate(key, 100.0, make_sweeps(9)))


def test_ring_is_available_to_beam_reduction_before_it_is_overwritten():
    key, sweeps = make_cloud(5000), make_sweeps(9, 5000)
    clean = accumulate(key, 100.0, sweeps)
    red = accumulate(key, 100.0, sweeps, corruption="beam_reduction", severity=5)
    assert len(red) < 0.25 * len(clean)  # 4 of 32 beams kept, in key *and* every sweep


def test_missing_sweeps_drops_metadata_in_step_with_points():
    key, sweeps = make_cloud(200), make_sweeps(9, 200)
    out5 = accumulate(key, 100.0, sweeps, corruption="missing_sweeps", severity=5)
    assert len(out5) == len(key) and np.all(out5[:, 4] == 0)    # only the keyframe remains
    out3 = accumulate(key, 100.0, sweeps, corruption="missing_sweeps", severity=3, seed=0, frame_key="a")
    lags = {round(float(v), 3) for v in np.unique(np.round(out3[len(key):, 4], 3))}
    assert len(lags) == 9 - 5                                    # 60% of 9 -> 5 dropped
    # surviving lags must correspond to real sweep timestamps
    valid = {round(0.05 * (i + 1), 3) for i in range(9)}
    assert lags <= valid


def test_point_drop_applies_to_every_sweep_and_is_reproducible():
    key, sweeps = make_cloud(2000), make_sweeps(9, 2000)
    a = accumulate(key, 100.0, sweeps, corruption="point_drop", severity=3, seed=1, frame_key="s")
    b = accumulate(key, 100.0, sweeps, corruption="point_drop", severity=3, seed=1, frame_key="s")
    clean = accumulate(key, 100.0, sweeps)
    assert np.array_equal(a, b)
    assert len(a) / len(clean) == pytest.approx(1 - C.POINT_DROP_PROB[2], abs=0.02)


def test_requires_ring_column():
    with pytest.raises(ValueError):
        accumulate(make_cloud(10)[:, :4], 1.0, [])
