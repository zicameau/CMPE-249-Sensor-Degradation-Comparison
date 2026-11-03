import numpy as np

from sdc_lidar.detector_core import SweepBuffer, box_to_nuscenes, range_filter


def cloud(n, base=0.0, seed=0):
    r = np.random.default_rng(seed)
    p = r.uniform(2, 20, size=(n, 5)).astype(np.float32)
    p[:, 4] = base            # ring slot
    return p


def fill(buf, n_clouds, dt_ns=50_000_000, n=50):
    for i in range(n_clouds):
        buf.push(1_000_000_000 + i * dt_ns, "lidar", cloud(n, seed=i))


def test_key_is_newest_and_sweeps_are_nearest_first_with_time_lag():
    buf = SweepBuffer(sweeps_num=3)
    fill(buf, 6)
    pts, used = buf.build(lookup=None)
    assert used == 3
    lags = np.unique(np.round(pts[:, 4], 6))
    # key lag 0, then 0.05, 0.10, 0.15 s: the three most recent previous sweeps
    assert np.allclose(lags, [0.0, 0.05, 0.10, 0.15], atol=1e-6)


def test_key_stamp_selects_an_older_key_and_ignores_newer_clouds():
    buf = SweepBuffer(sweeps_num=2)
    fill(buf, 6)
    key = 1_000_000_000 + 3 * 50_000_000
    _, used = buf.build(lookup=None, key_stamp_ns=key)
    assert used == 2
    pts, _ = buf.build(lookup=None, key_stamp_ns=key)
    assert pts[:, 4].max() <= 0.10 + 1e-6      # nothing newer than the key leaks in


def test_sweeps_older_than_max_age_are_skipped():
    buf = SweepBuffer(sweeps_num=9, max_age_s=0.12)
    fill(buf, 8)
    _, used = buf.build(lookup=None)
    assert used == 2                              # 0.05 and 0.10 s old; 0.15 s is too old


def test_lookup_transform_is_applied_and_missing_pose_skips_the_sweep():
    buf = SweepBuffer(sweeps_num=2)
    fill(buf, 3, n=10)

    def lookup(kf, kt, sf, st):
        if kt - st > 60_000_000:                  # the older sweep has no pose
            return None
        return np.eye(3), np.array([100.0, 0.0, 0.0])

    pts, used = buf.build(lookup)
    assert used == 1
    # the one aligned sweep is translated by +100 m in x; the key cloud (lag 0) is not
    assert pts[pts[:, 4] == 0][:, 0].max() < 30
    assert pts[pts[:, 4] > 0][:, 0].min() > 100


def test_corruption_goes_through_the_same_accumulator():
    buf = SweepBuffer(sweeps_num=3)
    fill(buf, 5, n=200)
    clean, _ = buf.build(None)
    dropped, _ = buf.build(None, corruption="point_drop", severity=3, seed=1, frame_key=7)
    again, _ = buf.build(None, corruption="point_drop", severity=3, seed=1, frame_key=7)
    assert len(dropped) < len(clean)
    assert np.array_equal(dropped, again)         # seeded by (seed, frame_key): reproducible


def test_box_to_nuscenes_matches_output_to_nusc_box_convention():
    # BEVFusion box: x y z dx dy dz yaw vx vy; the converter uses dims as (w, l, h), heading -yaw - pi/2,
    # and the gravity centre (z + dz / 2).
    box = np.array([[10.0, -3.0, -1.0, 1.9, 4.5, 1.6, 0.3, 0.0, 0.0]])
    centre, size, heading = box_to_nuscenes(box)
    assert np.allclose(centre, [[10.0, -3.0, -0.2]])
    assert np.allclose(size, [[4.5, 1.9, 1.6]])             # length, width, height
    assert np.allclose(heading, [-0.3 - np.pi / 2])


def test_range_filter_matches_the_config_range():
    p = np.array([[0, 0, 0, 0, 0], [60, 0, 0, 0, 0], [0, -54.1, 0, 0, 0], [0, 0, 4, 0, 0], [53, 53, 2, 0, 0]], float)
    assert len(range_filter(p)) == 2
