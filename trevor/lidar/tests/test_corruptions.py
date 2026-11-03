import numpy as np
import pytest

from sdc_lidar import corruptions as C


def make_cloud(n=20000, seed=0, beams=32):
    rng = np.random.default_rng(seed)
    r = rng.uniform(1, 80, n)
    th = rng.uniform(-np.pi, np.pi, n)
    ring = rng.integers(0, beams, n)
    z = r * np.tan(np.radians(-30 + ring * (40.0 / beams)))
    z = np.clip(z, -3, 5)
    inten = rng.uniform(0, 255, n)
    return np.stack([r * np.cos(th), r * np.sin(th), z, inten, ring], 1).astype(np.float32)


def test_severity_zero_is_identity_for_all():
    pts = make_cloud()
    for name in C.REGISTRY:
        out = C.apply_to_sweeps(name, [pts, pts], 0)
        assert len(out) == 2 and all(o is pts or np.array_equal(o, pts) for o in out)


@pytest.mark.parametrize("name", ["point_drop", "beam_reduction", "range_limit"])
def test_monotonic_point_loss(name):
    pts = make_cloud()
    counts = [len(C.apply(name, pts, s, seed=0, frame_key="f")) for s in range(0, 6)]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] < counts[0]


def test_point_drop_rate_matches_table():
    pts = make_cloud(n=200000)
    for sev, p in enumerate(C.POINT_DROP_PROB, start=1):
        kept = len(C.apply("point_drop", pts, sev, seed=1, frame_key="x")) / len(pts)
        assert kept == pytest.approx(1 - p, abs=0.01)


def test_point_drop_is_seeded_and_order_independent():
    pts = make_cloud()
    a = C.apply("point_drop", pts, 3, seed=0, frame_key="scene-1/frame-7")
    b = C.apply("point_drop", pts, 3, seed=0, frame_key="scene-1/frame-7")
    c = C.apply("point_drop", pts, 3, seed=1, frame_key="scene-1/frame-7")
    d = C.apply("point_drop", pts, 3, seed=0, frame_key="scene-1/frame-8")
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c) and not np.array_equal(a, d)


def test_point_drop_output_is_subset_with_columns_intact():
    pts = make_cloud()
    out = C.apply("point_drop", pts, 2, seed=0)
    assert out.shape[1] == pts.shape[1]
    assert set(map(tuple, out[:50])) <= set(map(tuple, pts))


def test_beam_reduction_keeps_requested_ring_count():
    pts = make_cloud()
    for sev, n in enumerate(C.BEAMS_KEPT, start=1):
        rings = np.unique(C.apply("beam_reduction", pts, sev)[:, C.RING])
        assert len(rings) == n


def test_beam_reduction_fallback_without_ring_column():
    pts = make_cloud()[:, :4]
    out = C.beam_reduction(pts, 3, has_ring=False)
    assert 0 < len(out) < len(pts)


def test_range_limit_bound():
    pts = make_cloud()
    for sev, r in enumerate(C.RANGE_LIMIT_M, start=1):
        out = C.apply("range_limit", pts, sev)
        assert np.hypot(out[:, 0], out[:, 1]).max() <= r


def test_missing_sweeps_keeps_current_and_counts():
    sweeps = [make_cloud(100, seed=i) for i in range(10)]
    for sev, frac in enumerate(C.SWEEPS_DROPPED_FRAC, start=1):
        out = C.apply_to_sweeps("missing_sweeps", sweeps, sev, seed=0, frame_key="k")
        assert out[0] is sweeps[0]
        assert len(out) == 10 - int(round(frac * 9))
    assert len(C.apply_to_sweeps("missing_sweeps", sweeps, 5)) == 1


def test_missing_sweeps_seeded():
    sweeps = [make_cloud(10, seed=i) for i in range(10)]
    pick = lambda seed: [id(s) for s in C.apply_to_sweeps("missing_sweeps", sweeps, 3, seed, "k")]
    assert pick(0) == pick(0)
    assert any(pick(0) != pick(s) for s in (1, 2, 3, 4))


def test_apply_rejects_bad_input():
    pts = make_cloud()
    with pytest.raises(ValueError):
        C.apply("point_drop", pts, 6)
    with pytest.raises(KeyError):
        C.apply_to_sweeps("nope", [pts], 1)
    with pytest.raises(ValueError):
        C.apply("missing_sweeps", pts, 1)
    with pytest.raises(ValueError):
        C.apply("range_limit", pts[:, :2], 1)


def test_seeds_for():
    assert C.seeds_for("point_drop") == (0, 1, 2)
    assert C.seeds_for("range_limit") == (0,)
