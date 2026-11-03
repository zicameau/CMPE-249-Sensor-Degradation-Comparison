import types

import numpy as np
import pytest

from sdc_lidar import corruptions as C
from sdc_lidar.ros_cloud import StreamCorruptor, rows_to_xyzirr, structured_view
from test_corruptions import make_cloud


def fake_msg(arr, extra_time=True):
    """Build a PointCloud2-like object with x y z intensity ring (+ a float64 'time' field)."""
    n = len(arr)
    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"), ("ring", "<u2")]
    if extra_time:
        dt.append(("time", "<f8"))
    buf = np.zeros(n, dtype=np.dtype(dt, align=True))
    for k, j in zip("xyz", range(3)):
        buf[k] = arr[:, j]
    buf["intensity"], buf["ring"] = arr[:, 3], arr[:, 4].astype(np.uint16)
    if extra_time:
        buf["time"] = np.arange(n) * 1e-6
    fields = []
    for name in buf.dtype.names:
        code = {"f4": 7, "u2": 4, "f8": 8}[buf.dtype[name].str[1:]]
        fields.append(types.SimpleNamespace(name=name, offset=buf.dtype.fields[name][1], datatype=code, count=1))
    return types.SimpleNamespace(fields=fields, point_step=buf.dtype.itemsize, width=n, height=1,
                                 data=buf.tobytes(), is_bigendian=False), buf


def test_roundtrip_view_matches_source():
    arr = make_cloud(1000)
    msg, buf = fake_msg(arr)
    rows = structured_view(msg)
    assert np.array_equal(rows["x"], buf["x"]) and np.array_equal(rows["ring"], buf["ring"])
    x = rows_to_xyzirr(rows)
    assert x.shape == (1000, 6) and np.array_equal(x[:, 5], np.arange(1000))


@pytest.mark.parametrize("name", ["point_drop", "beam_reduction", "range_limit"])
def test_stream_matches_offline_corruption(name):
    arr = make_cloud(5000)
    msg, _ = fake_msg(arr)
    rows = structured_view(msg)
    sc = StreamCorruptor(name, 3, seed=1)
    keep = sc.surviving_rows(rows, frame_key=12345)
    offline = C.apply(name, arr, 3, seed=1, frame_key=12345)
    assert len(keep) == len(offline)
    assert np.allclose(rows["x"][keep], offline[:, 0])           # same points, original fields intact
    assert "time" in rows[keep].dtype.names


def test_passthrough_modes_and_validation():
    msg, _ = fake_msg(make_cloud(100))
    rows = structured_view(msg)
    for name, sev in (("none", 3), ("point_drop", 0), ("missing_sweeps", 4)):
        assert len(StreamCorruptor(name, sev).surviving_rows(rows, 1)) == 100
    with pytest.raises(ValueError):
        StreamCorruptor("nope", 1)
    with pytest.raises(ValueError):
        StreamCorruptor("point_drop", 9)


def test_missing_ring_is_estimated():
    arr = make_cloud(2000)
    msg, _ = fake_msg(arr)
    msg.fields = [f for f in msg.fields if f.name != "ring"]
    rows = structured_view(msg)
    keep = StreamCorruptor("beam_reduction", 3).surviving_rows(rows, 1)
    assert 0 < len(keep) < 2000
