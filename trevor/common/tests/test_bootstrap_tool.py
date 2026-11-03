import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "bootstrap_sweep.py"
spec = importlib.util.spec_from_file_location("bootstrap_sweep", TOOL)
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)


def test_discover_runs(tmp_path):
    for name in ["clean", "point_drop_s2_seed1", "range_limit_s5_seed0", "junk", "point_drop_s9_seed0"]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "results_nusc.json").write_text("{}")
    (tmp_path / "beam_reduction_s1_seed0").mkdir()          # no results file -> ignored (unfinished run)
    clean, runs = bs.discover_runs(tmp_path)
    assert clean.name == "clean"
    assert set(runs) == {("point_drop", 2, 1), ("range_limit", 5, 0)}


def test_split_key():
    assert bs.split_key("map") == ("map", "all")
    assert bs.split_key("ap_nusc/construction_vehicle") == ("ap_nusc", "construction_vehicle")


def test_end_to_end_with_synthetic_runs(tmp_path):
    """clean + progressively degraded predictions -> RR falls, rows validate and land in the CSV."""
    pytest.importorskip("nuscenes")                          # only for the synthetic-data generator
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_nusc_eval import make_data, my_table
    from sdc_common import nusc_eval as NE, robustness as RB, stats
    from sdc_common.results import ResultRow, append_rows

    gt, pred, scene_of = make_data(11, n_scenes=12)
    W = stats.bootstrap_weights(12, 50, seed=1)
    rng = np.random.default_rng(0)
    clean = NE.bootstrap_draws(my_table(gt, pred, scene_of), weights=W)
    runs = {}
    for sev, keep in zip(range(1, 6), (0.9, 0.75, 0.55, 0.35, 0.15)):
        p2 = type(pred)()
        for tok in pred.sample_tokens:
            p2.add_boxes(tok, [b for b in pred[tok] if rng.random() < keep])
        runs[("point_drop", sev, 0)] = NE.bootstrap_draws(my_table(gt, p2, scene_of), weights=W)
    s = RB.summarize(clean, runs, "map")
    rr = [s.rr[("point_drop", sev, 0)].value for sev in range(1, 6)]
    assert rr == sorted(rr, reverse=True) and rr[-1] < 0.5
    assert 1 <= s.failure["point_drop"].value <= 5 and 0 < s.mrr.value < 1
    rows = [ResultRow("m", "lidar", "point_drop", sev, 0, "laptop", "rr_map", "all", e.value, e.ci_low, e.ci_high)
            for (c, sev, _), e in s.rr.items()]
    append_rows(tmp_path / "r.csv", rows)
    assert len(list(csv.DictReader(open(tmp_path / "r.csv")))) == 5
