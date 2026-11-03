"""Shared long-format results CSV (schema from the project proposal).

Every evaluation run, in every lane, appends rows here; all plots and tables read it.
"""
from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Optional

COLUMNS = ["model", "lane", "corruption", "severity", "seed", "hardware",
           "metric", "class", "value", "ci_low", "ci_high"]

LANES = {"camera", "lidar", "fusion"}

# Pooled summary rows (mrr_*, failure_threshold_*): severity 0 and seed POOLED_SEED.
# mrr_* use corruption="all"; failure_threshold_* name the corruption they describe.
AGGREGATE = "all"
POOLED_SEED = -1


@dataclass
class ResultRow:
    model: str
    lane: str
    corruption: str          # "clean" for severity 0
    severity: int            # 0 = clean, 1..5
    seed: int
    hardware: str            # e.g. laptop, hpc_a40
    metric: str              # e.g. ap2d_50, map, nds, recall_0_20, rr, latency_p50_ms
    cls: str                 # class name or "all"
    value: float
    ci_low: Optional[float] = None   # None -> blank cell = "no interval computed yet" (never a fake one)
    ci_high: Optional[float] = None

    def validate(self) -> None:
        if self.lane not in LANES:
            raise ValueError(f"lane must be one of {sorted(LANES)}, got {self.lane!r}")
        if not 0 <= int(self.severity) <= 5:
            raise ValueError(f"severity must be 0..5, got {self.severity}")
        if (self.severity == 0 and self.corruption not in ("clean", AGGREGATE)
                and self.seed != POOLED_SEED):
            raise ValueError("severity 0 rows must use corruption='clean' (pooled summaries: seed=-1)")
        if self.value is None or self.value != self.value:  # None / NaN
            raise ValueError(f"missing value for {self.metric}/{self.cls}")
        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError("ci_low and ci_high must both be set or both be None")
        if self.ci_low is not None:
            if self.ci_low != self.ci_low or self.ci_high != self.ci_high:
                raise ValueError(f"NaN CI for {self.metric}/{self.cls}")
            if not self.ci_low <= self.ci_high:
                raise ValueError(f"ci_low > ci_high for {self.metric}/{self.cls}")

    def as_csv_dict(self) -> dict:
        d = asdict(self)
        d["class"] = d.pop("cls")
        for k in ("ci_low", "ci_high"):
            if d[k] is None:
                d[k] = ""
        return d


def append_rows(path: str | os.PathLike, rows: Iterable[ResultRow], replace: bool = False) -> int:
    """Append rows (writing the header if the file is new). Returns rows written.

    ``replace=True`` makes re-running an analysis idempotent: first drop existing rows with the same
    (model, lane, hardware, metric) as any new row, so a rerun overwrites instead of duplicating.
    """
    rows = list(rows)
    for r in rows:
        r.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace and path.exists() and path.stat().st_size > 0:
        mine = {(r.model, r.lane, r.hardware, r.metric) for r in rows}
        with open(path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        kept = [e for e in existing if (e["model"], e["lane"], e["hardware"], e["metric"]) not in mine]
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(kept)
        os.replace(tmp, path)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r.as_csv_dict())
    return len(rows)


assert [f.name for f in fields(ResultRow) if f.name != "cls"] == [c for c in COLUMNS if c != "class"]
