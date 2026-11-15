"""Utilities for recording and summarizing detector latency."""

import csv
import statistics
from pathlib import Path


FIELDS = [
    "frame_index",
    "inference_ms",
    "callback_ms",
    "detection_count",
]


class LatencyRecorder:
    def __init__(self, output_path, warmup_frames=5):
        self.output_path = Path(output_path)
        self.warmup_frames = warmup_frames
        self.records = []

    def add(self, inference_ms, callback_ms, detection_count):
        record = {
            "frame_index": len(self.records),
            "inference_ms": float(inference_ms),
            "callback_ms": float(callback_ms),
            "detection_count": int(detection_count),
        }

        self.records.append(record)
        return record

    def save(self):
        self.output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        with open(
            self.output_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.records)

    def summary(self):
        measured = self.records[self.warmup_frames:]

        if not measured:
            return None

        inference = [
            row["inference_ms"] for row in measured
        ]

        callback = [
            row["callback_ms"] for row in measured
        ]

        def percentile(values, percent):
            ordered = sorted(values)
            index = (len(ordered) - 1) * percent / 100
            lower = int(index)
            upper = min(lower + 1, len(ordered) - 1)
            weight = index - lower

            return (
                ordered[lower] * (1 - weight)
                + ordered[upper] * weight
            )

        return {
            "frames": len(measured),
            "warmup_excluded": min(
                self.warmup_frames, len(self.records)
            ),
            "inference_mean_ms": statistics.mean(inference),
            "inference_median_ms": statistics.median(inference),
            "inference_p95_ms": percentile(inference, 95),
            "inference_std_ms": (
                statistics.stdev(inference)
                if len(inference) > 1 else 0.0
            ),
            "callback_mean_ms": statistics.mean(callback),
            "callback_median_ms": statistics.median(callback),
            "callback_p95_ms": percentile(callback, 95),
            "inference_fps_estimate": (
                1000.0 / statistics.mean(inference)
                if statistics.mean(inference) > 0
                else None
            ),
        }
