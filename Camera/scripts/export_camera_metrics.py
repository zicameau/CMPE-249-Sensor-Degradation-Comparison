"""Export Camera evaluation metrics to the team's shared CSV schema."""

import argparse
import csv
import json
from pathlib import Path


COLUMNS = [
    "model",
    "lane",
    "corruption",
    "severity",
    "seed",
    "hardware",
    "metric",
    "class",
    "value",
    "ci_low",
    "ci_high",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="results/camera_shared_mini_val",
    )
    parser.add_argument(
        "--model",
        default="YOLO11s",
    )
    parser.add_argument(
        "--hardware",
        default="",
        help="Hardware used for the experiment, if known",
    )
    parser.add_argument(
        "--out",
        default="results/camera_shared_metrics.csv",
    )
    args = parser.parse_args()

    root = Path(args.results)

    metrics = json.loads(
        (root / "metrics.json").read_text()
    )
    distance = json.loads(
        (root / "distance_recall.json").read_text()
    )

    rows = []

    def add_row(condition, severity, metric, cls, value):
        if value is None:
            return

        rows.append({
            "model": args.model,
            "lane": "camera",
            "corruption": condition,
            "severity": severity,
            "seed": "",
            "hardware": args.hardware,
            "metric": metric,
            "class": cls,
            "value": value,
            "ci_low": "",
            "ci_high": "",
        })

    clean_map = metrics["clean_0"]["map2d_50"]

    for label, result in metrics.items():
        condition, severity = label.rsplit("_", 1)
        severity = int(severity)

        map50 = result["map2d_50"]

        add_row(
            condition,
            severity,
            "map2d_50",
            "all",
            map50,
        )

        if clean_map is not None and clean_map > 0:
            add_row(
                condition,
                severity,
                "robustness_ratio_map2d_50",
                "all",
                map50 / clean_map,
            )

        for cls, class_metrics in result["per_class"].items():
            for metric in ("ap2d_50", "precision", "recall"):
                add_row(
                    condition,
                    severity,
                    metric,
                    cls,
                    class_metrics[metric],
                )

        for distance_bin, bin_metrics in distance[label].items():
            add_row(
                condition,
                severity,
                "distance_recall_2d_50_conf025",
                distance_bin,
                bin_metrics["recall"],
            )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", output)
    print("Metric rows:", len(rows))
    print("Conditions:", len(metrics))
    print("Columns:", ", ".join(COLUMNS))


if __name__ == "__main__":
    main()
