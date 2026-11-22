
#!/usr/bin/env python3
"""Build the shared CAM_FRONT 2D AP50 comparison table."""

import argparse
import csv
import json
from pathlib import Path


CLASSES = [
    "car",
    "truck",
    "bus",
    "person",
    "bicycle",
    "motorcycle",
]

MODELS = {
    "YOLO11s": (
        "yolo11s_clean_2d_metrics.json",
        "verified_clean",
    ),
    "TransFusion-L": (
        "lidar_corrected_2d_metrics.json",
        "diagnostic_z_corrected",
    ),
    "BEVFusion": (
        "bevfusion_corrected_2d_metrics.json",
        "diagnostic_z_corrected",
    ),
}


def build_comparison(results_dir):
    rows = []

    for model, (filename, status) in MODELS.items():
        path = results_dir / filename

        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)

        metrics = data.get("metrics", data)

        if status == "diagnostic_z_corrected":
            if data.get("z_correction") != "half-height":
                raise ValueError(
                    f"{model} does not record half-height Z correction"
                )

            if data.get("z_correction_status") != "diagnostic_unverified":
                raise ValueError(
                    f"{model} is missing diagnostic correction metadata"
                )

        row = {
            "model": model,
            "map2d_50": metrics["map2d_50"],
            "evaluation_status": status,
        }

        for cls in CLASSES:
            row[f"{cls}_ap50"] = (
                metrics["per_class"]
                .get(cls, {})
                .get("ap2d_50")
            )

        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/shared_2d_comparison"),
    )
    args = parser.parse_args()

    rows = build_comparison(args.results_dir)

    output = args.results_dir / "three_model_comparison.csv"

    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output}")

    for row in rows:
        print(
            f"{row['model']}: "
            f"{row['map2d_50']:.4f} "
            f"({row['evaluation_status']})"
        )


if __name__ == "__main__":
    main()
