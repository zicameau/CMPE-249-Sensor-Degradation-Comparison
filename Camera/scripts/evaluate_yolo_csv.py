"""Evaluate saved YOLO11 predictions using the shared 2D scorer."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from nuscenes.nuscenes import NuScenes

from sdc_camera.evaluation_2d import evaluate
from sdc_camera.projection import COMMON_CLASSES, project_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--out", default="results/yolo_shared_eval.json")
    args = parser.parse_args()

    with open(args.predictions, newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))

    if not raw:
        raise ValueError("Prediction CSV is empty")

    tokens = sorted({r["sample_token"] for r in raw})
    conditions = sorted({r["condition"] for r in raw})

    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=args.dataroot,
        verbose=False,
    )

    # Use the same ground truth for every condition.
    gt = []
    for token in tokens:
        gt.extend(project_sample(nusc, nusc.get("sample", token)))

    # COCO names already match five of our classes.
    # COCO 'person' also matches our shared taxonomy.
    valid_classes = set(COMMON_CLASSES)

    results = {}

    for condition in conditions:
        predictions = []

        for row in raw:
            if row["condition"] != condition:
                continue

            cls = row["class_name_coco"]
            if cls not in valid_classes:
                continue

            predictions.append({
                "sample_token": row["sample_token"],
                "camera": row["camera"],
                "cls": cls,
                "confidence": float(row["confidence"]),
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
            })

        metrics = evaluate(predictions, gt)

        results[condition] = {
            "sample_count": len(tokens),
            "prediction_count": len(predictions),
            **metrics,
        }

        print(
            f"{condition}: "
            f"{len(predictions)} predictions, "
            f"mAP50={metrics['map2d_50']:.4f}"
        )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))

    print("Saved:", output)


if __name__ == "__main__":
    main()
