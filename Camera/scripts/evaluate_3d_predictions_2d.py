
#!/usr/bin/env python3
"""Evaluate nuScenes global 3D detection exports against CAM_FRONT 2D GT.

Run from the Camera repository root.

Original evaluation:
    python scripts/evaluate_3d_predictions_2d.py \
        --predictions path/results_nusc.json \
        --dataroot path/to/nuscenes \
        --name lidar

Diagnostic half-height Z correction:
    python scripts/evaluate_3d_predictions_2d.py \
        --predictions path/results_nusc.json \
        --dataroot path/to/nuscenes \
        --name lidar_corrected \
        --z-correction half-height \
        --strict-missing

Note:
    The half-height correction is diagnostic. The original exporters'
    coordinate conventions have not been independently verified.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from sdc_camera.evaluation_2d import evaluate
from sdc_camera.prediction_adapter import (
    load_nuscenes_predictions,
    project_predictions,
)


def load_manifest(path):
    tokens = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError(
            "Manifest must contain unique, nonempty sample tokens"
        )

    return tokens


def load_gt(path, allowed, camera):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)

        required = {
            "sample_token",
            "camera",
            "cls",
            "x1",
            "y1",
            "x2",
            "y2",
        }

        columns = set(reader.fieldnames or [])
        missing = required - columns

        if missing:
            raise ValueError(
                f"Ground truth missing columns: {sorted(missing)}"
            )

        rows = []

        for row in reader:
            if (
                row["sample_token"] in allowed
                and row["camera"] == camera
            ):
                for field in ("x1", "y1", "x2", "y2"):
                    row[field] = float(row[field])

                rows.append(row)

    return rows


def apply_z_correction(predictions, correction):
    """Apply an optional diagnostic Z-coordinate correction in memory.

    For 'half-height', subtract half the box height from the
    exported translation Z coordinate.

    The source prediction JSON is never modified.
    """

    if correction == "none":
        return 0

    if correction != "half-height":
        raise ValueError(
            f"Unsupported Z correction: {correction}"
        )

    corrected_count = 0

    for sample_token, detections in predictions.items():
        for index, det in enumerate(detections):
            translation = det.get("translation")
            size = det.get("size")

            if (
                not isinstance(translation, (list, tuple))
                or not isinstance(size, (list, tuple))
                or len(translation) != 3
                or len(size) != 3
            ):
                raise ValueError(
                    "Invalid 3D box translation or size for "
                    f"sample {sample_token}, detection {index}"
                )

            z = float(translation[2])
            height = float(size[2])

            if (
                not math.isfinite(z)
                or not math.isfinite(height)
                or height <= 0
            ):
                raise ValueError(
                    "Invalid Z coordinate or box height for "
                    f"sample {sample_token}, detection {index}"
                )

            corrected_translation = list(translation)
            corrected_translation[2] = z - height / 2.0

            det["translation"] = corrected_translation
            corrected_count += 1

    return corrected_count


def write_csv(path, summary):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "model",
                "cls",
                "ap2d_50",
                "gt_count",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
            ],
        )

        writer.writeheader()

        for cls, metrics in summary["metrics"]["per_class"].items():
            writer.writerow({
                "model": summary["model"],
                "cls": cls,
                **metrics,
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="nuScenes results_nusc.json",
    )

    parser.add_argument(
        "--dataroot",
        type=Path,
        required=True,
        help="nuScenes dataset root",
    )

    parser.add_argument(
        "--version",
        default="v1.0-mini",
        help="nuScenes version, e.g. v1.0-mini",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/mini_val_sample_tokens.txt"),
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path(
            "results/camera_shared_mini_val/ground_truth.csv"
        ),
    )

    parser.add_argument(
        "--camera",
        default="CAM_FRONT",
    )

    parser.add_argument(
        "--name",
        default="3d_model",
        help="Model name in outputs",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/shared_2d_comparison"),
    )

    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if any manifest sample is missing in predictions",
    )

    parser.add_argument(
        "--z-correction",
        choices=["none", "half-height"],
        default="none",
        help=(
            "Optional diagnostic correction: subtract half "
            "the 3D box height from translation Z"
        ),
    )

    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes

    # Load the shared evaluation manifest.
    tokens = load_manifest(args.manifest)
    allowed = set(tokens)

    # Load the original nuScenes-format prediction export.
    predictions = load_nuscenes_predictions(args.predictions)

    # Check sample coverage.
    missing_tokens = [
        token for token in tokens
        if token not in predictions
    ]

    if missing_tokens and args.strict_missing:
        raise ValueError(
            f"{len(missing_tokens)} manifest samples missing "
            f"from prediction JSON; first: {missing_tokens[:5]}"
        )

    if missing_tokens:
        print(
            f"WARNING: {len(missing_tokens)} manifest samples "
            "absent from prediction export; "
            "treated as zero predictions."
        )

    # Keep only the samples in the shared manifest.
    filtered = {
        token: predictions[token]
        for token in tokens
        if token in predictions
    }

    for token, detections in filtered.items():
        if not isinstance(detections, list):
            raise ValueError(
                f"Predictions for {token} must be a list"
            )

    # Apply the optional diagnostic correction.
    corrected_count = apply_z_correction(
        filtered,
        args.z_correction,
    )

    if args.z_correction == "half-height":
        print(
            "Diagnostic Z correction applied: "
            f"translation[2] -= size[2] / 2 "
            f"for {corrected_count} detections"
        )

    # Load shared 2D ground truth.
    gt = load_gt(
        args.ground_truth,
        allowed,
        args.camera,
    )

    if not gt:
        raise ValueError(
            "No ground-truth rows match the manifest and camera"
        )

    # Load nuScenes calibration and annotation metadata.
    nusc = NuScenes(
        version=args.version,
        dataroot=str(args.dataroot),
        verbose=False,
    )

    # Project 3D detections into camera image coordinates.
    projected = project_predictions(
        nusc,
        filtered,
        camera=args.camera,
        allowed_tokens=allowed,
    )

    # Evaluate 2D detection performance.
    metrics = evaluate(projected, gt)

    # Record the evaluation configuration and results.
    summary = {
        "model": args.name,
        "camera": args.camera,
        "metric_definition": (
            "VOC-style continuous AP at IoU=0.50; "
            "excludes classes without GT"
        ),
        "manifest_sample_count": len(tokens),
        "samples_present_in_predictions": len(filtered),
        "missing_sample_count": len(missing_tokens),
        "missing_sample_tokens": missing_tokens,
        "extra_prediction_sample_count": len(
            set(predictions) - allowed
        ),
        "empty_detection_sample_count": sum(
            not detections for detections in filtered.values()
        ),
        "z_correction": args.z_correction,
        "z_correction_status": (
            "diagnostic_unverified"
            if args.z_correction == "half-height"
            else "none"
        ),
        "z_corrected_detection_count": corrected_count,
        "projected_detection_count": len(projected),
        "ground_truth_box_count": len(gt),
        "metrics": metrics,
    }

    # Save evaluation outputs.
    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        args.outdir / f"{args.name}_2d_metrics.json"
    )

    csv_path = (
        args.outdir / f"{args.name}_2d_per_class.csv"
    )

    projected_path = (
        args.outdir
        / f"{args.name}_projected_predictions.csv"
    )

    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    write_csv(csv_path, summary)

    with open(
        projected_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sample_token",
                "camera",
                "cls",
                "x1",
                "y1",
                "x2",
                "y2",
                "confidence",
            ],
        )

        writer.writeheader()
        writer.writerows(projected)

    print(
        f'{args.name}: 2D mAP50 = '
        f'{metrics["map2d_50"]:.4f} | '
        f'{len(filtered)}/{len(tokens)} samples present'
    )

    print(
        f"Wrote {json_path}, {csv_path}, {projected_path}"
    )


if __name__ == "__main__":
    main()
