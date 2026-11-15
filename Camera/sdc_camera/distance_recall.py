"""Distance-binned recall for 2D object detection."""

from collections import defaultdict

from .evaluation_2d import iou


DISTANCE_BINS = {
    "near_0_20m": (0, 20),
    "medium_20_40m": (20, 40),
    "far_40m_plus": (40, float("inf")),
}


def evaluate_distance_recall(
    predictions,
    ground_truth,
    iou_threshold=0.5,
    confidence_threshold=0.25,
):
    """
    Calculate recall by ground-truth object distance.

    Matching is greedy by descending prediction confidence.
    Each GT object can be matched only once.

    Distance bins:
      near:   0 <= d < 20
      medium: 20 <= d < 40
      far:    d >= 40
    """

    gt_by_frame_class = defaultdict(list)

    for index, gt in enumerate(ground_truth):
        key = (
            gt["sample_token"],
            gt.get("camera", "CAM_FRONT"),
            gt["cls"],
        )
        gt_by_frame_class[key].append(index)

    sorted_predictions = sorted(
        (
            p for p in predictions
            if float(p["confidence"]) >= confidence_threshold
        ),
        key=lambda p: -float(p["confidence"]),
    )

    matched_gt = set()

    for pred in sorted_predictions:
        key = (
            pred["sample_token"],
            pred.get("camera", "CAM_FRONT"),
            pred["cls"],
        )

        best_index = None
        best_iou = 0.0

        for index in gt_by_frame_class.get(key, []):
            if index in matched_gt:
                continue

            overlap = iou(pred, ground_truth[index])

            if overlap > best_iou:
                best_iou = overlap
                best_index = index

        if (
            best_index is not None
            and best_iou >= iou_threshold
        ):
            matched_gt.add(best_index)

    results = {}

    for name, (lower, upper) in DISTANCE_BINS.items():
        indices = [
            index
            for index, gt in enumerate(ground_truth)
            if lower <= float(gt["distance_m"]) < upper
        ]

        tp = sum(index in matched_gt for index in indices)
        total = len(indices)

        results[name] = {
            "gt_count": total,
            "tp": tp,
            "fn": total - tp,
            "recall": tp / total if total else None,
        }

    return results
