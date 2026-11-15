import pytest

from sdc_camera.distance_recall import evaluate_distance_recall


def gt(distance, x1=0, cls="car"):
    return {
        "sample_token": "sample_1",
        "camera": "CAM_FRONT",
        "cls": cls,
        "x1": x1,
        "y1": 0,
        "x2": x1 + 10,
        "y2": 10,
        "distance_m": distance,
    }


def pred(x1=0, confidence=0.9, cls="car"):
    return {
        "sample_token": "sample_1",
        "camera": "CAM_FRONT",
        "cls": cls,
        "x1": x1,
        "y1": 0,
        "x2": x1 + 10,
        "y2": 10,
        "confidence": confidence,
    }


def test_distance_bin_boundaries():
    ground_truth = [
        gt(19.99, 0),
        gt(20.0, 20),
        gt(39.99, 40),
        gt(40.0, 60),
    ]
    predictions = [
        pred(0),
        pred(20),
        pred(40),
        pred(60),
    ]

    result = evaluate_distance_recall(predictions, ground_truth)

    assert result["near_0_20m"]["gt_count"] == 1
    assert result["medium_20_40m"]["gt_count"] == 2
    assert result["far_40m_plus"]["gt_count"] == 1

    assert all(r["recall"] == 1.0 for r in result.values())


def test_one_prediction_cannot_match_two_objects():
    ground_truth = [
        gt(10),
        gt(15),
    ]
    predictions = [pred()]

    result = evaluate_distance_recall(predictions, ground_truth)

    assert result["near_0_20m"]["tp"] == 1
    assert result["near_0_20m"]["fn"] == 1
    assert result["near_0_20m"]["recall"] == 0.5


def test_low_confidence_prediction_excluded():
    result = evaluate_distance_recall(
        [pred(confidence=0.1)],
        [gt(10)],
    )

    assert result["near_0_20m"]["recall"] == 0.0


def test_wrong_class_does_not_match():
    result = evaluate_distance_recall(
        [pred(cls="truck")],
        [gt(10, cls="car")],
    )

    assert result["near_0_20m"]["recall"] == 0.0
