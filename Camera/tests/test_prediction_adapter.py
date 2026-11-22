import numpy as np
import pytest

from nuscenes.nuscenes import NuScenes

from sdc_camera.projection import project_sample
from sdc_camera.prediction_adapter import (
    prediction_class,
    project_predictions,
)


@pytest.mark.parametrize("name,expected", [
    ("car", "car"),
    ("truck", "truck"),
    ("bus", "bus"),
    ("pedestrian", "person"),
    ("bicycle", "bicycle"),
    ("motorcycle", "motorcycle"),
    ("human.pedestrian.adult", "person"),
    ("vehicle.car", "car"),
    ("trailer", None),
])
def test_prediction_class(name, expected):
    assert prediction_class(name) == expected


def test_projection_matches_ground_truth():
    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=r"C:\SJSU\datasets\nuScenes",
        verbose=False,
    )

    sample = nusc.sample[0]
    gt = project_sample(nusc, sample)

    predictions = []

    for token in sample["anns"]:
        ann = nusc.get("sample_annotation", token)

        predictions.append({
            "translation": ann["translation"],
            "size": ann["size"],
            "rotation": ann["rotation"],
            "detection_name": ann["category_name"],
            "detection_score": 1.0,
        })

    projected = project_predictions(
        nusc,
        {sample["token"]: predictions},
        allowed_tokens={sample["token"]},
    )

    def normalize(rows):
        return sorted(
            (
                row["cls"],
                tuple(row[k] for k in ("x1", "y1", "x2", "y2")),
            )
            for row in rows
        )

    expected = normalize(gt)
    actual = normalize(projected)

    assert len(expected) == len(actual)

    for (expected_cls, expected_box), (actual_cls, actual_box) in zip(
        expected, actual
    ):
        assert expected_cls == actual_cls
        np.testing.assert_allclose(
            expected_box,
            actual_box,
            rtol=0,
            atol=1e-5,
        )


def test_rejects_predictions_outside_manifest():
    class FakeNuScenes:
        pass

    with pytest.raises(ValueError, match="not in manifest"):
        project_predictions(
            FakeNuScenes(),
            {"unexpected_sample": []},
            allowed_tokens={"expected_sample"},
        )
