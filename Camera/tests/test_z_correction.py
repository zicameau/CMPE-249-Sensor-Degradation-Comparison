
import copy
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_3d_predictions_2d.py"
)

spec = importlib.util.spec_from_file_location(
    "evaluate_3d_predictions_2d",
    SCRIPT_PATH,
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

apply_z_correction = module.apply_z_correction


def test_half_height_z_correction():
    predictions = {
        "sample_1": [
            {
                "translation": [10.0, 20.0, 2.0],
                "size": [1.0, 2.0, 1.6],
            }
        ]
    }

    corrected = apply_z_correction(
        predictions,
        "half-height",
    )

    assert corrected == 1
    assert predictions["sample_1"][0]["translation"][2] == pytest.approx(1.2)


def test_no_correction_preserves_predictions():
    predictions = {
        "sample_1": [
            {
                "translation": [10.0, 20.0, 2.0],
                "size": [1.0, 2.0, 1.6],
            }
        ]
    }

    original = copy.deepcopy(predictions)

    corrected = apply_z_correction(predictions, "none")

    assert corrected == 0
    assert predictions == original


def test_invalid_box_height_raises_error():
    predictions = {
        "sample_1": [
            {
                "translation": [10.0, 20.0, 2.0],
                "size": [1.0, 2.0, -1.0],
            }
        ]
    }

    with pytest.raises(ValueError):
        apply_z_correction(predictions, "half-height")
