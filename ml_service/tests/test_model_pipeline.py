"""
Unit and integration tests for model building, saved weights, and inference predictor.
"""

import os
import sys
import numpy as np
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
for p in [current_dir, parent_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import tensorflow as tf
from ml_service.src.model import build_landslide_cnn
from ml_service.src.predict import LandslidePredictor
from ml_service.src.data_loader import LandslideDatasetLoader


def test_model_architecture():
    model = build_landslide_cnn(input_shape=(128, 128, 4))
    assert model.input_shape == (None, 128, 128, 4)
    assert model.output_shape == (None, 1)
    assert model.count_params() > 50000
    assert model.count_params() < 500000


def test_saved_model_exists():
    model_path = "ml_service/models/cnn_model.keras"
    assert os.path.exists(model_path), f"Saved model not found at {model_path}"
    assert os.path.getsize(model_path) > 500000


def test_predictor_single_patch():
    predictor = LandslidePredictor(
        model_path="ml_service/models/cnn_model.keras",
        config_path="ml_service/models/preprocessing_config.json",
        decision_threshold=0.5,
    )

    # Load real test sample
    loader = LandslideDatasetLoader(data_dir="ml_service/data/raw")
    test_samples = loader.get_sample_paths("test")
    assert len(test_samples) > 0

    img_4ch, mask, label = loader.load_single_sample(test_samples[0][1], test_samples[0][2])
    res = predictor.predict_patch(img_4ch)

    assert "prediction_class" in res
    assert res["prediction_class"] in ("LANDSLIDE DETECTED", "NORMAL / STABLE")
    assert 0.0 <= res["probability"] <= 1.0
    assert res["inference_time_ms"] > 0
    assert res["model_version"] == "GeoGuard_CNN_v1"


def test_predictor_invalid_shape():
    predictor = LandslidePredictor(
        model_path="ml_service/models/cnn_model.keras",
        config_path="ml_service/models/preprocessing_config.json",
    )
    bad_patch = np.zeros((64, 64, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        predictor.predict_patch(bad_patch)


if __name__ == "__main__":
    test_model_architecture()
    test_saved_model_exists()
    test_predictor_single_patch()
    test_predictor_invalid_shape()
    print("ALL MODEL PIPELINE TESTS PASSED!")
