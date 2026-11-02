"""
Unit tests for data_loader and preprocessing modules.
"""

import os
import tempfile
import numpy as np
import pytest

from ml_service.src.data_loader import (
    LandslideDatasetLoader,
    DEFAULT_SELECTED_BAND_INDICES,
    DEFAULT_PIXEL_THRESHOLD,
)
from ml_service.src.preprocessing import SatellitePreprocessor


def test_data_loader_initialization():
    loader = LandslideDatasetLoader(
        data_dir="ml_service/data/raw",
        selected_bands=[1, 2, 3, 7],
        pixel_threshold=100,
    )
    assert loader.selected_bands == [1, 2, 3, 7]
    assert loader.pixel_threshold == 100


def test_load_sample_paths():
    loader = LandslideDatasetLoader(data_dir="ml_service/data/raw")
    train_samples = loader.get_sample_paths("train")
    assert len(train_samples) > 0, "No train samples found in ml_service/data/raw/train"

    # Verify first sample paths exist
    sample_id, img_path, mask_path = train_samples[0]
    assert os.path.exists(img_path)
    assert os.path.exists(mask_path)


def test_load_single_sample():
    loader = LandslideDatasetLoader(data_dir="ml_service/data/raw")
    train_samples = loader.get_sample_paths("train")
    _, img_path, mask_path = train_samples[0]

    img_subset, mask, label = loader.load_single_sample(img_path, mask_path)
    assert img_subset.shape == (128, 128, 4)
    assert mask.shape == (128, 128)
    assert label in (0, 1)


def test_load_split():
    loader = LandslideDatasetLoader(data_dir="ml_service/data/raw", pixel_threshold=100)
    X, y, meta = loader.load_split("train", max_samples=10)

    assert X.shape == (10, 128, 128, 4)
    assert y.shape == (10,)
    assert len(meta) == 10
    assert set(np.unique(y)).issubset({0, 1})


def test_preprocessor_fit_transform():
    # Synthetic 4-channel satellite data
    X = np.random.normal(loc=1.5, scale=0.5, size=(10, 128, 128, 4)).astype(np.float32)

    preprocessor = SatellitePreprocessor(band_names=["B2", "B3", "B4", "B8"])
    X_norm = preprocessor.fit_transform(X)

    assert X_norm.shape == X.shape
    # Check that normalized channels have approximately mean ~ 0 and std ~ 1
    for c in range(4):
        assert abs(X_norm[:, :, :, c].mean()) < 0.1
        assert abs(X_norm[:, :, :, c].std() - 1.0) < 0.2


def test_preprocessor_save_load_config():
    X = np.random.uniform(0.5, 3.0, size=(5, 64, 64, 4)).astype(np.float32)
    prep = SatellitePreprocessor().fit(X)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        prep.save_config(tmp_path)
        assert os.path.exists(tmp_path)

        loaded_prep = SatellitePreprocessor.load_config(tmp_path)
        np.testing.assert_allclose(prep.mean, loaded_prep.mean, rtol=1e-5)
        np.testing.assert_allclose(prep.std, loaded_prep.std, rtol=1e-5)
        assert loaded_prep.band_names == prep.band_names
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_rgb_and_false_color_conversion():
    patch = np.random.uniform(0.5, 3.0, size=(128, 128, 4)).astype(np.float32)
    rgb = SatellitePreprocessor.to_rgb_display(patch)
    cir = SatellitePreprocessor.to_false_color_display(patch)

    assert rgb.shape == (128, 128, 3)
    assert rgb.dtype == np.uint8
    assert cir.shape == (128, 128, 3)
    assert cir.dtype == np.uint8


if __name__ == "__main__":
    test_data_loader_initialization()
    test_load_sample_paths()
    test_load_single_sample()
    test_load_split()
    test_preprocessor_fit_transform()
    test_preprocessor_save_load_config()
    test_rgb_and_false_color_conversion()
    print("ALL TESTS PASSED SUCCESSFULLY!")
