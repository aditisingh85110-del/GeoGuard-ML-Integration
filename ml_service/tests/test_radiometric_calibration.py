"""
Unit tests for radiometric calibration and BOA reflectance conversions.
"""

import numpy as np
import pytest

from ml_service.src.radiometric_calibration import (
    OPTICAL_4CH_NAMES,
    OPTICAL_4CH_BOA_SCALE_FACTORS,
    convert_l4s_h5_to_boa,
    validate_satellite_radiometry,
)


def test_scale_factors_positive_and_ordered():
    assert len(OPTICAL_4CH_BOA_SCALE_FACTORS) == 4
    # All scale factors must be positive
    assert np.all(OPTICAL_4CH_BOA_SCALE_FACTORS > 0.0)
    # NIR scale factor is expected to be highest
    assert OPTICAL_4CH_BOA_SCALE_FACTORS[3] > OPTICAL_4CH_BOA_SCALE_FACTORS[0]


def test_convert_single_patch_3d():
    dummy = np.ones((128, 128, 4), dtype=np.float32)
    converted = convert_l4s_h5_to_boa(dummy)
    assert converted.shape == (128, 128, 4)
    # Check that each channel is scaled by its factor
    for i in range(4):
        np.testing.assert_allclose(converted[:, :, i], OPTICAL_4CH_BOA_SCALE_FACTORS[i], rtol=1e-5)


def test_convert_batch_4d():
    dummy = np.full((3, 128, 128, 4), 2.0, dtype=np.float32)
    converted = convert_l4s_h5_to_boa(dummy)
    assert converted.shape == (3, 128, 128, 4)
    for i in range(4):
        np.testing.assert_allclose(converted[:, :, :, i], 2.0 * OPTICAL_4CH_BOA_SCALE_FACTORS[i], rtol=1e-5)


def test_validate_valid_boa_patch():
    # Valid BOA patch with typical reflectance (B02: 0.06, B03: 0.08, B04: 0.07, B08: 0.28)
    valid_patch = np.zeros((128, 128, 4), dtype=np.float32)
    valid_patch[:, :, 0] = 0.06
    valid_patch[:, :, 1] = 0.08
    valid_patch[:, :, 2] = 0.07
    valid_patch[:, :, 3] = 0.28

    is_valid, status, diag = validate_satellite_radiometry(valid_patch)
    assert is_valid is True
    assert status == "VALID"
    assert diag["invalid_ratio"] == 0.0


def test_validate_corrupted_zero_fill():
    corrupted = np.zeros((128, 128, 4), dtype=np.float32)
    is_valid, status, diag = validate_satellite_radiometry(corrupted)
    assert is_valid is False
    assert status == "INSUFFICIENT_DATA"


def test_validate_cloud_saturation():
    cloud = np.full((128, 128, 4), 0.85, dtype=np.float32)
    is_valid, status, diag = validate_satellite_radiometry(cloud)
    assert is_valid is False
    assert status == "INSUFFICIENT_DATA"


def test_validate_uncalibrated_dn():
    raw_dn = np.full((128, 128, 4), 1500.0, dtype=np.float32)
    is_valid, status, diag = validate_satellite_radiometry(raw_dn)
    assert is_valid is False
    assert status == "RADIOMETRIC_MISMATCH"
