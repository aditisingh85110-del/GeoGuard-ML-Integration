"""
Unit tests for spatial tiling and native 10m resolution calculations.
"""

import numpy as np
import pytest

from ml_service.src.spatial_tiling import (
    calculate_native_10m_pixel_dims,
    compute_128px_patch_bbox,
    extract_sliding_window_patches,
    compute_bbox_dimensions_meters,
)


def test_native_10m_pixel_dimensions():
    # 0.10 deg bbox in lat/lon
    bbox = [76.95, 12.00, 77.05, 12.10]
    w_px, h_px, rx, ry = calculate_native_10m_pixel_dims(bbox)
    # Ground distance is ~10.9 km x 11.1 km
    assert 1000 <= w_px <= 1200
    assert 1000 <= h_px <= 1200
    # Effective resolution should be approximately 10.0m
    assert 9.5 <= rx <= 10.5
    assert 9.5 <= ry <= 10.5


def test_compute_128px_patch_bbox():
    center_lon, center_lat = 77.00, 12.05
    bbox = compute_128px_patch_bbox(center_lon, center_lat)
    w_m, h_m, _ = compute_bbox_dimensions_meters(bbox)
    # Should span approximately 1,280 meters in both dimensions
    assert abs(w_m - 1280.0) < 5.0
    assert abs(h_m - 1280.0) < 5.0


def test_extract_sliding_window_patches():
    # Scene of 256x256 pixels
    scene = np.ones((256, 256, 4), dtype=np.float32)
    bbox = [76.98, 12.03, 77.02, 12.07]
    patches = extract_sliding_window_patches(scene, bbox, patch_size=128, overlap_ratio=0.20)

    # 256x256 with patch_size 128 and 20% overlap gives a 3x3 grid = 9 patches
    assert len(patches) == 9
    for p in patches:
        assert p["patch_arr"].shape == (128, 128, 4)
        assert p["is_usable"] is True
        assert len(p["bbox_wgs84"]) == 4
        assert "latitude" in p["centroid"]
        assert "longitude" in p["centroid"]
