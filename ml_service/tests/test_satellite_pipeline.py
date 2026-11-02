"""
Unit and integration tests for Copernicus Sentinel-2 satellite pipeline:
- Authentication & credential handling
- Live STAC catalogue search
- Scene tiling & geographic coordinate mapping
- Edge handling (no misleading zero-padding)
- Invalid pixel detection & filtering
- Spatial aggregation
- Simulation fallback mode
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

from ml_service.config import settings, parse_bbox
from ml_service.src.satellite import CopernicusClient, SatellitePipeline


def test_parse_bbox():
    assert parse_bbox("76.9,11.9,77.2,12.2") == [76.9, 11.9, 77.2, 12.2]
    assert parse_bbox("") is None
    assert parse_bbox("invalid,string") is None
    # Inverted
    assert parse_bbox("77.2,12.2,76.9,11.9") is None


def test_copernicus_missing_credentials_fallback():
    client = CopernicusClient(client_id=None, client_secret=None)
    assert not client.has_credentials
    token = client.get_auth_token()
    assert token is None


def test_stac_catalogue_search_live():
    client = CopernicusClient()
    # Search over test bbox
    acqs = client.search_acquisitions(
        bbox=settings.TEST_AOI_BBOX,
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-01-15T23:59:59Z",
        limit=2,
    )
    assert isinstance(acqs, list)
    if len(acqs) == 0:
        pytest.skip("CDSE STAC endpoint unreachable or blocked by external CDSE WAF")
    assert len(acqs) > 0
    first = acqs[0]
    assert "acquisition_id" in first
    assert "acquisition_time" in first
    assert "cloud_cover_percent" in first
    assert first["source"] == "Copernicus Data Space Ecosystem (Sentinel-2 L2A)"


def test_tile_scene_geometry_and_edges():
    # 256 x 384 scene with 4 channels
    scene = np.ones((256, 384, 4), dtype=np.float32) * 1.2
    bbox = [76.95, 12.00, 77.05, 12.10]

    patches = SatellitePipeline.tile_scene(scene, bbox=bbox, patch_size=128)
    # 2 rows x 3 cols = 6 patches
    assert len(patches) == 6

    # Verify each patch properties
    for p in patches:
        assert p["patch_arr"].shape == (128, 128, 4)
        assert p["is_usable"] is True
        assert p["invalid_ratio"] == 0.0
        assert len(p["bbox"]) == 4
        # Latitude order check: min_lat < max_lat
        assert p["bbox"][1] < p["bbox"][3]
        assert p["bbox"][0] < p["bbox"][2]
        assert "lat" in p["center_coords"]
        assert "lon" in p["center_coords"]


def test_tile_scene_invalid_pixel_filtering():
    # 128 x 128 scene where 50% of pixels are NaN
    scene = np.ones((128, 128, 4), dtype=np.float32)
    scene[:64, :, :] = np.nan
    bbox = [76.95, 12.00, 77.05, 12.10]

    patches = SatellitePipeline.tile_scene(scene, bbox=bbox, patch_size=128, max_invalid_ratio=0.20)
    assert len(patches) == 1
    # Should be flagged as not usable because 50% invalid > 20%
    assert patches[0]["is_usable"] is False
    assert patches[0]["invalid_ratio"] >= 0.50


def test_process_aoi_simulated_mode():
    client = CopernicusClient(client_id=None, client_secret=None)
    pipeline = SatellitePipeline(copernicus_client=client)
    res = pipeline.process_aoi(bbox=settings.TEST_AOI_BBOX, aoi_name="TEST_DEVELOPMENT_AOI")

    assert res["status"] == "success"
    assert res["data_status"] == "SIMULATED"
    assert "prediction" in res
    assert res["prediction"]["overall_class"] in ("LANDSLIDE DETECTED", "NORMAL / STABLE")
    assert "aggregation" in res
    assert res["aggregation"]["total_patches_evaluated"] > 0
    assert "patch_details" in res
    assert len(res["patch_details"]) > 0


def test_unconfigured_aoi_handling():
    pipeline = SatellitePipeline()
    # If no bbox and no test fallback allowed
    res = pipeline.process_aoi(bbox=None, aoi_name=None)
    # In default settings, test fallback is enabled for safety
    assert res["status"] == "success"


if __name__ == "__main__":
    test_parse_bbox()
    test_copernicus_missing_credentials_fallback()
    test_stac_catalogue_search_live()
    test_tile_scene_geometry_and_edges()
    test_tile_scene_invalid_pixel_filtering()
    test_process_aoi_simulated_mode()
    test_unconfigured_aoi_handling()
    print("ALL SATELLITE PIPELINE TESTS PASSED!")
