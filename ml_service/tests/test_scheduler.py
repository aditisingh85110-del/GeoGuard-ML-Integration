"""
Unit tests for GeoGuard autonomous satellite scheduler.
"""

import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch

from ml_service.scheduler import SatelliteScheduler
from ml_service.src import database
from ml_service.config import settings


@pytest.fixture
def temp_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database.init_db(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def mock_pipeline():
    mock = MagicMock()
    # Mock copernicus client
    mock.copernicus = MagicMock()
    mock.copernicus.search_acquisitions.return_value = [
        {
            "acquisition_id": "S2A_TEST_NEW_001",
            "acquisition_time": "2024-03-01T08:00:00Z",
            "cloud_cover_percent": 12.0,
            "is_usable": True,
        }
    ]
    # Mock pipeline process_aoi
    mock.process_aoi.return_value = {
        "status": "success",
        "data_status": "VALID",
        "prediction": {
            "overall_class": "LANDSLIDE DETECTED",
            "max_probability": 0.88,
            "decision_threshold": 0.40,
            "model_version": "GeoGuard_CNN_v1",
        },
        "satellite": {
            "source": "Copernicus Sentinel-2",
            "acquisition_id": "S2A_TEST_NEW_001",
            "acquisition_time": "2024-03-01T08:00:00Z",
            "cloud_cover_percent": 12.0,
            "mode": "REAL_COPERNICUS",
        },
        "aoi": {
            "name": "TEST_DEVELOPMENT_AOI",
            "bbox": [76.95, 12.00, 77.05, 12.10],
            "is_test_aoi": True,
        },
        "aggregation": {
            "total_patches_evaluated": 2,
            "landslide_patches": 1,
            "normal_patches": 1,
            "high_risk_coordinates": [{"lat": 12.05, "lon": 77.00, "prob": 0.88}],
        },
    }
    return mock


def test_scheduler_poll_new_acquisition(temp_db_path, mock_pipeline):
    scheduler = SatelliteScheduler(pipeline=mock_pipeline, db_path=temp_db_path)
    result = scheduler.poll_once()

    assert result["status"] == "success"
    assert result["acquisition_id"] == "S2A_TEST_NEW_001"
    assert result["overall_class"] == "LANDSLIDE DETECTED"

    # Verify saved in DB
    latest = database.get_latest_prediction(db_path=temp_db_path)
    assert latest is not None
    assert latest["acquisition_id"] == "S2A_TEST_NEW_001"

    # Verify log
    latest_log = database.get_latest_polling_log(db_path=temp_db_path)
    assert latest_log["status"] == "NEW_ACQUISITION_PROCESSED"


def test_scheduler_poll_duplicate_handling(temp_db_path, mock_pipeline):
    scheduler = SatelliteScheduler(pipeline=mock_pipeline, db_path=temp_db_path)
    # First poll processes
    scheduler.poll_once()

    # Second poll should detect duplicate and skip re-computation
    mock_pipeline.process_aoi.reset_mock()
    result = scheduler.poll_once()

    assert result["status"] == "idle"
    assert result["reason"] == "already_processed"
    mock_pipeline.process_aoi.assert_not_called()

    latest_log = database.get_latest_polling_log(db_path=temp_db_path)
    assert latest_log["status"] == "NO_NEW_ACQUISITION_AVAILABLE"


def test_scheduler_poll_excessive_cloud(temp_db_path, mock_pipeline):
    # Configure mock with high cloud cover
    mock_pipeline.copernicus.search_acquisitions.return_value = [
        {
            "acquisition_id": "S2A_TEST_CLOUDY_002",
            "acquisition_time": "2024-03-02T08:00:00Z",
            "cloud_cover_percent": 65.5,
            "is_usable": False,
        }
    ]

    scheduler = SatelliteScheduler(pipeline=mock_pipeline, db_path=temp_db_path)
    result = scheduler.poll_once()

    assert result["status"] == "skipped"
    assert result["reason"] == "excessive_cloud_cover"
    assert result["cloud_cover_percent"] == 65.5
    mock_pipeline.process_aoi.assert_not_called()

    latest_log = database.get_latest_polling_log(db_path=temp_db_path)
    assert latest_log["status"] == "EXCESSIVE_CLOUD"


def test_scheduler_network_error_resilience(temp_db_path, mock_pipeline):
    # Simulate network failure
    mock_pipeline.copernicus.search_acquisitions.side_effect = Exception("Connection timed out to Copernicus STAC")

    scheduler = SatelliteScheduler(pipeline=mock_pipeline, db_path=temp_db_path)
    result = scheduler.poll_once()

    assert result["status"] == "error"
    assert result["reason"] == "catalogue_connection_error"

    latest_log = database.get_latest_polling_log(db_path=temp_db_path)
    assert latest_log["status"] == "ERROR"
    assert "Connection timed out" in latest_log["message"]
