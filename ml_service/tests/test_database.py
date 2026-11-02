"""
Unit tests for GeoGuard SQLite database layer.
"""

import os
import json
import pytest
import tempfile
from ml_service.src import database


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database.init_db(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_init_db(temp_db):
    assert os.path.exists(temp_db)
    # Check tables
    conn = database.get_connection(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    assert "predictions" in tables
    assert "polling_logs" in tables
    conn.close()


def test_save_and_get_latest_prediction(temp_db):
    pred_data = {
        "prediction": {
            "overall_class": "LANDSLIDE DETECTED",
            "max_probability": 0.895,
            "decision_threshold": 0.40,
            "model_version": "GeoGuard_CNN_v1",
        },
        "satellite": {
            "source": "Copernicus Sentinel-2",
            "acquisition_id": "S2A_MSIL2A_TEST_001",
            "acquisition_time": "2024-03-01T10:30:00Z",
            "cloud_cover_percent": 4.5,
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
            "high_risk_coordinates": [{"lat": 12.05, "lon": 77.00, "prob": 0.895}],
        },
        "data_status": "VALID",
    }

    pred_id = database.save_prediction(pred_data, db_path=temp_db)
    assert pred_id == 1

    latest = database.get_latest_prediction(db_path=temp_db)
    assert latest is not None
    assert latest["prediction_class"] == "LANDSLIDE DETECTED"
    assert pytest.approx(latest["probability"], 0.001) == 0.895
    assert latest["acquisition_id"] == "S2A_MSIL2A_TEST_001"
    assert latest["is_simulated"] == 1  # because is_test_aoi is True


def test_deduplication_check(temp_db):
    assert not database.is_acquisition_processed("S2A_MSIL2A_TEST_001", db_path=temp_db)

    pred_data = {
        "prediction_class": "NORMAL / STABLE",
        "probability": 0.12,
        "acquisition_id": "S2A_MSIL2A_TEST_001",
        "aoi_name": "CORRIDOR_1",
    }
    database.save_prediction(pred_data, db_path=temp_db)

    assert database.is_acquisition_processed("S2A_MSIL2A_TEST_001", db_path=temp_db)
    assert not database.is_acquisition_processed("S2A_MSIL2A_TEST_002", db_path=temp_db)


def test_polling_logs(temp_db):
    log_id = database.log_polling_event(
        status="NO_NEW_ACQUISITION_AVAILABLE",
        latest_catalog_id="S2A_MSIL2A_TEST_001",
        message="No new scene found in CDSE STAC",
        db_path=temp_db,
    )
    assert log_id == 1

    latest_log = database.get_latest_polling_log(db_path=temp_db)
    assert latest_log is not None
    assert latest_log["status"] == "NO_NEW_ACQUISITION_AVAILABLE"
    assert latest_log["latest_catalog_acquisition_id"] == "S2A_MSIL2A_TEST_001"
