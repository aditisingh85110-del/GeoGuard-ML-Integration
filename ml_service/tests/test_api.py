"""
Unit and integration tests for GeoGuard FastAPI microservice endpoints.
"""

import os
import io
import h5py
import pytest
import numpy as np
from fastapi.testclient import TestClient

from ml_service.app import app
from ml_service.src import database
from ml_service.config import settings


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "GeoGuard" in data["service"]
    assert data["status"] == "online"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["database_connected"] is True
    assert data["model_version"] == "GeoGuard_CNN_v1"


def test_latest_prediction_endpoint(client):
    response = client.get("/latest-prediction")
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["LANDSLIDE DETECTED", "NORMAL / STABLE"]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["decision_threshold"] == 0.40
    assert "processing_timestamp" in data
    assert data["model_version"] == "GeoGuard_CNN_v1"
    assert "high_risk_coordinates" in data


def test_predictions_history_endpoint(client):
    response = client.get("/predictions?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert "total_count" in data
    assert data["limit"] == 10
    assert data["offset"] == 0
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) > 0


def test_predict_manual_valid_h5(client):
    # Create an in-memory 4-channel test H5 patch (128, 128, 4)
    patch_data = np.random.normal(loc=1.0, scale=0.2, size=(128, 128, 4)).astype(np.float32)
    bio = io.BytesIO()
    with h5py.File(bio, "w") as f:
        f.create_dataset("img", data=patch_data)
    bio.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test_patch.h5", bio.getvalue(), "application/x-hdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["LANDSLIDE DETECTED", "NORMAL / STABLE"]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["data_status"] == "MANUAL_TEST"
    assert data["input_shape"] == [128, 128, 4]
    assert "notice" in data


def test_predict_manual_invalid_shape(client):
    # Bad shape: 64x64x4
    patch_data = np.random.normal(size=(64, 64, 4)).astype(np.float32)
    bio = io.BytesIO()
    with h5py.File(bio, "w") as f:
        f.create_dataset("img", data=patch_data)
    bio.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("bad_shape.h5", bio.getvalue(), "application/x-hdf")},
    )
    assert response.status_code == 400
    assert "Invalid tensor shape" in response.json()["detail"]


def test_predict_manual_invalid_file_extension(client):
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an h5 file", "text/plain")},
    )
    assert response.status_code == 400
    assert "HDF5" in response.json()["detail"]


def test_satellite_status_endpoint(client):
    response = client.get("/satellite-status")
    assert response.status_code == 200
    data = response.json()
    assert data["connection_status"] in ["CONNECTED", "UNAUTHENTICATED_TEST_MODE", "ERROR"]
    assert "dataspace.copernicus.eu" in data["stac_catalogue_url"]
    assert "monitored_aoi" in data
    assert data["polling_interval_minutes"] == 15


def test_trigger_satellite_check_endpoint(client):
    response = client.post("/trigger-satellite-check")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["completed", "success", "error"]


def test_iot_fusion_preview_endpoint(client):
    response = client.get("/iot-fusion-preview")
    assert response.status_code == 200
    data = response.json()
    assert "combined_risk_level" in data
    assert data["combined_risk_level"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert "action_recommendation" in data
    assert "satellite_macro_status" in data
    assert "iot_micro_telemetry" in data


def test_iot_fusion_cross_validate_post(client):
    telemetry_payload = {
        "sensor_id": "GEO-NODE-TEST",
        "soil_moisture_percent": 48.0,
        "rainfall_mm_24h": 75.0,
        "vibration_intensity": 0.22,
        "tilt_angle_deg": 3.5,
        "battery_percent": 90.0,
    }
    response = client.post("/iot-fusion-cross-validate", json=telemetry_payload)
    assert response.status_code == 200
    data = response.json()
    assert "combined_risk_level" in data
    assert "action_recommendation" in data
    assert data["iot_micro_telemetry"]["sensor_id"] == "GEO-NODE-TEST"
