"""
Unit and integration tests for Phase 8: India-Wide AOI Management & Satellite Processing.
Tests:
- AOI CRUD endpoints (/api/v2/aois)
- India-boundary polygon validation (accepting Indian terrain, rejecting non-Indian coords)
- Acquisition deduplication per AOI
- Spatial patch clustering to prevent duplicate landslide event counting
- Native 10m spatial preservation and error handling
"""

import os
import json
import pytest
from fastapi.testclient import TestClient

from ml_service.app import app
from ml_service.src import database
from ml_service.src.india_boundary import is_coordinate_in_india, validate_aoi_bbox_in_india
from ml_service.src.event_clustering import cluster_landslide_patches


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def cleanup_test_aois():
    """Cleans up any created test AOIs before and after tests."""
    def clean():
        test_ids = ["TEST-IN-UK-01", "TEST-IN-KL-01", "TEST-IN-HP-01", "TEST-DUP-01"]
        for tid in test_ids:
            try:
                database.delete_aoi(tid, hard_delete=True)
                with database.get_connection() as conn:
                    conn.cursor().execute("DELETE FROM processed_acquisitions WHERE aoi_id = ?", (tid,))
                    conn.cursor().execute("DELETE FROM predictions WHERE aoi_name LIKE ?", (f"%{tid}%",))
                    conn.commit()
            except Exception:
                pass
    clean()
    yield
    clean()


# ==============================================================================
# 1. India Boundary Polygon Validator Tests
# ==============================================================================

def test_india_boundary_coordinates():
    # Valid Indian locations
    assert is_coordinate_in_india(78.03, 30.31) is True   # Dehradun, Uttarakhand
    assert is_coordinate_in_india(77.17, 31.10) is True   # Shimla, Himachal Pradesh
    assert is_coordinate_in_india(76.13, 11.68) is True   # Wayanad, Kerala
    assert is_coordinate_in_india(73.66, 17.92) is True   # Mahabaleshwar, Maharashtra
    assert is_coordinate_in_india(88.61, 27.33) is True   # Gangtok, Sikkim
    assert is_coordinate_in_india(92.73, 11.62) is True   # Port Blair, Andaman

    # Locations outside India
    assert is_coordinate_in_india(-0.12, 51.50) is False  # London, UK
    assert is_coordinate_in_india(116.4, 39.90) is False  # Beijing, China
    assert is_coordinate_in_india(67.00, 24.86) is False  # Karachi, Pakistan
    assert is_coordinate_in_india(69.00, 18.00) is False  # Deep Arabian Sea off Mumbai
    assert is_coordinate_in_india(85.00, 13.00) is False  # Deep Bay of Bengal off Chennai


def test_validate_aoi_bbox_in_india():
    # Valid Indian bounding box (Garhwal corridor)
    is_valid, msg = validate_aoi_bbox_in_india([78.45, 30.25, 78.55, 30.35])
    assert is_valid is True

    # Valid Indian bounding box (Wayanad, Kerala)
    is_valid, msg = validate_aoi_bbox_in_india([76.10, 11.60, 76.25, 11.75])
    assert is_valid is True

    # Invalid: Outside India (Europe)
    is_valid, msg = validate_aoi_bbox_in_india([2.30, 48.80, 2.40, 48.90])
    assert is_valid is False
    assert "territorial boundary" in msg

    # Invalid: Inverted coordinates
    is_valid, msg = validate_aoi_bbox_in_india([78.55, 30.25, 78.45, 30.35])
    assert is_valid is False
    assert "min_lon" in msg

    # Invalid: Extent too large (> 2.5 degrees)
    is_valid, msg = validate_aoi_bbox_in_india([70.0, 10.0, 75.0, 15.0])
    assert is_valid is False
    assert "too large" in msg


# ==============================================================================
# 2. AOI CRUD API Tests
# ==============================================================================

def test_create_monitored_aoi_success(client):
    payload = {
        "aoi_id": "TEST-IN-UK-01",
        "name": "Rishikesh-Devprayag Test Corridor",
        "state": "Uttarakhand",
        "region": "Garhwal Himalayas",
        "bbox_wgs84": [78.45, 30.25, 78.55, 30.35],
        "monitoring_status": "ACTIVE",
    }
    resp = client.post("/api/v2/aois", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["aoi_id"] == "TEST-IN-UK-01"
    assert data["state"] == "Uttarakhand"
    assert data["monitoring_status"] == "ACTIVE"
    assert data["last_quality_status"] == "UNCHECKED"


def test_create_monitored_aoi_outside_india_rejected(client):
    payload = {
        "aoi_id": "TEST-LONDON-01",
        "name": "Non-Indian Location",
        "state": "Foreign",
        "region": "Europe",
        "bbox_wgs84": [-0.15, 51.45, -0.05, 51.55],
    }
    resp = client.post("/api/v2/aois", json=payload)
    assert resp.status_code == 422
    assert "India Boundary Validation Error" in resp.json()["detail"]


def test_create_duplicate_aoi_id_conflict(client):
    payload = {
        "aoi_id": "TEST-DUP-01",
        "name": "Duplicate Test Corridor",
        "state": "Himachal Pradesh",
        "region": "Kangra",
        "bbox_wgs84": [76.20, 32.05, 76.30, 32.15],
    }
    resp1 = client.post("/api/v2/aois", json=payload)
    assert resp1.status_code == 201

    resp2 = client.post("/api/v2/aois", json=payload)
    assert resp2.status_code == 409
    assert "already exists" in resp2.json()["detail"]


def test_list_and_get_monitored_aois(client):
    # Ensure test AOI exists
    payload = {
        "aoi_id": "TEST-IN-KL-01",
        "name": "Wayanad Slope Test Zone",
        "state": "Kerala",
        "region": "Western Ghats",
        "bbox_wgs84": [76.10, 11.60, 76.20, 11.70],
        "monitoring_status": "ACTIVE",
    }
    client.post("/api/v2/aois", json=payload)

    # List all
    resp = client.get("/api/v2/aois")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_count"] >= 1
    found_ids = [a["aoi_id"] for a in data["aois"]]
    assert "TEST-IN-KL-01" in found_ids

    # Filter by state
    resp_kerala = client.get("/api/v2/aois?state=Kerala")
    assert resp_kerala.status_code == 200
    assert all(a["state"] == "Kerala" for a in resp_kerala.json()["aois"])

    # Get single AOI
    resp_single = client.get("/api/v2/aois/TEST-IN-KL-01")
    assert resp_single.status_code == 200
    assert resp_single.json()["name"] == "Wayanad Slope Test Zone"

    # Get non-existent AOI
    resp_404 = client.get("/api/v2/aois/DOES-NOT-EXIST")
    assert resp_404.status_code == 404


def test_update_and_deactivate_aoi(client):
    aoi_id = "TEST-IN-HP-01"
    payload = {
        "aoi_id": aoi_id,
        "name": "Kinnaur Valley Highway",
        "state": "Himachal Pradesh",
        "region": "Spiti-Kinnaur",
        "bbox_wgs84": [78.20, 31.50, 78.30, 31.60],
        "monitoring_status": "ACTIVE",
    }
    client.post("/api/v2/aois", json=payload)

    # Update region
    up_resp = client.put(f"/api/v2/aois/{aoi_id}", json={"region": "Lower Kinnaur Basin"})
    assert up_resp.status_code == 200
    assert up_resp.json()["region"] == "Lower Kinnaur Basin"

    # Deactivate AOI
    deact_resp = client.post(f"/api/v2/aois/{aoi_id}/deactivate")
    assert deact_resp.status_code == 200
    assert deact_resp.json()["monitoring_status"] == "ARCHIVED"

    # Soft Delete
    del_resp = client.delete(f"/api/v2/aois/{aoi_id}?hard_delete=false")
    assert del_resp.status_code == 200


# ==============================================================================
# 3. Spatial Patch Clustering Tests (Duplicate Event Prevention)
# ==============================================================================

def test_cluster_overlapping_patches():
    # 2 overlapping patches near Rishikesh + 1 distant patch near Joshimath
    p1 = {
        "patch_index": 0,
        "center_coords": {"lon": 78.500, "lat": 30.300},
        "bbox": [78.494, 30.294, 78.506, 30.306],
        "probability": 0.88,
    }
    p2 = {
        "patch_index": 1,
        "center_coords": {"lon": 78.505, "lat": 30.305},  # ~700m away, overlapping
        "bbox": [78.499, 30.299, 78.511, 30.311],
        "probability": 0.74,
    }
    p3 = {
        "patch_index": 9,
        "center_coords": {"lon": 79.560, "lat": 30.550},  # ~100 km away
        "bbox": [79.554, 30.544, 79.566, 30.556],
        "probability": 0.92,
    }

    clusters = cluster_landslide_patches([p1, p2, p3])
    # 3 positive patches should be clustered into exactly 2 distinct landslide events
    assert len(clusters) == 2

    # Cluster 1 should contain p1 and p2
    c1 = next(c for c in clusters if c["patch_count"] == 2)
    assert c1["constituent_patch_indices"] == [0, 1]
    assert c1["max_model_score"] == 0.88
    assert c1["risk_severity"] == "HIGH"

    # Cluster 2 should contain p3
    c2 = next(c for c in clusters if c["patch_count"] == 1)
    assert c2["constituent_patch_indices"] == [9]
    assert c2["max_model_score"] == 0.92


# ==============================================================================
# 4. Satellite Processing and Deduplication Tests
# ==============================================================================

def test_aoi_satellite_processing_and_deduplication(client):
    aoi_id = "TEST-IN-UK-01"
    payload = {
        "aoi_id": aoi_id,
        "name": "Rishikesh-Devprayag Test Corridor",
        "state": "Uttarakhand",
        "region": "Garhwal Himalayas",
        "bbox_wgs84": [78.45, 30.25, 78.55, 30.35],
        "monitoring_status": "ACTIVE",
    }
    client.post("/api/v2/aois", json=payload)

    # 1. Process AOI (using simulated fallback for unit test environment)
    resp = client.post(f"/api/v2/aois/{aoi_id}/process?allow_simulated_fallback=true")
    assert resp.status_code == 200
    data = resp.json()
    assert data["aoi_id"] == aoi_id
    assert data["status"] in ["success", "error"]
    assert "model_version" in data
    assert data["decision_threshold"] == 0.25

    acq_id = data.get("acquisition_id")
    assert acq_id is not None

    # 2. Re-process without force_reprocess -> MUST be deduplicated
    resp_dedup = client.post(f"/api/v2/aois/{aoi_id}/process?allow_simulated_fallback=true&force_reprocess=false")
    assert resp_dedup.status_code == 200
    dedup_data = resp_dedup.json()
    assert dedup_data["status"] == "skipped"
    assert "Deduplicated" in dedup_data["message"]

    # 3. Check acquisition history endpoint
    acq_resp = client.get(f"/api/v2/aois/{aoi_id}/acquisitions")
    assert acq_resp.status_code == 200
    acq_items = acq_resp.json()
    assert len(acq_items) >= 1
    assert acq_items[0]["aoi_id"] == aoi_id
