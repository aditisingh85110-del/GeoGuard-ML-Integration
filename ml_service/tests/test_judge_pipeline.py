"""
Unit tests for Hackathon Sprint Deliverables:
- BiTemporalChangeDetector (NDVI, reflectance difference, spectral candidate mask)
- TerrainContextAnalyzer (Horn slope calculation, geomorphic hazard classification)
- HonestPredictionEngine (Honest status system, OOD detection, alert suppression)
- FastAPI /api/v2/monitoring/judge-demo and /api/v2/monitoring/judge-evaluate endpoints
"""

import os
import sys
import numpy as np
import pytest
from fastapi.testclient import TestClient

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
for p in [current_dir, parent_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml_service.app import app
from ml_service.src.bi_temporal_change import BiTemporalChangeDetector
from ml_service.src.terrain import TerrainContextAnalyzer, classify_slope_hazard
from ml_service.src.judge_pipeline import HonestPredictionEngine


def test_bi_temporal_change_detector_calculations():
    detector = BiTemporalChangeDetector(ndvi_drop_threshold=-0.15, visible_gain_threshold=0.03)

    # 128x128 4-band rasters
    pre_raster = np.zeros((128, 128, 4), dtype=np.float32)
    # Healthy pre-event vegetation: B04=0.05, B08=0.50 -> NDVI ~ (0.5-0.05)/(0.5+0.05) = 0.818
    pre_raster[:, :, 2] = 0.05
    pre_raster[:, :, 3] = 0.50

    post_raster = pre_raster.copy()
    # Apply synthetic landslide scar to center 50x50 block
    # Stripped vegetation: B04=0.20, B08=0.15 -> NDVI ~ (0.15-0.20)/(0.15+0.20) = -0.143 (ΔNDVI = -0.96)
    # Increased visible: B02, B03, B04 rise by ~0.15
    post_raster[40:90, 40:90, :3] += 0.15
    post_raster[40:90, 40:90, 3] = 0.15

    res = detector.analyze_pair(
        pre_raster=pre_raster,
        post_raster=post_raster,
        pre_date="2021-02-05T00:00:00Z",
        post_date="2021-02-10T00:00:00Z",
    )

    assert res["status"] == "SIGNIFICANT_SPECTRAL_CHANGE_DETECTED"
    assert res["spatial_change_metrics"]["is_significant_change"] is True
    # 50x50 = 2500 pixels
    assert res["spatial_change_metrics"]["candidate_change_pixel_count"] == 2500
    assert res["ndvi_analysis"]["mean_delta_ndvi"] < 0.0


def test_bi_temporal_no_change_detection():
    detector = BiTemporalChangeDetector()
    # Identical rasters (stable terrain across passes)
    pre_raster = np.ones((128, 128, 4), dtype=np.float32) * 0.10
    pre_raster[:, :, 3] = 0.40
    post_raster = pre_raster.copy()

    res = detector.analyze_pair(
        pre_raster=pre_raster,
        post_raster=post_raster,
        pre_date="2021-02-05T00:00:00Z",
        post_date="2021-02-10T00:00:00Z",
    )
    assert res["status"] == "NO_SIGNIFICANT_CHANGE"
    assert res["spatial_change_metrics"]["is_significant_change"] is False
    assert res["spatial_change_metrics"]["candidate_change_pixel_count"] == 0


def test_terrain_context_analyzer():
    analyzer = TerrainContextAnalyzer(default_pixel_spacing_m=10.0)

    # Flat grid
    dem_flat = np.ones((128, 128), dtype=np.float32) * 500.0
    res_flat = analyzer.analyze_terrain([76.9, 12.0, 77.0, 12.1], dem_flat)
    assert res_flat["status"] == "AVAILABLE"
    assert res_flat["slope_context"]["mean_slope_degrees"] == 0.0
    assert res_flat["slope_context"]["hazard_classification"] == "FLAT_LOW_INITIATION_HAZARD"

    # Steep inclined plane: dz/dy = 10m per 10m pixel -> slope = 45 deg
    y_coords, _ = np.mgrid[0:128, 0:128]
    dem_steep = (y_coords * 10.0).astype(np.float32)
    res_steep = analyzer.analyze_terrain([79.7, 30.4, 79.8, 30.5], dem_steep)
    assert res_steep["status"] == "AVAILABLE"
    assert abs(res_steep["slope_context"]["mean_slope_degrees"] - 45.0) < 1.0
    assert "STEEP" in res_steep["slope_context"]["hazard_classification"]

    # Unavailable fallback
    res_unavail = analyzer.analyze_terrain([79.7, 30.4, 79.8, 30.5], None)
    assert res_unavail["status"] == "UNAVAILABLE"
    assert res_unavail["slope_context"]["mean_slope_degrees"] is None


def test_honest_prediction_engine_status():
    engine = HonestPredictionEngine()

    # Case A: Cloudy input -> INSUFFICIENT_DATA
    dummy_raster = np.ones((128, 128, 4), dtype=np.float32) * 0.1
    res_cloud = engine.evaluate_aoi_observation(
        aoi_name="Cloud_Test",
        aoi_bbox=[79.0, 30.0, 79.1, 30.1],
        post_raster=dummy_raster,
        post_date="2021-02-10T00:00:00Z",
        post_acquisition_id="ACQ_CLOUD",
        cloud_cover_percent=85.0,  # Exceeds max cloud threshold
    )
    assert res_cloud["final_candidate_status"] == "INSUFFICIENT_DATA"

    # Case B: Single-image Indian high-biomass vegetation -> OUT_OF_DISTRIBUTION & alert suppressed
    veg_raster = np.zeros((128, 128, 4), dtype=np.float32)
    veg_raster[:, :, 0] = 0.04
    veg_raster[:, :, 1] = 0.06
    veg_raster[:, :, 2] = 0.05
    veg_raster[:, :, 3] = 0.50  # High NIR

    res_veg = engine.evaluate_aoi_observation(
        aoi_name="Veg_Test",
        aoi_bbox=[76.1, 11.5, 76.2, 11.6],
        post_raster=veg_raster,
        post_date="2024-08-14T00:00:00Z",
        post_acquisition_id="ACQ_VEG",
        cloud_cover_percent=2.0,
        pre_raster=None,  # Single-image only
    )
    assert res_veg["final_candidate_status"] == "OUT_OF_DISTRIBUTION"
    assert res_veg["cnn_inference_diagnostics"]["out_of_distribution_detected"] is True
    assert res_veg["cnn_inference_diagnostics"]["is_autonomous_alert_suppressed"] is True


def test_judge_demo_api_endpoints():
    with TestClient(app) as client:
        # 1. GET /api/v2/monitoring/judge-demo
        r_demo = client.get("/api/v2/monitoring/judge-demo")
        assert r_demo.status_code == 200
        data = r_demo.json()
        assert "aoi_metadata" in data
        assert "final_candidate_status" in data
        assert "status_explanation" in data
        assert "data_provenance" in data
        assert "cnn_inference_diagnostics" in data
        assert "bi_temporal_change_detection" in data
        assert "terrain_topography_context" in data
        assert data["final_candidate_status"] == "HIGH_PRIORITY_REVIEW"
        assert data["risk_screening_status"] == "HIGH_PRIORITY_REVIEW"
        assert "candidate_alert" in data

        # 2. POST /api/v2/monitoring/judge-evaluate
        r_eval = client.post("/api/v2/monitoring/judge-evaluate")
        assert r_eval.status_code == 200
        eval_data = r_eval.json()
        assert "final_candidate_status" in eval_data
        assert "cnn_inference_diagnostics" in eval_data
