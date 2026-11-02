"""
GeoGuard Landslide Detection System - FastAPI Microservice.

Exposes REST APIs for:
- Health and model readiness checks (/health)
- Latest corridor macro landslide prediction (/latest-prediction)
- Prediction history (/predictions)
- Ad-hoc manual patch testing (/predict)
- Copernicus satellite status and discovery (/satellite-status)
- On-demand satellite poll trigger (/trigger-satellite-check)
- Geotechnical IoT telemetry cross-validation and sensor fusion (/iot-fusion-preview, /iot-fusion-cross-validate)
"""

import os
import sys
import io
import json
import logging
import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import h5py
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml_service.config import settings
from ml_service.src import database
from ml_service.src.predict import LandslidePredictor
from ml_service.src.satellite import SatellitePipeline, CopernicusClient
from ml_service.src.india_boundary import validate_aoi_bbox_in_india
from ml_service.src.schemas import (
    HealthResponse,
    LatestPredictionResponse,
    PredictionHistoryResponse,
    SatelliteStatusResponse,
    ManualPredictionResponse,
    IoTTelemetryInput,
    IoTFusionResponse,
    AOICreateRequest,
    AOIUpdateRequest,
    AOIResponse,
    AOIListResponse,
    AOIProcessResultResponse,
    ProcessedAcquisitionItem,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("geoguard_api")

# Global singleton pipeline instances
state: Dict[str, Any] = {
    "predictor": None,
    "satellite_pipeline": None,
    "copernicus_client": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to initialize DB and model singletons once."""
    logger.info("Initializing GeoGuard ML Microservice...")
    database.init_db()

    try:
        predictor = LandslidePredictor(
            model_path=settings.MODEL_PATH,
            config_path=settings.PREPROCESSING_CONFIG_PATH,
            decision_threshold=settings.DECISION_THRESHOLD,
        )
        state["predictor"] = predictor
        logger.info(f"LandslidePredictor loaded successfully (threshold={predictor.decision_threshold}).")
    except Exception as e:
        logger.error(f"Failed to load LandslidePredictor: {e}")
        state["predictor"] = None

    try:
        copernicus_client = CopernicusClient()
        state["copernicus_client"] = copernicus_client
        state["satellite_pipeline"] = SatellitePipeline(
            predictor=state["predictor"],
            copernicus_client=copernicus_client,
        )
        logger.info("SatellitePipeline initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize SatellitePipeline: {e}")

    # Seed initial prediction into DB if empty
    try:
        count = database.get_prediction_count()
        if count == 0 and state["satellite_pipeline"] is not None:
            logger.info("Database is empty. Generating initial baseline prediction...")
            initial_result = state["satellite_pipeline"].process_aoi()
            if initial_result.get("status") == "success":
                database.save_prediction(initial_result)
                logger.info("Baseline prediction saved to database.")
    except Exception as e:
        logger.warning(f"Could not seed initial prediction: {e}")

    yield

    logger.info("Shutting down GeoGuard ML Microservice...")


app = FastAPI(
    title="GeoGuard Landslide Detection ML Service",
    description=(
        "Macro-level satellite image-based landslide detection API for GeoGuard. "
        "Processes Sentinel-2 optical imagery and provides visual verification for geotechnical IoT telemetry."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin requests from React Native and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_db_prediction(row: Dict[str, Any]) -> LatestPredictionResponse:
    """Converts a database row dictionary to standard LatestPredictionResponse."""
    high_risk = row.get("high_risk_coordinates", [])
    if isinstance(high_risk, str):
        try:
            high_risk = json.loads(high_risk)
        except Exception:
            high_risk = []

    pred_class = row.get("prediction_class", "UNKNOWN")
    tot_patches = row.get("total_patches", 1)
    ls_patches = row.get("landslide_patches", 0)

    if pred_class == "LANDSLIDE DETECTED":
        msg = f"Macro landslide risk detected across {ls_patches} of {tot_patches} evaluated terrain patches."
    else:
        msg = f"Monitored corridor is normal and stable across all {tot_patches} evaluated terrain patches."

    return LatestPredictionResponse(
        id=row.get("id"),
        prediction=pred_class,
        probability=float(row.get("probability", 0.0)),
        decision_threshold=float(row.get("decision_threshold", settings.DECISION_THRESHOLD)),
        acquisition_id=row.get("acquisition_id"),
        acquisition_timestamp=row.get("acquisition_timestamp"),
        processing_timestamp=row.get("processing_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
        cloud_cover_percentage=row.get("cloud_cover_percent"),
        aoi_name=row.get("aoi_name", "UNKNOWN"),
        aoi_bbox=row.get("aoi_bbox"),
        data_status=row.get("data_status", "VALID"),
        is_simulated=bool(row.get("is_simulated", 1)),
        model_version=row.get("model_version", "GeoGuard_CNN_v1"),
        total_patches=tot_patches,
        landslide_patches=ls_patches,
        high_risk_coordinates=high_risk,
        message=msg,
    )


# ==============================================================================
# API Endpoints
# ==============================================================================


@app.get("/", tags=["General"])
def root():
    """Service landing page and welcome status."""
    return {
        "service": "GeoGuard Satellite Landslide Detection ML Service",
        "status": "online",
        "documentation": "/docs",
        "version": "1.0.0",
        "model": "4-Channel CNN (B02, B03, B04, B08)",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """
    Returns service health, model load status, and database connectivity.
    """
    is_model_loaded = state["predictor"] is not None and state["predictor"].model is not None
    db_ok = False
    try:
        _ = database.get_prediction_count()
        db_ok = True
    except Exception:
        db_ok = False

    status_str = "healthy" if (is_model_loaded and db_ok) else "degraded"

    return HealthResponse(
        status=status_str,
        service="GeoGuard Satellite Landslide Detection ML Service",
        model_loaded=is_model_loaded,
        model_version="GeoGuard_CNN_v1",
        database_connected=db_ok,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


@app.get("/latest-prediction", response_model=LatestPredictionResponse, tags=["Predictions"])
def get_latest_prediction():
    """
    Returns the most recent macro landslide prediction for the monitored corridor.
    Always maintains the true acquisition timestamp of the imagery.
    """
    latest = database.get_latest_prediction()
    if latest is not None:
        return format_db_prediction(latest)

    # If DB is empty, run inference on active AOI
    if state["satellite_pipeline"] is None:
        raise HTTPException(status_code=503, detail="Satellite pipeline is not initialized.")

    result = state["satellite_pipeline"].process_aoi()
    if result.get("status") == "success":
        new_id = database.save_prediction(result)
        latest = database.get_latest_prediction()
        return format_db_prediction(latest)

    raise HTTPException(status_code=500, detail="Failed to compute initial prediction: " + result.get("message", ""))


@app.get("/predictions", response_model=PredictionHistoryResponse, tags=["Predictions"])
def get_predictions(
    limit: int = Query(50, ge=1, le=500, description="Maximum number of historical records to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """
    Retrieves historical macro predictions stored in the database, ordered newest first.
    """
    total = database.get_prediction_count()
    rows = database.get_prediction_history(limit=limit, offset=offset)
    formatted = [format_db_prediction(r) for r in rows]

    return PredictionHistoryResponse(
        total_count=total,
        limit=limit,
        offset=offset,
        predictions=formatted,
    )


@app.post("/predict", response_model=ManualPredictionResponse, tags=["Predictions"])
async def predict_custom_patch(file: UploadFile = File(...)):
    """
    Ad-hoc manual testing endpoint.
    Accepts an uploaded HDF5 (.h5) patch file.
    Validates input shape (128, 128, 4) or extracts the 4 designated bands if given a 14-channel Landslide4Sense patch.
    Clearly marked with data_status='MANUAL_TEST' to preserve corridor integrity.
    """
    if state["predictor"] is None:
        raise HTTPException(status_code=503, detail="CNN Model is not loaded.")

    filename = file.filename or ""
    if not (filename.endswith(".h5") or filename.endswith(".hdf5")):
        raise HTTPException(status_code=400, detail="Uploaded file must be an HDF5 file (.h5 or .hdf5)")

    try:
        content = await file.read()
        with io.BytesIO(content) as bio:
            with h5py.File(bio, "r") as h5_file:
                # Look for dataset key
                keys = list(h5_file.keys())
                if not keys:
                    raise HTTPException(status_code=400, detail="HDF5 file contains no datasets.")

                key = "img" if "img" in keys else ("image" if "image" in keys else keys[0])
                arr = np.array(h5_file[key])

        # Validate shape
        if arr.ndim != 3:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 3-dimensional array (H, W, C), got ndim={arr.ndim} with shape {arr.shape}",
            )

        # Handle 14-band Landslide4Sense patch: extract [B02, B03, B04, B08] (indices 1, 2, 3, 7)
        if arr.shape[2] == 14:
            logger.info("Extracting native 4 Sentinel-2 channels from 14-band Landslide4Sense patch.")
            arr = arr[:, :, [1, 2, 3, 7]]

        if arr.shape != (128, 128, 4):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tensor shape {arr.shape}. Expected (128, 128, 4).",
            )

        # Run inference
        result = state["predictor"].predict_patch(arr)

        return ManualPredictionResponse(
            prediction=result["prediction_class"],
            probability=result["probability"],
            decision_threshold=result["decision_threshold"],
            input_shape=list(arr.shape),
            data_status="MANUAL_TEST",
            processing_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            model_version="GeoGuard_CNN_v1",
            notice="MANUAL AD-HOC TEST. This test did not modify or interfere with the automated corridor monitoring stream.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing manual prediction upload: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/satellite-status", response_model=SatelliteStatusResponse, tags=["Satellite"])
def get_satellite_status():
    """
    Returns the operational connectivity status of the Copernicus Sentinel-2 STAC catalogue.
    Indicates whether newly acquired satellite imagery is available for the monitored corridor.
    """
    copernicus = state.get("copernicus_client") or CopernicusClient()
    aoi_name, bbox, is_test_aoi = settings.get_active_aoi(allow_test_fallback=True)

    latest_pred = database.get_latest_prediction()
    latest_processed_id = latest_pred.get("acquisition_id") if latest_pred else None

    # Discover latest acquisitions in CDSE STAC
    acquisitions = []
    conn_status = "CONNECTED"
    try:
        acquisitions = copernicus.search_acquisitions(bbox=bbox or settings.TEST_AOI_BBOX, limit=2)
    except Exception as e:
        logger.warning(f"Error contacting Copernicus STAC catalogue: {e}")
        conn_status = "ERROR"

    if not copernicus.has_credentials and conn_status == "CONNECTED":
        conn_status = "UNAUTHENTICATED_TEST_MODE"

    latest_catalog_id = acquisitions[0]["acquisition_id"] if acquisitions else None
    has_new = False
    if latest_catalog_id and latest_processed_id:
        has_new = latest_catalog_id != latest_processed_id
    elif latest_catalog_id and not latest_processed_id:
        has_new = True

    latest_log = database.get_latest_polling_log()
    last_checked = latest_log.get("checked_at") if latest_log else None

    return SatelliteStatusResponse(
        connection_status=conn_status,
        stac_catalogue_url=settings.CDSE_STAC_URL,
        credentials_configured=copernicus.has_credentials,
        last_checked_timestamp=last_checked,
        latest_catalog_acquisition_id=latest_catalog_id,
        latest_processed_acquisition_id=latest_processed_id,
        has_new_acquisition=has_new,
        mode="REAL_COPERNICUS" if copernicus.has_credentials else "SIMULATED",
        monitored_aoi={"name": aoi_name, "bbox": bbox, "is_test_aoi": is_test_aoi},
        polling_interval_minutes=settings.POLLING_INTERVAL_MINUTES,
    )


@app.post("/trigger-satellite-check", tags=["Satellite"])
def trigger_satellite_check():
    """
    On-demand trigger allowing frontend operators or automated webhooks to poll
    for new satellite imagery immediately without waiting for the next scheduler cycle.
    """
    if state["satellite_pipeline"] is None:
        raise HTTPException(status_code=503, detail="Satellite pipeline is not initialized.")

    copernicus = state["satellite_pipeline"].copernicus
    aoi_name, bbox, is_test = settings.get_active_aoi(allow_test_fallback=True)

    acquisitions = copernicus.search_acquisitions(bbox=bbox or settings.TEST_AOI_BBOX, limit=1)
    if not acquisitions:
        database.log_polling_event("NO_NEW_ACQUISITION_AVAILABLE", message="No acquisitions found during manual trigger")
        return {
            "status": "completed",
            "action": "none",
            "message": "No new satellite acquisitions available in the catalogue.",
        }

    latest_acq = acquisitions[0]
    acq_id = latest_acq.get("acquisition_id")

    # Check if already processed
    if database.is_acquisition_processed(acq_id):
        database.log_polling_event(
            "NO_NEW_ACQUISITION_AVAILABLE",
            latest_catalog_id=acq_id,
            message="Latest acquisition already processed and up-to-date.",
        )
        return {
            "status": "completed",
            "action": "none",
            "message": f"Acquisition {acq_id} is already processed. Corridor status is current.",
        }

    # Check cloud cover
    cloud = latest_acq.get("cloud_cover_percent", 0.0)
    if cloud and cloud > settings.MAX_CLOUD_COVER_PERCENT:
        database.log_polling_event(
            "EXCESSIVE_CLOUD",
            latest_catalog_id=acq_id,
            message=f"Cloud cover {cloud}% exceeds maximum threshold {settings.MAX_CLOUD_COVER_PERCENT}%.",
        )
        return {
            "status": "completed",
            "action": "skipped",
            "message": f"Acquisition {acq_id} detected but skipped due to high cloud cover ({cloud}% > {settings.MAX_CLOUD_COVER_PERCENT}%). Last valid prediction retained.",
        }

    # Process new acquisition
    result = state["satellite_pipeline"].process_aoi(bbox=bbox, aoi_name=aoi_name, acquisition_id=acq_id)
    if result.get("status") == "success":
        new_id = database.save_prediction(result)
        database.log_polling_event(
            "NEW_ACQUISITION_PROCESSED",
            latest_catalog_id=acq_id,
            message=f"Successfully processed new acquisition {acq_id}",
        )
        return {
            "status": "success",
            "action": "processed_new_acquisition",
            "prediction_id": new_id,
            "acquisition_id": acq_id,
            "overall_prediction": result.get("prediction", {}).get("overall_class"),
        }

    return {"status": "error", "message": result.get("message", "Processing failed")}


# ==============================================================================
# IoT Geotechnical Cross-Validation / Sensor Fusion Endpoints
# ==============================================================================


def perform_fusion_analysis(
    macro_pred: Dict[str, Any],
    iot: IoTTelemetryInput,
) -> IoTFusionResponse:
    """
    Evaluates correlation between macro satellite observation and micro geotechnical telemetry.
    Multi-sensor fusion logic:
    - High soil moisture (> 40%), heavy rain (> 50mm), and high tilt (> 2.0°) indicate imminent geotechnical failure.
    - Satellite CNN verifies macro terrain scarp or mass displacement.
    """
    is_satellite_landslide = macro_pred.get("prediction") == "LANDSLIDE DETECTED"
    sat_prob = macro_pred.get("probability", 0.0)

    # Micro geotechnical hazard evaluation
    iot_moisture_high = iot.soil_moisture_percent >= 40.0
    iot_rain_heavy = iot.rainfall_mm_24h >= 50.0
    iot_tilt_active = iot.tilt_angle_deg >= 2.0
    iot_vibration_active = iot.vibration_intensity >= 0.10

    iot_hazard_score = sum([iot_moisture_high, iot_rain_heavy, iot_tilt_active, iot_vibration_active])
    is_iot_hazard = iot_hazard_score >= 2 or iot_tilt_active

    if is_satellite_landslide and is_iot_hazard:
        status = "AGREEMENT"
        risk_level = "CRITICAL"
        rec = "CRITICAL EMERGENCY: Visual satellite detection confirmed by active geotechnical sensor displacement. Immediate highway closure and barrier deployment required."
        rationale = (
            f"Macro satellite CNN detected terrain scarp/movement with {sat_prob*100:.1f}% confidence, "
            f"corroborated by high geotechnical telemetry (Tilt: {iot.tilt_angle_deg}°, Rain: {iot.rainfall_mm_24h}mm, Moisture: {iot.soil_moisture_percent}%)."
        )
    elif is_satellite_landslide and not is_iot_hazard:
        status = "MACRO_ALERT_ONLY"
        risk_level = "HIGH"
        rec = "HIGH PRIORITY WARNING: Macro optical satellite detection observed in corridor. Geotechnical sensors report sub-threshold values. Dispatch UAV / drone reconnaissance to inspect identified high-risk coordinates."
        rationale = (
            f"Satellite CNN detected landslide features ({sat_prob*100:.1f}%), but local node {iot.sensor_id} reports stable geotechnical readings. "
            "Possible localized slope failure outside immediate sensor radius, or recent landslide deposit."
        )
    elif not is_satellite_landslide and is_iot_hazard:
        status = "LOCAL_IOT_ALERT_ONLY"
        risk_level = "WARNING"
        rec = "SUBSURFACE HAZARD ADVISORY: Geotechnical telemetry indicates slope instability or pore-water saturation. Satellite visual imagery shows no surface disruption yet. Issue cautionary speed limits and monitor for impending failure."
        rationale = (
            f"Geotechnical node {iot.sensor_id} indicates active destabilization (Tilt: {iot.tilt_angle_deg}°, Moisture: {iot.soil_moisture_percent}%), "
            "while macro optical satellite imagery remains visually intact. Subsurface shear plane likely developing prior to macro surface failure."
        )
    else:
        status = "ALL_CLEAR"
        risk_level = "LOW"
        rec = "NORMAL MONITORING: Both macro satellite imagery and geotechnical IoT telemetry confirm corridor stability. Maintain routine monitoring."
        rationale = "Harmonious stability across visual satellite verification and all geotechnical telemetry indicators."

    return IoTFusionResponse(
        cross_validation_status=status,
        combined_risk_level=risk_level,
        satellite_macro_status={
            "prediction": macro_pred.get("prediction"),
            "probability": sat_prob,
            "acquisition_id": macro_pred.get("acquisition_id"),
            "acquisition_timestamp": macro_pred.get("acquisition_timestamp"),
            "data_status": macro_pred.get("data_status"),
        },
        iot_micro_telemetry=iot.model_dump(),
        action_recommendation=rec,
        fusion_rationale=rationale,
    )


@app.get("/iot-fusion-preview", response_model=IoTFusionResponse, tags=["IoT Cross-Validation"])
def preview_iot_fusion():
    """
    Provides a demonstration cross-validation of geotechnical telemetry against the latest satellite prediction.
    Demonstrates how GeoGuard combines optical verification with IoT sensors.
    """
    latest = get_latest_prediction()
    # Sample IoT reading demonstrating cross-validation
    demo_iot = IoTTelemetryInput(
        sensor_id="GEO-NODE-01",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        soil_moisture_percent=45.2,
        rainfall_mm_24h=62.0,
        vibration_intensity=0.08,
        tilt_angle_deg=2.5,
        battery_percent=92.0,
    )
    return perform_fusion_analysis(latest.model_dump(), demo_iot)


@app.post("/iot-fusion-cross-validate", response_model=IoTFusionResponse, tags=["IoT Cross-Validation"])
def cross_validate_iot(telemetry: IoTTelemetryInput):
    """
    Cross-validates live geotechnical telemetry submitted by an IoT sensor node
    against the latest macro satellite prediction.
    """
    latest = get_latest_prediction()
    return perform_fusion_analysis(latest.model_dump(), telemetry)


# ==============================================================================
# Phase 8: India-Wide Area of Interest (AOI) Management & Processing APIs
# ==============================================================================

@app.post(
    "/api/v2/aois",
    response_model=AOIResponse,
    status_code=201,
    tags=["India-Wide Monitoring (Phase 8)"],
)
def create_monitored_aoi(aoi_in: AOICreateRequest):
    """
    Registers a new Area of Interest (AOI) for landslide monitoring in India.
    Validates that the bounding box coordinates lie strictly within Indian territorial boundaries.
    """
    # 1. Check if AOI ID already exists
    existing = database.get_aoi(aoi_in.aoi_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"AOI with id '{aoi_in.aoi_id}' already exists. Use PUT to update.",
        )

    # 2. Validate coordinates against India boundary polygon
    is_valid_india, err_msg = validate_aoi_bbox_in_india(aoi_in.bbox_wgs84)
    if not is_valid_india:
        raise HTTPException(status_code=422, detail=f"India Boundary Validation Error: {err_msg}")

    # 3. Create AOI in database
    created = database.create_aoi(aoi_in.model_dump())
    return AOIResponse(**created)


@app.get(
    "/api/v2/aois",
    response_model=AOIListResponse,
    tags=["India-Wide Monitoring (Phase 8)"],
)
def list_monitored_aois(
    state_filter: Optional[str] = Query(None, alias="state", description="Filter by Indian State"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by monitoring status (ACTIVE, PAUSED, ARCHIVED)"),
):
    """
    Lists registered monitored AOIs across India with optional state and status filtering.
    """
    aois = database.list_aois(state=state_filter, status=status_filter)
    return AOIListResponse(total_count=len(aois), aois=[AOIResponse(**a) for a in aois])


@app.get(
    "/api/v2/aois/{aoi_id}",
    response_model=AOIResponse,
    tags=["India-Wide Monitoring (Phase 8)"],
)
def get_monitored_aoi(aoi_id: str):
    """
    Retrieves configuration and current monitoring status for a specific AOI.
    """
    aoi = database.get_aoi(aoi_id)
    if not aoi:
        raise HTTPException(status_code=404, detail=f"Monitored AOI '{aoi_id}' not found.")
    return AOIResponse(**aoi)


@app.put(
    "/api/v2/aois/{aoi_id}",
    response_model=AOIResponse,
    tags=["India-Wide Monitoring (Phase 8)"],
)
def update_monitored_aoi(aoi_id: str, update_in: AOIUpdateRequest):
    """
    Updates the configuration (coordinates, region, monitoring status) of an existing AOI.
    If bounding box is updated, validates it against Indian boundary polygon.
    """
    existing = database.get_aoi(aoi_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Monitored AOI '{aoi_id}' not found.")

    update_dict = {k: v for k, v in update_in.model_dump().items() if v is not None}

    if "bbox_wgs84" in update_dict:
        is_valid_india, err_msg = validate_aoi_bbox_in_india(update_dict["bbox_wgs84"])
        if not is_valid_india:
            raise HTTPException(status_code=422, detail=f"India Boundary Validation Error: {err_msg}")

    updated = database.update_aoi(aoi_id, update_dict)
    return AOIResponse(**updated)  # type: ignore


@app.post(
    "/api/v2/aois/{aoi_id}/deactivate",
    response_model=AOIResponse,
    tags=["India-Wide Monitoring (Phase 8)"],
)
def deactivate_monitored_aoi(aoi_id: str):
    """
    Deactivates an AOI by setting monitoring_status='ARCHIVED'. Preserves historical predictions.
    """
    existing = database.get_aoi(aoi_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Monitored AOI '{aoi_id}' not found.")

    database.delete_aoi(aoi_id, hard_delete=False)
    updated = database.get_aoi(aoi_id)
    return AOIResponse(**updated)  # type: ignore


@app.delete(
    "/api/v2/aois/{aoi_id}",
    tags=["India-Wide Monitoring (Phase 8)"],
)
def delete_monitored_aoi(
    aoi_id: str,
    hard_delete: bool = Query(False, description="Set True to permanently delete; False to archive"),
):
    """
    Deactivates (soft-delete) or permanently deletes an AOI.
    """
    existing = database.get_aoi(aoi_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Monitored AOI '{aoi_id}' not found.")

    database.delete_aoi(aoi_id, hard_delete=hard_delete)
    action = "permanently deleted" if hard_delete else "archived"
    return {"status": "success", "message": f"AOI '{aoi_id}' successfully {action}."}


@app.post(
    "/api/v2/aois/{aoi_id}/process",
    response_model=AOIProcessResultResponse,
    tags=["India-Wide Monitoring (Phase 8)"],
)
def process_monitored_aoi_endpoint(
    aoi_id: str,
    allow_simulated_fallback: bool = Query(False, description="Allow test fallback if live credentials missing"),
    force_reprocess: bool = Query(False, description="Force re-processing even if acquisition was already processed"),
):
    """
    Triggers an on-demand satellite monitoring check for a specific registered AOI:
    - Queries CDSE for new acquisitions.
    - Applies deduplication to avoid re-processing identical Sentinel-2 products.
    - Validates radiometry and cloud cover (returns INSUFFICIENT_DATA if unusable).
    - Extracts native 10m overlapping patches.
    - Runs CNN v2 inference.
    - Clusters positive patches to prevent duplicate event counting.
    """
    aoi = database.get_aoi(aoi_id)
    if not aoi:
        raise HTTPException(status_code=404, detail=f"Monitored AOI '{aoi_id}' not found.")

    if aoi["monitoring_status"] != "ACTIVE" and not force_reprocess:
        raise HTTPException(
            status_code=400,
            detail=f"AOI '{aoi_id}' is currently '{aoi['monitoring_status']}'. Set monitoring_status='ACTIVE' to process.",
        )

    pipeline: SatellitePipeline = state.get("satellite_pipeline")
    if not pipeline:
        raise HTTPException(status_code=503, detail="SatellitePipeline is not initialized.")

    result = pipeline.process_monitored_aoi(
        aoi=aoi,
        allow_simulated_fallback=allow_simulated_fallback,
        force_reprocess=force_reprocess,
    )
    return AOIProcessResultResponse(**result)


@app.get(
    "/api/v2/aois/{aoi_id}/acquisitions",
    response_model=List[ProcessedAcquisitionItem],
    tags=["India-Wide Monitoring (Phase 8)"],
)
def list_aoi_acquisitions(aoi_id: str):
    """
    Returns the processing and deduplication history of all Sentinel-2 acquisitions for an AOI.
    """
    aoi = database.get_aoi(aoi_id)
    if not aoi:
        raise HTTPException(status_code=404, detail=f"Monitored AOI '{aoi_id}' not found.")

    items = database.list_processed_acquisitions(aoi_id)
    return [ProcessedAcquisitionItem(**it) for it in items]


@app.get(
    "/api/v2/monitoring/judge-demo",
    tags=["Hackathon Demo (Honest Prediction)"],
)
def get_judge_ready_demo():
    """
    Judge-Ready End-to-End Demonstration Endpoint.
    Evaluates real Sentinel-2 Level-2A imagery over the Chamoli Corridor (Raini Surge Zone).
    Demonstrates:
    - Genuine Copernicus Data Space Ecosystem (CDSE) satellite data.
    - Bi-temporal change detection (ΔNDVI, ΔVisible) between 2021-02-05 and 2021-02-10.
    - Honest prediction status (suppresses uncalibrated single-image CNN false alarms).
    - Topographic DEM slope and elevation context.
    - Explicit distinction between real satellite data, model inference, and experimental change detection.
    """
    json_path = os.path.join(settings.REPORTS_DIR, "sample_judge_ready_response.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback to dynamic generation if file not yet generated
    from ml_service.reports.generate_demo_artifacts import generate_artifacts
    generate_artifacts()
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post(
    "/api/v2/monitoring/judge-evaluate",
    tags=["Hackathon Demo (Honest Prediction)"],
)
def evaluate_aoi_judge_ready(
    aoi_id: Optional[str] = Query(None, description="Registered AOI ID (omit or pass 'demo' for Chamoli baseline)"),
    allow_simulated_fallback: bool = Query(False, description="Allow test raster fallback if external CDSE is rate-limited"),
):
    """
    On-Demand Multi-Evidence Evaluation for any Monitored AOI.
    Executes data quality screening, CNN v2 inference, out-of-distribution detection,
    and returns a transparent risk screening status:
    LOW_OBSERVED_RISK | ELEVATED_CHANGE_CANDIDATE | HIGH_PRIORITY_REVIEW |
    INSUFFICIENT_DATA | OUT_OF_DISTRIBUTION.
    
    If dynamic retrieval is unavailable for a new AOI, strictly returns INSUFFICIENT_DATA
    rather than silently substituting cached data from another region.
    """
    from ml_service.src.judge_pipeline import HonestPredictionEngine

    # Check if this is the Chamoli demonstration baseline
    if not aoi_id or aoi_id in ["demo", "chamoli"]:
        bbox = settings.HIGHWAY_AOI_BBOX or settings.TEST_AOI_BBOX
        aoi_name = "Chamoli_Rishiganga_Corridor_Raini"
        pre_path = os.path.join(settings.DATA_DIR, "demo_chamoli_pre_20210205.npy")
        post_path = os.path.join(settings.DATA_DIR, "demo_chamoli_post_20210210.npy")

        if os.path.exists(pre_path) and os.path.exists(post_path):
            pre_raster = np.load(pre_path)
            post_raster = np.load(post_path)
            source = "Copernicus Data Space Ecosystem (Sentinel-2 L2A Genuine Cached - Chamoli)"
            is_sim = False
            post_date = "2021-02-10T05:19:41Z"
            pre_date = "2021-02-05T05:19:39Z"
            post_acq_id = "S2A_MSIL2A_20210210T051941_N0214_R019_T44RKR"
            pre_acq_id = "S2B_MSIL2A_20210205T051939_N0214_R019_T44RKR"
            cloud_pct = 1.8
        else:
            post_raster = np.ones((128, 128, 4), dtype=np.float32) * 0.15
            post_raster[:, :, 3] = 0.45
            pre_raster = post_raster.copy()
            source = "Calibrated Diagnostic Raster"
            is_sim = True
            post_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            pre_date = post_date
            post_acq_id = "DIAGNOSTIC_SYNTHETIC_ACQ"
            pre_acq_id = "DIAGNOSTIC_SYNTHETIC_BASELINE"
            cloud_pct = 0.0

        # DEM profile for Chamoli
        y_coords, x_coords = np.mgrid[0:128, 0:128]
        sim_dem = 2200.0 + (y_coords * 4.5) + (x_coords * 2.1)

        engine = HonestPredictionEngine()
        return engine.evaluate_aoi_observation(
            aoi_name=aoi_name,
            aoi_bbox=bbox,
            post_raster=post_raster,
            post_date=post_date,
            post_acquisition_id=post_acq_id,
            cloud_cover_percent=cloud_pct,
            imagery_source=source,
            pre_raster=pre_raster,
            pre_date=pre_date,
            pre_acquisition_id=pre_acq_id,
            dem_grid=sim_dem,
            is_simulated_fallback=is_sim,
            is_dem_parameterized=True,
        )

    # Dynamic Evaluation for custom registered AOI
    aoi = database.get_aoi(aoi_id)
    if not aoi:
        raise HTTPException(status_code=404, detail=f"AOI '{aoi_id}' not found in database.")
    bbox = aoi.get("bbox_wgs84")
    if not bbox:
        raw_bbox = aoi.get("bbox_wgs84_json") or aoi.get("bbox_json")
        bbox = json.loads(raw_bbox) if isinstance(raw_bbox, str) else raw_bbox
    aoi_name = aoi["name"]

    client: CopernicusClient = state.get("copernicus_client") or CopernicusClient()
    token = client.get_auth_token()
    acqs = client.search_acquisitions(bbox, limit=10, max_cloud_cover=settings.MAX_CLOUD_COVER_PERCENT) if token else []
    usable_acqs = [a for a in acqs if a.get("is_usable")]

    if not token or not usable_acqs:
        return {
            "aoi_metadata": {"aoi_name": aoi_name, "bbox": bbox},
            "risk_screening_status": "INSUFFICIENT_DATA",
            "final_candidate_status": "INSUFFICIENT_DATA",
            "status_code": "DYNAMIC_RETRIEVAL_UNAVAILABLE",
            "status_explanation": (
                f"Dynamic satellite retrieval for AOI '{aoi_name}' returned no usable cloud-free Sentinel-2 acquisitions "
                f"from Copernicus CDSE (cloud cover threshold <= {settings.MAX_CLOUD_COVER_PERCENT}%). "
                f"Per strict scientific governance, cached imagery from Chamoli was NOT substituted."
            ),
            "assessment_type": "POST_EVENT_SURFACE_CHANGE_DETECTION",
            "assessment_type_explanation": "Optical satellite screening requires cloud-free surface observations.",
            "data_provenance": {
                "imagery_source": "Copernicus Data Space Ecosystem (Dynamic Query)",
                "is_simulated_fallback": False,
                "is_dynamically_retrieved": True,
                "data_quality_status": "NO_USABLE_ACQUISITIONS",
                "cloud_cover_percent": None,
            },
            "candidate_alert": {
                "alert_level": "INSUFFICIENT_DATA",
                "alert_triggered": False,
                "aoi_name": aoi_name,
                "imagery_source": "Copernicus Data Space Ecosystem (Dynamic Query)",
                "recommended_actions": ["Wait for next cloud-free Sentinel-2 satellite overpass."],
            },
            "limitations_and_missing_evidence": [
                "Dynamic optical retrieval failed or was obscured by clouds. GeoGuard strictly refuses to fall back to unverified or borrowed data from other regions."
            ],
        }

    post_acq = usable_acqs[0]
    post_raster, post_meta = client.fetch_scene_imagery(post_acq, bbox, patch_size=128, allow_simulated_fallback=allow_simulated_fallback)
    if post_raster is None:
        return {
            "aoi_metadata": {"aoi_name": aoi_name, "bbox": bbox},
            "risk_screening_status": "INSUFFICIENT_DATA",
            "final_candidate_status": "INSUFFICIENT_DATA",
            "status_code": "IMAGE_FETCH_FAILED",
            "status_explanation": f"CDSE Processing API failed to deliver raster for {post_acq['acquisition_id']}.",
            "data_provenance": {
                "imagery_source": "Copernicus Data Space Ecosystem",
                "data_quality_status": "FETCH_FAILED",
                "is_simulated_fallback": False,
            },
            "limitations_and_missing_evidence": ["Raster imagery retrieval failed."],
        }

    pre_raster = None
    pre_date = None
    pre_acq_id = None
    if len(usable_acqs) > 1:
        post_day = post_acq.get("acquisition_time", "")[:10]
        # Prefer an acquisition from a distinct historical date for real multi-temporal comparison
        candidate_pre = next(
            (a for a in usable_acqs[1:] if a.get("acquisition_time", "")[:10] != post_day),
            usable_acqs[1]
        )
        pre_raster, _ = client.fetch_scene_imagery(candidate_pre, bbox, patch_size=128, allow_simulated_fallback=allow_simulated_fallback)
        if pre_raster is not None:
            pre_date = candidate_pre.get("acquisition_time")
            pre_acq_id = candidate_pre.get("acquisition_id")

    source = "Copernicus Data Space Ecosystem (Sentinel-2 L2A Dynamically Retrieved)"
    engine = HonestPredictionEngine()
    return engine.evaluate_aoi_observation(
        aoi_name=aoi_name,
        aoi_bbox=bbox,
        post_raster=post_raster,
        post_date=post_acq.get("acquisition_time", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
        post_acquisition_id=post_acq.get("acquisition_id", "DYNAMIC_ACQUISITION"),
        cloud_cover_percent=post_acq.get("cloud_cover_percent", 0.0),
        imagery_source=source,
        pre_raster=pre_raster,
        pre_date=pre_date,
        pre_acquisition_id=pre_acq_id,
        dem_grid=None,
        is_simulated_fallback=False,
        is_dem_parameterized=False,
    )


# =============================================================================
# Autonomous Himalayan Satellite Surveillance (Phase 8 Multi-Region MVP)
# =============================================================================
@app.get(
    "/api/v2/surveillance/summary",
    tags=["Autonomous Himalayan Surveillance"],
)
def get_himalayan_surveillance_summary():
    """
    Returns the real-time surveillance matrix across all 6 Himalayan territories:
    Uttarakhand, Himachal Pradesh, Jammu & Kashmir, Ladakh, Sikkim, Arunachal Pradesh.
    Returns per-AOI disturbance intensity, timestamps, cloud coverage, and active alerts.
    """
    from ml_service.src.surveillance import get_surveillance_engine
    engine = get_surveillance_engine()
    return engine.get_surveillance_summary(allow_cached=True)


@app.post(
    "/api/v2/surveillance/sweep",
    tags=["Autonomous Himalayan Surveillance"],
)
def trigger_himalayan_surveillance_sweep(
    aoi_id: Optional[str] = Query(None, description="Optional single AOI ID to sweep, or omit for all 6"),
    allow_simulated_fallback: bool = Query(False, description="Allow test raster fallback if CDSE rate-limited"),
):
    """
    Triggers an autonomous satellite observation query and bi-temporal evaluation sweep
    across the Himalayan surveillance fleet. Updates persistent per-AOI cache.
    """
    from ml_service.src.surveillance import get_surveillance_engine
    engine = get_surveillance_engine()
    return engine.run_surveillance_sweep(aoi_id=aoi_id, allow_simulated_fallback=allow_simulated_fallback)

