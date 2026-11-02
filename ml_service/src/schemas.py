"""
GeoGuard Landslide Detection System - Pydantic Request & Response Schemas.

Defines strongly-typed schemas for:
- Service health & model readiness
- Live satellite inference & macro aggregation
- Historical prediction queries
- Manual patch testing
- Satellite connection status
- Geotechnical IoT telemetry cross-validation / sensor fusion
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    service: str = Field(..., examples=["GeoGuard Satellite Landslide Detection ML Service"])
    model_loaded: bool = Field(..., examples=[True])
    model_version: str = Field(..., examples=["GeoGuard_CNN_v1"])
    database_connected: bool = Field(..., examples=[True])
    timestamp: str


class CoordinatePoint(BaseModel):
    lat: float = Field(..., examples=[12.05])
    lon: float = Field(..., examples=[77.00])
    prob: Optional[float] = Field(None, examples=[0.895])


class PatchResult(BaseModel):
    patch_index: int
    center_coords: Dict[str, float]
    bbox: List[float]
    prediction_class: str
    probability: float


class AggregationDetail(BaseModel):
    total_patches_evaluated: int
    landslide_patches: int
    normal_patches: int
    landslide_patch_percentage: float
    high_risk_coordinates: List[Dict[str, Any]]


class SatelliteDetail(BaseModel):
    source: str
    acquisition_id: Optional[str] = None
    acquisition_time: Optional[str] = None
    cloud_cover_percent: Optional[float] = None
    mode: str = Field(..., examples=["REAL_COPERNICUS"])


class AoiDetail(BaseModel):
    name: str
    bbox: Optional[List[float]] = None
    is_test_aoi: bool = False


class PredictionSummary(BaseModel):
    overall_class: str
    max_probability: float
    decision_threshold: float
    model_version: str


class LatestPredictionResponse(BaseModel):
    id: Optional[int] = None
    prediction: str = Field(..., examples=["LANDSLIDE DETECTED"])
    probability: float = Field(..., examples=[0.895])
    decision_threshold: float = Field(..., examples=[0.40])
    acquisition_id: Optional[str] = Field(None, examples=["S2B_MSIL2A_20240113T051159_N0510_R019_T43RER_20240113T070440"])
    acquisition_timestamp: Optional[str] = Field(None, examples=["2024-01-13T05:11:59Z"])
    processing_timestamp: str = Field(..., examples=["2024-03-01T10:30:00Z"])
    cloud_cover_percentage: Optional[float] = Field(None, examples=[4.5])
    aoi_name: str = Field(..., examples=["TEST_DEVELOPMENT_AOI"])
    aoi_bbox: Optional[List[float]] = None
    data_status: str = Field(..., examples=["SIMULATED"])
    is_simulated: bool = Field(..., examples=[True])
    model_version: str = Field(..., examples=["GeoGuard_CNN_v1"])
    total_patches: int = Field(1, examples=[2])
    landslide_patches: int = Field(0, examples=[1])
    high_risk_coordinates: List[Dict[str, Any]] = Field(default_factory=list)
    message: str = Field(..., examples=["Macro landslide detected across 1 of 2 evaluated terrain patches."])


class PredictionHistoryResponse(BaseModel):
    total_count: int
    limit: int
    offset: int
    predictions: List[LatestPredictionResponse]


class SatelliteStatusResponse(BaseModel):
    connection_status: str = Field(..., examples=["CONNECTED"])
    stac_catalogue_url: str
    credentials_configured: bool
    last_checked_timestamp: Optional[str] = None
    latest_catalog_acquisition_id: Optional[str] = None
    latest_processed_acquisition_id: Optional[str] = None
    has_new_acquisition: bool
    mode: str = Field(..., examples=["SIMULATED"])
    monitored_aoi: Dict[str, Any]
    polling_interval_minutes: int


class ManualPredictionResponse(BaseModel):
    prediction: str = Field(..., examples=["LANDSLIDE DETECTED"])
    probability: float = Field(..., examples=[0.912])
    decision_threshold: float = Field(..., examples=[0.40])
    input_shape: List[int] = Field(..., examples=[[128, 128, 4]])
    data_status: str = Field(default="MANUAL_TEST")
    processing_timestamp: str
    model_version: str = Field(..., examples=["GeoGuard_CNN_v1"])
    notice: str = Field(default="MANUAL AD-HOC TEST. Not linked to autonomous satellite corridor stream.")


class IoTTelemetryInput(BaseModel):
    sensor_id: str = Field("GEO-NODE-01", examples=["GEO-NODE-01"])
    timestamp: Optional[str] = None
    soil_moisture_percent: float = Field(..., examples=[42.5], description="Volumetric water content percentage")
    rainfall_mm_24h: float = Field(..., examples=[85.0], description="Cumulative 24-hour rainfall in mm")
    vibration_intensity: float = Field(..., examples=[0.15], description="Geophone or IMU vibration intensity (g)")
    tilt_angle_deg: float = Field(..., examples=[3.2], description="Slope inclinometer deviation in degrees")
    battery_percent: Optional[float] = Field(95.0, examples=[95.0])


class IoTFusionResponse(BaseModel):
    cross_validation_status: str = Field(..., examples=["AGREEMENT"])
    combined_risk_level: str = Field(..., examples=["CRITICAL"])
    satellite_macro_status: Dict[str, Any]
    iot_micro_telemetry: Dict[str, Any]
    action_recommendation: str
    fusion_rationale: str


# --- Phase 8 India-Wide AOI Management Schemas ---

class AOICreateRequest(BaseModel):
    aoi_id: str = Field(..., min_length=2, max_length=64, examples=["IN-UK-01"], description="Unique identifier for the corridor")
    name: str = Field(..., min_length=3, max_length=128, examples=["Rishikesh-Joshimath Corridor"])
    state: str = Field(..., min_length=2, max_length=64, examples=["Uttarakhand"])
    region: str = Field(..., min_length=2, max_length=64, examples=["Garhwal Himalayas"])
    bbox_wgs84: List[float] = Field(..., min_length=4, max_length=4, examples=[[78.45, 30.25, 78.60, 30.40]], description="[min_lon, min_lat, max_lon, max_lat]")
    polygon_geojson: Optional[Dict[str, Any]] = None
    monitoring_status: Optional[str] = Field("ACTIVE", examples=["ACTIVE"])


class AOIUpdateRequest(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    region: Optional[str] = None
    bbox_wgs84: Optional[List[float]] = None
    polygon_geojson: Optional[Dict[str, Any]] = None
    monitoring_status: Optional[str] = None


class AOIResponse(BaseModel):
    aoi_id: str
    name: str
    state: str
    region: str
    bbox_wgs84: List[float]
    polygon_geojson: Optional[Dict[str, Any]] = None
    monitoring_status: str
    last_acquisition_id: Optional[str] = None
    last_acquisition_time: Optional[str] = None
    last_quality_status: str = "UNCHECKED"
    created_at: str
    updated_at: str


class AOIListResponse(BaseModel):
    total_count: int
    aois: List[AOIResponse]


class LandslideClusterEvent(BaseModel):
    event_id: str
    patch_count: int
    constituent_patch_indices: List[int]
    centroid: Dict[str, float]
    bounding_box: List[float]
    max_model_score: float
    mean_model_score: float
    risk_severity: str
    notice: str


class AOIProcessResultResponse(BaseModel):
    aoi_id: str
    aoi_name: str
    state: str
    region: str
    status: str
    acquisition_id: Optional[str] = None
    acquisition_time: Optional[str] = None
    cloud_cover_percent: Optional[float] = None
    data_quality_status: str
    total_patches_evaluated: int = 0
    landslide_patches_detected: int = 0
    distinct_landslide_events: int = 0
    clustered_events: List[LandslideClusterEvent] = []
    model_version: str
    decision_threshold: float
    is_simulated: bool = False
    message: str


class ProcessedAcquisitionItem(BaseModel):
    id: int
    aoi_id: str
    acquisition_id: str
    acquisition_time: Optional[str] = None
    cloud_cover: Optional[float] = None
    processing_status: str
    total_patches_extracted: int
    landslide_patches_detected: int
    landslide_events_clustered: int
    processed_at: str

