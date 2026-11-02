"""
GeoGuard Landslide Detection System - Copernicus Sentinel-2 Satellite Module.

Handles:
- CDSE STAC search for Sentinel-2 Level-2A acquisitions over configured AOI.
- Authentication via Copernicus Data Space Ecosystem OAuth2.
- Retrieval and band alignment for B02 (Blue), B03 (Green), B04 (Red), B08 (NIR).
- Scene tiling into 128x128 patches with geographic coordinate tracking.
- Edge handling without introducing misleading synthetic artifacts.
- Macro spatial aggregation for AOI-level decision making.
- Clear separation between REAL Copernicus observations and SIMULATED test mode.
"""

import os
import sys
import time
import logging
import datetime
from typing import Dict, Any, List, Optional, Tuple

import requests
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml_service.config import settings
from ml_service.src.preprocessing import SatellitePreprocessor
from ml_service.src.predict import LandslidePredictor
from ml_service.src.data_loader import LandslideDatasetLoader
from ml_service.src.spatial_tiling import calculate_native_10m_pixel_dims, extract_sliding_window_patches
from ml_service.src.radiometric_calibration import validate_satellite_radiometry
from ml_service.src.event_clustering import cluster_landslide_patches
from ml_service.src import database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


_DEFAULT = object()


class CopernicusClient:
    """Client for Copernicus Data Space Ecosystem (CDSE) STAC and Processing APIs."""

    def __init__(
        self,
        client_id: Any = _DEFAULT,
        client_secret: Any = _DEFAULT,
        stac_url: str = settings.CDSE_STAC_URL,
        token_url: str = settings.CDSE_TOKEN_URL,
        sh_process_url: str = settings.CDSE_SH_PROCESS_URL,
    ):
        self.client_id = settings.CDSE_CLIENT_ID if client_id is _DEFAULT else client_id
        self.client_secret = settings.CDSE_CLIENT_SECRET if client_secret is _DEFAULT else client_secret
        self.stac_url = stac_url
        self.token_url = token_url
        self.sh_process_url = sh_process_url

        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0

    @property
    def has_credentials(self) -> bool:
        """Returns True only if both client ID and client secret are configured."""
        return bool(
            self.client_id
            and self.client_secret
            and self.client_id != "your_cdse_client_id_here"
        )

    def get_auth_token(self) -> Optional[str]:
        """
        Retrieves OAuth2 access token from CDSE identity provider.
        Caches token until 60 seconds before expiration.
        """
        if not self.has_credentials:
            logger.info("CDSE credentials not configured; operating in unauthenticated / test mode.")
            return None

        now = time.time()
        if self._access_token and now < self._token_expiry - 60:
            return self._access_token

        logger.info("Authenticating with Copernicus Data Space OAuth2 service...")
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        try:
            resp = requests.post(self.token_url, data=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                self._access_token = data.get("access_token")
                expires_in = data.get("expires_in", 3600)
                self._token_expiry = now + expires_in
                logger.info(f"CDSE token acquired successfully (expires in {expires_in}s).")
                return self._access_token
            else:
                logger.error(f"CDSE Authentication failed: HTTP {resp.status_code} - {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Error connecting to CDSE token service: {e}")
            return None

    def search_acquisitions(
        self,
        bbox: List[float],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_cloud_cover: float = settings.MAX_CLOUD_COVER_PERCENT,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Discovers Sentinel-2 Level-2A acquisitions for given AOI bounding box.
        Uses public CDSE STAC v1 endpoint.
        """
        if end_date is None:
            end_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if start_date is None:
            # Default to past 30 days
            start_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)
            start_date = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        datetime_param = f"{start_date}/{end_date}"
        payload = {
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "datetime": datetime_param,
            "limit": limit,
        }

        logger.info(
            f"Querying CDSE STAC catalogue for bbox={bbox}, datetime={datetime_param}, "
            f"max_cloud={max_cloud_cover}%"
        )
        try:
            resp = requests.post(self.stac_url, json=payload, timeout=20)
            if resp.status_code != 200:
                logger.error(f"CDSE STAC search failed: HTTP {resp.status_code} - {resp.text[:300]}")
                return []

            if "json" not in resp.headers.get("Content-Type", "").lower():
                logger.warning(f"CDSE STAC search returned non-JSON response ({resp.headers.get('Content-Type')}): {resp.text[:200]}")
                return []

            features = resp.json().get("features", [])
            usable = []
            for feat in features:
                props = feat.get("properties", {})
                cloud = props.get("eo:cloud_cover", props.get("cloudCover", 100.0))
                acq_id = feat.get("id")
                acq_time = props.get("datetime")

                is_usable = cloud <= max_cloud_cover
                usable.append(
                    {
                        "acquisition_id": acq_id,
                        "acquisition_time": acq_time,
                        "cloud_cover_percent": float(cloud) if cloud is not None else None,
                        "is_usable": is_usable,
                        "bbox": feat.get("bbox", bbox),
                        "stac_assets": list(feat.get("assets", {}).keys()),
                        "source": "Copernicus Data Space Ecosystem (Sentinel-2 L2A)",
                    }
                )
            logger.info(f"CDSE STAC found {len(usable)} acquisitions ({sum(1 for a in usable if a['is_usable'])} usable).")
            return usable
        except Exception as e:
            logger.error(f"Exception during CDSE STAC search: {e}")
            return []

    def fetch_scene_imagery(
        self,
        acquisition: Dict[str, Any],
        bbox: List[float],
        patch_size: int = settings.PATCH_SIZE,
        allow_simulated_fallback: bool = False,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Retrieves the 4-channel [B02, B03, B04, B08] optical raster over the AOI bbox.
        If allow_simulated_fallback is False, strictly fails if credentials are missing or network fails.
        If allow_simulated_fallback is True, securely returns SIMULATED test imagery for unit tests.
        """
        token = self.get_auth_token()
        if token and self.has_credentials:
            # Operational Copernicus Processing API path
            logger.info(f"Requesting 4-band imagery for {acquisition['acquisition_id']} via Sentinel Hub Processing API...")
            try:
                # Sentinel Hub Evalscript for 10m native bands B02, B03, B04, B08
                evalscript = """
                //VERSION=3
                function setup() {
                  return {
                    input: ["B02", "B03", "B04", "B08"],
                    output: { bands: 4, sampleType: "FLOAT32" }
                  };
                }
                function evaluatePixel(sample) {
                  return [sample.B02, sample.B03, sample.B04, sample.B08];
                }
                """
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                }
                # Compute native 10m pixel dimensions bounded by API limits
                width_px, height_px, eff_x, eff_y = calculate_native_10m_pixel_dims(bbox)
                max_dim = getattr(settings, "MAX_TILE_REQUEST_DIM", 1500)
                req_width = max(patch_size, min(width_px, max_dim))
                req_height = max(patch_size, min(height_px, max_dim))
                logger.info(
                    f"Sentinel-2 native 10m target: {width_px}x{height_px} px. Requesting: {req_width}x{req_height} px "
                    f"(Effective GSD: {eff_x:.2f}m x {eff_y:.2f}m)"
                )

                data_filter: Dict[str, Any] = {"maxCloudCoverage": settings.MAX_CLOUD_COVER_PERCENT}
                acq_time = acquisition.get("acquisition_time")
                if acq_time:
                    try:
                        acq_dt = datetime.datetime.fromisoformat(acq_time.replace("Z", "+00:00"))
                        t_from = (acq_dt - datetime.timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
                        t_to = (acq_dt + datetime.timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
                        data_filter["timeRange"] = {"from": t_from, "to": t_to}
                    except Exception as e:
                        logger.warning(f"Could not parse acquisition time {acq_time}: {e}")

                body = {
                    "input": {
                        "bounds": {
                            "bbox": bbox,
                            "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                        },
                        "data": [{"type": "sentinel-2-l2a", "dataFilter": data_filter}],
                    },
                    "output": {"width": req_width, "height": req_height, "responses": [{"format": {"type": "image/tiff"}}]},
                    "evalscript": evalscript,
                }
                resp = requests.post(self.sh_process_url, headers=headers, json=body, timeout=30)
                if resp.status_code == 200:
                    import io, tifffile
                    with io.BytesIO(resp.content) as buf:
                        data = tifffile.imread(buf)
                    logger.info(f"Successfully retrieved Copernicus scene of shape {data.shape}")
                    return data.astype(np.float32), {"mode": "REAL_COPERNICUS", "data_status": "VALID"}
                else:
                    logger.warning(f"Processing API returned HTTP {resp.status_code} - {resp.text[:200]}")
                    if not allow_simulated_fallback:
                        return None, {
                            "mode": "FAILED",
                            "data_status": "API_ERROR",
                            "error": f"Sentinel Hub Processing API returned HTTP {resp.status_code}: {resp.text[:200]}",
                        }
            except Exception as e:
                logger.warning(f"Failed to fetch live scene imagery from Copernicus: {e}")
                if not allow_simulated_fallback:
                    return None, {
                        "mode": "FAILED",
                        "data_status": "CONNECTION_ERROR",
                        "error": str(e),
                    }

        if not allow_simulated_fallback:
            logger.warning("CDSE credentials are not configured and allow_simulated_fallback is False.")
            return None, {
                "mode": "BLOCKED",
                "data_status": "CREDENTIALS_MISSING",
                "error": "Copernicus Data Space Ecosystem credentials (CDSE_CLIENT_ID / CDSE_CLIENT_SECRET) are not configured. Cannot download live satellite raster.",
            }

        # SIMULATED / Development Test Mode
        logger.info("Operating in SIMULATED test mode (using verified Landslide4Sense test patches).")
        loader = LandslideDatasetLoader(data_dir=settings.RAW_DATA_DIR)
        test_samples = loader.get_sample_paths("test")
        if not test_samples:
            return None, {"mode": "SIMULATED", "data_status": "INSUFFICIENT_DATA", "error": "No test samples available"}

        # Combine two 128x128 patches into a 128x256 composite scene to test multi-patch tiling
        patch1, _, _ = loader.load_single_sample(test_samples[0][1], test_samples[0][2])
        patch2, _, _ = loader.load_single_sample(test_samples[1][1], test_samples[1][2])
        sim_scene = np.concatenate([patch1, patch2], axis=1)  # shape (128, 256, 4)

        meta = {
            "mode": "SIMULATED",
            "data_status": "SIMULATED",
            "simulated_notice": "TEST/DEVELOPMENT MODE ONLY. NEVER DISPLAY AS REAL OBSERVATIONS.",
            "scene_shape": list(sim_scene.shape),
        }
        return sim_scene, meta


class SatellitePipeline:
    """Tiling, prediction, and spatial aggregation engine."""

    def __init__(
        self,
        predictor: Optional[LandslidePredictor] = None,
        copernicus_client: Optional[CopernicusClient] = None,
    ):
        self.predictor = predictor or LandslidePredictor()
        self.copernicus = copernicus_client or CopernicusClient()

        # Dedicated CNN v2 predictor for Phase 8 India-wide monitoring
        v2_model = os.path.join(settings.MODELS_DIR, "cnn_model_v2.keras")
        v2_cfg = os.path.join(settings.MODELS_DIR, "preprocessing_config_v2.json")
        if os.path.exists(v2_model) and os.path.exists(v2_cfg):
            try:
                self.predictor_v2 = LandslidePredictor(
                    model_path=v2_model,
                    config_path=v2_cfg,
                    decision_threshold=0.25,
                    model_version="GeoGuard_CNN_v2",
                )
                logger.info("Initialized dedicated GeoGuard_CNN_v2 predictor for Phase 8 multi-AOI monitoring.")
            except Exception as e:
                logger.warning(f"Could not initialize GeoGuard_CNN_v2: {e}")
                self.predictor_v2 = self.predictor
        else:
            self.predictor_v2 = self.predictor

    @staticmethod
    def tile_scene(
        scene: np.ndarray,
        bbox: List[float],
        patch_size: int = settings.PATCH_SIZE,
        max_invalid_ratio: float = settings.MAX_INVALID_PIXEL_RATIO,
    ) -> List[Dict[str, Any]]:
        """
        Divides a 4-channel satellite scene into 128x128 patches with geographic tracking.
        Edge handling: Excludes incomplete border remnants without synthetic zero-padding.
        """
        H, W, C = scene.shape
        if C != 4:
            raise ValueError(f"Expected 4 channels, got {C}")

        min_lon, min_lat, max_lon, max_lat = bbox
        lon_per_pixel = (max_lon - min_lon) / W
        lat_per_pixel = (max_lat - min_lat) / H

        patches = []
        n_rows = H // patch_size
        n_cols = W // patch_size

        for r in range(n_rows):
            r_start = r * patch_size
            r_end = r_start + patch_size

            for c in range(n_cols):
                c_start = c * patch_size
                c_end = c_start + patch_size

                patch_arr = scene[r_start:r_end, c_start:c_end, :]

                # Check invalid pixels (NaN, inf, or zeroes across all bands)
                invalid_mask = np.isnan(patch_arr).any(axis=-1) | np.isinf(patch_arr).any(axis=-1) | (patch_arr == 0).all(axis=-1)
                invalid_ratio = float(np.sum(invalid_mask) / (patch_size * patch_size))

                # Calculate geographic coordinates
                p_min_lon = min_lon + c_start * lon_per_pixel
                p_max_lon = min_lon + c_end * lon_per_pixel
                # Latitude decreases as row increases from top
                p_max_lat = max_lat - r_start * lat_per_pixel
                p_min_lat = max_lat - r_end * lat_per_pixel
                center_lon = (p_min_lon + p_max_lon) / 2.0
                center_lat = (p_min_lat + p_max_lat) / 2.0

                is_usable = invalid_ratio <= max_invalid_ratio

                patches.append(
                    {
                        "patch_index": len(patches),
                        "row": r,
                        "col": c,
                        "patch_arr": patch_arr,
                        "is_usable": is_usable,
                        "invalid_ratio": round(invalid_ratio, 4),
                        "bbox": [round(p_min_lon, 5), round(p_min_lat, 5), round(p_max_lon, 5), round(p_max_lat, 5)],
                        "center_coords": {"lat": round(center_lat, 5), "lon": round(center_lon, 5)},
                    }
                )

        logger.info(f"Tiled scene of {H}x{W} into {len(patches)} patches ({sum(1 for p in patches if p['is_usable'])} usable).")
        return patches

    def process_aoi(
        self,
        bbox: Optional[List[float]] = None,
        aoi_name: Optional[str] = None,
        acquisition_id: Optional[str] = None,
        allow_simulated_fallback: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end pipeline:
        1. Validates AOI.
        2. Discovers or selects acquisition.
        3. Retrieves scene raster (strictly real for real corridors; simulation allowed only for test AOI).
        4. Tiles into 128x128 patches.
        5. Runs CNN predictions.
        6. Aggregates into documented AOI-level response.
        """
        if bbox is None:
            aoi_name, bbox, is_test_aoi = settings.get_active_aoi(allow_test_fallback=True)
        else:
            is_test_aoi = (aoi_name == settings.TEST_AOI_NAME)

        can_fallback = allow_simulated_fallback or is_test_aoi

        if bbox is None:
            return {
                "status": "error",
                "data_status": "UNCONFIGURED_AOI",
                "message": "Highway corridor coordinates are not configured. Provide HIGHWAY_AOI_BBOX in .env.",
                "aoi": {"name": aoi_name or "UNCONFIGURED", "bbox": None},
            }

        # Discover or define acquisition
        acquisitions = self.copernicus.search_acquisitions(bbox=bbox, limit=3)
        if not acquisitions:
            if not can_fallback:
                return {
                    "status": "error",
                    "data_status": "NO_ACQUISITION_FOUND",
                    "message": f"No Sentinel-2 Level-2A acquisitions found in CDSE STAC catalogue for bbox={bbox}.",
                    "aoi": {"name": aoi_name, "bbox": bbox, "is_test_aoi": is_test_aoi},
                }
            logger.info("No live acquisitions found; using designated reference acquisition for test.")
            acquisition = {
                "acquisition_id": acquisition_id or "S2_REF_TEST_ACQUISITION_001",
                "acquisition_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "cloud_cover_percent": 5.0,
                "is_usable": True,
                "source": "SIMULATED_TEST_MODE",
            }
        else:
            # Pick first usable acquisition
            usable_acqs = [a for a in acquisitions if a.get("is_usable", False)]
            acquisition = usable_acqs[0] if usable_acqs else acquisitions[0]

        # Fetch Scene
        scene, meta = self.copernicus.fetch_scene_imagery(
            acquisition, bbox=bbox, allow_simulated_fallback=can_fallback
        )
        if scene is None:
            return {
                "status": "error",
                "data_status": meta.get("data_status", "INSUFFICIENT_DATA"),
                "message": meta.get(
                    "error",
                    "Unable to acquire usable satellite raster. Excessive cloud cover, missing credentials, or API unavailable.",
                ),
                "satellite": acquisition,
                "aoi": {"name": aoi_name, "bbox": bbox, "is_test_aoi": is_test_aoi},
            }

        # Tile Scene
        patches = self.tile_scene(scene, bbox=bbox)
        valid_patches = [p for p in patches if p["is_usable"]]

        if not valid_patches:
            return {
                "status": "error",
                "data_status": "INSUFFICIENT_DATA",
                "message": "All generated patches exceed the maximum invalid or cloud-contaminated pixel ratio.",
                "satellite": acquisition,
                "aoi": {"name": aoi_name, "bbox": bbox, "is_test_aoi": is_test_aoi},
            }

        # Run Predictions
        patch_results = []
        positive_count = 0
        probabilities = []

        for p in valid_patches:
            pred = self.predictor.predict_patch(p["patch_arr"])
            is_ls = pred["prediction_class"] == "LANDSLIDE DETECTED"
            if is_ls:
                positive_count += 1
            probabilities.append(pred["probability"])

            patch_results.append(
                {
                    "patch_index": p["patch_index"],
                    "center_coords": p["center_coords"],
                    "bbox": p["bbox"],
                    "prediction_class": pred["prediction_class"],
                    "probability": pred["probability"],
                }
            )

        # Macro AOI Spatial Aggregation
        total_valid = len(valid_patches)
        pos_percentage = round((positive_count / total_valid) * 100, 2)
        max_prob = max(probabilities) if probabilities else 0.0

        # High risk coordinates (patches with landslide detected)
        high_risk = [p for p in patch_results if p["prediction_class"] == "LANDSLIDE DETECTED"]

        # Macro decision: Macro landslide detected if any patch is positive and confidence is high
        overall_class = "LANDSLIDE DETECTED" if positive_count > 0 else "NORMAL / STABLE"

        return {
            "status": "success",
            "data_status": meta.get("data_status", "VALID"),
            "prediction": {
                "overall_class": overall_class,
                "max_probability": max_prob,
                "decision_threshold": self.predictor.decision_threshold,
                "model_version": self.predictor.model_version,
            },
            "satellite": {
                "source": acquisition.get("source", "Copernicus Sentinel-2"),
                "acquisition_id": acquisition.get("acquisition_id"),
                "acquisition_time": acquisition.get("acquisition_time"),
                "cloud_cover_percent": acquisition.get("cloud_cover_percent"),
                "mode": meta.get("mode", "REAL_COPERNICUS"),
            },
            "aoi": {
                "name": aoi_name,
                "bbox": bbox,
                "is_test_aoi": is_test_aoi,
            },
            "aggregation": {
                "total_patches_evaluated": total_valid,
                "landslide_patches": positive_count,
                "normal_patches": total_valid - positive_count,
                "landslide_patch_percentage": pos_percentage,
                "high_risk_coordinates": [
                    {"lat": hr["center_coords"]["lat"], "lon": hr["center_coords"]["lon"], "prob": hr["probability"]}
                    for hr in high_risk
                ],
            },
            "patch_details": patch_results,
        }

    def process_monitored_aoi(
        self,
        aoi: Dict[str, Any],
        allow_simulated_fallback: bool = False,
        force_reprocess: bool = False,
        overlap_ratio: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Executes autonomous Sentinel-2 monitoring for an India-wide registered AOI:
        1. Queries CDSE STAC catalogue for latest acquisitions.
        2. Deduplicates: Skips processing if this acquisition was already processed for this AOI.
        3. Validates cloud cover and returns INSUFFICIENT_DATA if cloudy.
        4. Ingests native 10m multi-spectral raster (B02, B03, B04, B08).
        5. Validates radiometry with physical BOA guardrails.
        6. Extracts overlapping 128x128 patches with 10m ground resolution.
        7. Runs CNN v2 inference using configured decision threshold (default 0.25).
        8. Clusters overlapping positive patches to prevent duplicate landslide event alerts.
        9. Records results in SQLite database and updates AOI status.
        """
        aoi_id = aoi["aoi_id"]
        name = aoi["name"]
        state = aoi.get("state", "India")
        region = aoi.get("region", "Region")
        bbox = aoi["bbox_wgs84"]

        logger.info(f"Processing monitored AOI {aoi_id} ({name}, {state}) bbox={bbox}")

        # 1. Discover Acquisitions via CDSE STAC
        acquisitions = self.copernicus.search_acquisitions(bbox=bbox, limit=5)
        if not acquisitions:
            if not allow_simulated_fallback:
                return {
                    "aoi_id": aoi_id,
                    "aoi_name": name,
                    "state": state,
                    "region": region,
                    "status": "error",
                    "data_quality_status": "NO_ACQUISITION_FOUND",
                    "message": f"No Sentinel-2 L2A acquisitions found for AOI {aoi_id} ({bbox}).",
                    "model_version": self.predictor.model_version,
                    "decision_threshold": self.predictor.decision_threshold,
                    "is_simulated": False,
                }
            logger.info("Using simulated reference acquisition for testing.")
            acquisition = {
                "acquisition_id": f"S2_SIM_{aoi_id}_REF",
                "acquisition_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "cloud_cover_percent": 4.0,
                "is_usable": True,
                "source": "SIMULATED_TEST_MODE",
            }
        else:
            usable_acqs = [a for a in acquisitions if a.get("is_usable", False)]
            acquisition = usable_acqs[0] if usable_acqs else acquisitions[0]

        acq_id = acquisition.get("acquisition_id", "UNKNOWN_ACQ")
        acq_time = acquisition.get("acquisition_time")
        cloud_pct = acquisition.get("cloud_cover_percent")

        # 2. Acquisition Deduplication
        if not force_reprocess and database.is_acquisition_processed(acq_id, aoi_id=aoi_id):
            logger.info(f"Acquisition '{acq_id}' has already been processed for AOI '{aoi_id}'. Skipping.")
            return {
                "aoi_id": aoi_id,
                "aoi_name": name,
                "state": state,
                "region": region,
                "status": "skipped",
                "acquisition_id": acq_id,
                "acquisition_time": acq_time,
                "cloud_cover_percent": cloud_pct,
                "data_quality_status": aoi.get("last_quality_status", "VALID"),
                "total_patches_evaluated": 0,
                "landslide_patches_detected": 0,
                "distinct_landslide_events": 0,
                "clustered_events": [],
                "model_version": self.predictor.model_version,
                "decision_threshold": self.predictor.decision_threshold,
                "is_simulated": False,
                "message": f"Acquisition {acq_id} has already been processed for AOI {aoi_id}. Deduplicated.",
            }

        # 3. Check Cloud Coverage Guardrail
        if cloud_pct is not None and cloud_pct > settings.MAX_CLOUD_COVER_PERCENT:
            logger.warning(f"Excessive cloud cover ({cloud_pct}%) for acquisition {acq_id}. Returning INSUFFICIENT_DATA.")
            database.record_processed_acquisition({
                "aoi_id": aoi_id,
                "acquisition_id": acq_id,
                "acquisition_time": acq_time,
                "cloud_cover": cloud_pct,
                "processing_status": "INSUFFICIENT_DATA",
            })
            database.update_aoi_acquisition_status(aoi_id, acq_id, acq_time, "INSUFFICIENT_DATA (Cloudy)")
            return {
                "aoi_id": aoi_id,
                "aoi_name": name,
                "state": state,
                "region": region,
                "status": "error",
                "acquisition_id": acq_id,
                "acquisition_time": acq_time,
                "cloud_cover_percent": cloud_pct,
                "data_quality_status": "INSUFFICIENT_DATA",
                "total_patches_evaluated": 0,
                "landslide_patches_detected": 0,
                "distinct_landslide_events": 0,
                "clustered_events": [],
                "model_version": self.predictor.model_version,
                "decision_threshold": self.predictor.decision_threshold,
                "is_simulated": False,
                "message": f"Acquisition cloud cover ({cloud_pct}%) exceeds threshold ({settings.MAX_CLOUD_COVER_PERCENT}%). Flagged INSUFFICIENT_DATA.",
            }

        # 4. Fetch Native-Scale Scene
        scene, meta = self.copernicus.fetch_scene_imagery(
            acquisition, bbox=bbox, allow_simulated_fallback=allow_simulated_fallback
        )
        if scene is None:
            data_status = meta.get("data_status", "INSUFFICIENT_DATA")
            database.update_aoi_acquisition_status(aoi_id, acq_id, acq_time, data_status)
            return {
                "aoi_id": aoi_id,
                "aoi_name": name,
                "state": state,
                "region": region,
                "status": "error",
                "acquisition_id": acq_id,
                "acquisition_time": acq_time,
                "cloud_cover_percent": cloud_pct,
                "data_quality_status": data_status,
                "total_patches_evaluated": 0,
                "landslide_patches_detected": 0,
                "distinct_landslide_events": 0,
                "clustered_events": [],
                "model_version": self.predictor.model_version,
                "decision_threshold": self.predictor.decision_threshold,
                "is_simulated": False,
                "message": meta.get("error", "Unable to retrieve satellite raster from CDSE."),
            }

        is_sim = (meta.get("mode") == "SIMULATED")

        # 5. Radiometric Guardrail Check
        if not is_sim:
            is_valid, rad_status, rad_diag = validate_satellite_radiometry(scene)
            if not is_valid:
                status_code = rad_status
                logger.warning(f"Radiometric check failed: {status_code}")
                database.record_processed_acquisition({
                    "aoi_id": aoi_id,
                    "acquisition_id": acq_id,
                    "acquisition_time": acq_time,
                    "cloud_cover": cloud_pct,
                    "processing_status": status_code,
                })
                database.update_aoi_acquisition_status(aoi_id, acq_id, acq_time, status_code)
                return {
                    "aoi_id": aoi_id,
                    "aoi_name": name,
                    "state": state,
                    "region": region,
                    "status": "error",
                    "acquisition_id": acq_id,
                    "acquisition_time": acq_time,
                    "cloud_cover_percent": cloud_pct,
                    "data_quality_status": status_code,
                    "total_patches_evaluated": 0,
                    "landslide_patches_detected": 0,
                    "distinct_landslide_events": 0,
                    "clustered_events": [],
                    "model_version": self.predictor.model_version,
                    "decision_threshold": self.predictor.decision_threshold,
                    "is_simulated": False,
                    "message": f"Radiometric check failed ({status_code}). Raster cannot be safely processed.",
                }

        # 6. Extract Native 10m Overlapping Patches
        patches = extract_sliding_window_patches(
            scene, bbox, patch_size=settings.PATCH_SIZE, overlap_ratio=overlap_ratio
        )
        valid_patches = [p for p in patches if p.get("is_usable", True)]

        if not valid_patches:
            database.update_aoi_acquisition_status(aoi_id, acq_id, acq_time, "INSUFFICIENT_DATA")
            return {
                "aoi_id": aoi_id,
                "aoi_name": name,
                "state": state,
                "region": region,
                "status": "error",
                "acquisition_id": acq_id,
                "acquisition_time": acq_time,
                "cloud_cover_percent": cloud_pct,
                "data_quality_status": "INSUFFICIENT_DATA",
                "total_patches_evaluated": 0,
                "landslide_patches_detected": 0,
                "distinct_landslide_events": 0,
                "clustered_events": [],
                "model_version": self.predictor.model_version,
                "decision_threshold": self.predictor.decision_threshold,
                "is_simulated": is_sim,
                "message": "All extracted patches exceeded tolerable corrupt/nodata pixel limits.",
            }

        # 7. Model Inference on Each Patch using CNN v2
        positive_patches = []
        all_patch_details = []

        for p in valid_patches:
            pred = self.predictor_v2.predict_patch(p["patch_arr"])
            is_ls = (pred["prediction_class"] == "LANDSLIDE DETECTED")
            prob = pred["probability"]

            patch_detail = {
                "patch_index": p["patch_index"],
                "center_coords": {
                    "lat": p["centroid"]["latitude"],
                    "lon": p["centroid"]["longitude"],
                },
                "bbox": p["bbox_wgs84"],
                "prediction_class": pred["prediction_class"],
                "probability": prob,
            }
            all_patch_details.append(patch_detail)

            if is_ls:
                positive_patches.append(patch_detail)

        # 8. Prevent Duplicate Patch Detections via Spatial Clustering
        clustered_events = cluster_landslide_patches(positive_patches)

        # 9. Record Results in Database
        total_eval = len(valid_patches)
        ls_patch_count = len(positive_patches)
        distinct_events_count = len(clustered_events)
        max_prob = max([p["probability"] for p in all_patch_details]) if all_patch_details else 0.0
        overall_class = "LANDSLIDE DETECTED" if ls_patch_count > 0 else "NORMAL / STABLE"

        macro_pred = {
            "prediction": {
                "overall_class": overall_class,
                "max_probability": max_prob,
                "decision_threshold": self.predictor_v2.decision_threshold,
                "model_version": self.predictor_v2.model_version,
            },
            "satellite": {
                "source": acquisition.get("source", "Copernicus Sentinel-2"),
                "acquisition_id": acq_id,
                "acquisition_time": acq_time,
                "cloud_cover_percent": cloud_pct,
                "mode": meta.get("mode", "REAL_COPERNICUS"),
            },
            "aoi": {
                "name": f"{aoi_id}: {name} ({state})",
                "bbox": bbox,
                "is_test_aoi": is_sim,
            },
            "aggregation": {
                "total_patches_evaluated": total_eval,
                "landslide_patches": ls_patch_count,
                "normal_patches": total_eval - ls_patch_count,
                "high_risk_coordinates": [
                    {"lat": e["centroid"]["lat"], "lon": e["centroid"]["lon"], "prob": e["max_model_score"]}
                    for e in clustered_events
                ],
            },
            "patch_details": all_patch_details,
            "data_status": "VALID",
        }
        database.save_prediction(macro_pred)

        database.record_processed_acquisition({
            "aoi_id": aoi_id,
            "acquisition_id": acq_id,
            "acquisition_time": acq_time,
            "cloud_cover": cloud_pct,
            "processing_status": "SUCCESS",
            "total_patches_extracted": total_eval,
            "landslide_patches_detected": ls_patch_count,
            "landslide_events_clustered": distinct_events_count,
        })
        database.update_aoi_acquisition_status(aoi_id, acq_id, acq_time, "VALID")

        return {
            "aoi_id": aoi_id,
            "aoi_name": name,
            "state": state,
            "region": region,
            "status": "success",
            "acquisition_id": acq_id,
            "acquisition_time": acq_time,
            "cloud_cover_percent": cloud_pct,
            "data_quality_status": "VALID",
            "total_patches_evaluated": total_eval,
            "landslide_patches_detected": ls_patch_count,
            "distinct_landslide_events": distinct_events_count,
            "clustered_events": clustered_events,
            "model_version": self.predictor_v2.model_version,
            "decision_threshold": self.predictor_v2.decision_threshold,
            "is_simulated": is_sim,
            "message": (
                f"Evaluated {total_eval} native 10m patches. Found {ls_patch_count} positive patch detections "
                f"clustered into {distinct_events_count} distinct landslide events."
            ),
        }
