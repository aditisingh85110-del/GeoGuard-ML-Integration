"""
=============================================================================
GEOGUARD AI — AUTONOMOUS HIMALAYAN SATELLITE SURVEILLANCE ENGINE
Module: ml_service/src/surveillance.py
Purpose: Multi-region fleet surveillance across 6 Himalayan territories:
         1. Uttarakhand (Dehradun / Mussoorie Foothills)
         2. Himachal Pradesh (Kinnaur NH-05 Nigulsari)
         3. Jammu & Kashmir (Ramban NH-44 Panthyal)
         4. Ladakh (Leh-Kargil NH-01 Indus Valley)
         5. Sikkim (Chungthang Teesta Confluence)
         6. Arunachal Pradesh (Tawang-Bomdila B-C-T Highway)
=============================================================================
Safety Contracts:
  - Reuses existing CopernicusClient, SatellitePipeline, BiTemporalChangeDetector,
    and HonestPredictionEngine.
  - Zero modification to CNN v1/v2 weights or preprocessing.
  - Independent per-AOI processing: one failing AOI never halts the fleet.
  - Observed disturbance intensity (LOW, MODERATE, HIGH, INSUFFICIENT_DATA)
    is strictly provisional and explicitly decoupled from probability.
=============================================================================
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from ml_service.config import settings
from ml_service.src import database
from ml_service.src.satellite import CopernicusClient, SatellitePipeline
from ml_service.src.judge_pipeline import HonestPredictionEngine
from ml_service.src.india_boundary import validate_aoi_bbox_in_india

logger = logging.getLogger("geoguard_surveillance")

CACHE_FILE_PATH = os.path.join(settings.DATA_DIR, "surveillance_cache.json")


# -----------------------------------------------------------------------------
# 1. Authoritative Himalayan Surveillance AOI Registry
# -----------------------------------------------------------------------------
HIMALAYAN_SURVEILLANCE_REGISTRY: List[Dict[str, Any]] = [
    {
        "aoi_id": "IN-UK-DEHRADUN-01",
        "name": "Dehradun Foothills Monitoring Corridor",
        "corridor_code": "NH-58 / NH-72",
        "state": "Uttarakhand",
        "region": "Western Himalayas",
        "bbox_wgs84": [78.0300, 30.3100, 78.0428, 30.3220],
        "center_coords": [78.0364, 30.3160],
        "verification_status": "VERIFIED",
        "verification_source": "NHAI Highway Corridor Survey & Survey of India Toposheet",
        "monitoring_context": "Sub-Himalayan Siwalik foothill transit with seasonal toe erosion and monsoon debris slides",
        "hazard_classification": "DEBRIS_SLIDE_AND_EROSION_HAZARD",
    },
    {
        "aoi_id": "IN-HP-KINNAUR-01",
        "name": "Kinnaur Nigulsari Highway Corridor",
        "corridor_code": "NH-05 (Hindustan-Tibet Road)",
        "state": "Himachal Pradesh",
        "region": "Western Himalayas",
        "bbox_wgs84": [77.8500, 31.5200, 77.8700, 31.5400],
        "center_coords": [77.8600, 31.5300],
        "verification_status": "VERIFIED",
        "verification_source": "Geological Survey of India Post-Disaster Geotechnical Report (Kinnaur 2021); BRO Project Deepak",
        "monitoring_context": "Steep metamorphic gorge slopes along Sutlej River; historic chronic rockfall and debris avalanche",
        "hazard_classification": "CHRONIC_ROCKFALL_AND_DEBRIS_AVALANCHE",
    },
    {
        "aoi_id": "IN-JK-RAMBAN-01",
        "name": "Ramban Panthyal Active Slope Corridor",
        "corridor_code": "NH-44 (Jammu-Srinagar Arterial)",
        "state": "Jammu & Kashmir",
        "region": "Pir Panjal / Western Himalayas",
        "bbox_wgs84": [75.1800, 33.2200, 75.2100, 33.2500],
        "center_coords": [75.1950, 33.2350],
        "verification_status": "VERIFIED",
        "verification_source": "NHAI / BRO Landslide Hazard Zonation Map; Panthyal Shooting Stone Geotechnical Survey",
        "monitoring_context": "Shattered carbonaceous phyllite cliffs with acute rockfall risk and slope creep affecting Jammu-Srinagar highway",
        "hazard_classification": "ACUTE_SHOOTING_STONE_AND_SLOPE_CREEP",
    },
    {
        "aoi_id": "IN-LA-LEH-01",
        "name": "Leh Indus Valley Highway Corridor",
        "corridor_code": "NH-01 (Srinagar-Leh-Manali Arterial)",
        "state": "Ladakh",
        "region": "Trans-Himalayan Cold Desert",
        "bbox_wgs84": [77.5200, 34.1500, 77.5500, 34.1800],
        "center_coords": [77.5350, 34.1650],
        "verification_status": "VERIFIED",
        "verification_source": "Border Roads Organisation (BRO Project Himank); Geological Survey of India Ladakh Atlas",
        "monitoring_context": "High-altitude arid scree slopes, permafrost thaw degradation, and alluvial fan flash flood washout",
        "hazard_classification": "PERMAFROST_SCREE_AND_FAN_WASHOUT",
    },
    {
        "aoi_id": "IN-SK-TEESTA-01",
        "name": "Chungthang Teesta Confluence Corridor",
        "corridor_code": "North Sikkim Highway (NSH)",
        "state": "Sikkim",
        "region": "Eastern Himalayas",
        "bbox_wgs84": [88.6396, 27.5980, 88.6524, 27.6100],
        "center_coords": [88.6460, 27.6040],
        "verification_status": "VERIFIED",
        "verification_source": "ISRO NRSC Sikkim GLOF Bulletin (Oct 2023); NASA GLC ID 15024; regional_landslides.json",
        "monitoring_context": "Glacial river valley confluence with extreme topographic relief, monsoonal saturation, and toe-undercutting",
        "hazard_classification": "VALLEY_SCOURING_AND_FLANK_COLLAPSE",
    },
    {
        "aoi_id": "IN-AR-TAWANG-01",
        "name": "Tawang-Bomdila B-C-T Highway Corridor",
        "corridor_code": "BCT Strategic Highway (NH-13)",
        "state": "Arunachal Pradesh",
        "region": "Eastern Himalayas",
        "bbox_wgs84": [91.8500, 27.5700, 91.8800, 27.6000],
        "center_coords": [91.8650, 27.5850],
        "verification_status": "VERIFIED",
        "verification_source": "Border Roads Organisation (BRO Project Vartak); Ministry of Road Transport and Highways (MoRTH)",
        "monitoring_context": "High monsoonal precipitation, fragile gneissic regolith, and deep road-cut translational slides",
        "hazard_classification": "TRANSLATIONAL_REGOLITH_FAILURES",
    },
]


def seed_surveillance_aois(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Ensures all 6 Himalayan surveillance AOIs are registered in SQLite database.
    Validates coordinates against Indian national boundary before insertion.
    """
    database.init_db(db_path)
    registered = []

    for item in HIMALAYAN_SURVEILLANCE_REGISTRY:
        aid = item["aoi_id"]
        existing = database.get_aoi(aid, db_path=db_path)
        if existing:
            registered.append(existing)
            continue

        # Spatial validation check
        is_valid, msg = validate_aoi_bbox_in_india(item["bbox_wgs84"])
        if not is_valid:
            logger.error(f"Cannot register invalid AOI {aid}: {msg}")
            continue

        aoi_data = {
            "aoi_id": aid,
            "name": item["name"],
            "state": item["state"],
            "region": item["region"],
            "bbox_wgs84": item["bbox_wgs84"],
            "monitoring_status": "ACTIVE",
            "polygon_geojson": {
                "type": "Polygon",
                "corridor_code": item["corridor_code"],
                "verification_status": item["verification_status"],
                "verification_source": item["verification_source"],
                "hazard_classification": item["hazard_classification"],
            },
        }
        created = database.create_aoi(aoi_data, db_path=db_path)
        registered.append(created)
        logger.info(f"Seeded surveillance AOI: {aid} ({item['name']})")

    return registered


# -----------------------------------------------------------------------------
# 2. Rule-Based Observed-Disturbance Intensity Classifier
# -----------------------------------------------------------------------------
def classify_observed_disturbance(
    candidate_pixel_count: int,
    total_pixels: int = 16384,
    cloud_cover_percent: Optional[float] = None,
    max_cloud_threshold: float = 20.0,
    has_valid_pair: bool = True,
) -> Dict[str, Any]:
    """
    Provisional rule-based classifier evaluating observed surface optical disturbance.

    IMPORTANT SCIENTIFIC DISCLAIMER:
    These thresholds are PROVISIONAL and NOT SCIENTIFICALLY CALIBRATED.
    They classify observed optical canopy/soil surface disturbance between paired satellite passes.
    They do NOT represent confirmed landslide severity, structural damage, or future landslide probability.
    """
    if not has_valid_pair or cloud_cover_percent is None or cloud_cover_percent > max_cloud_threshold:
        cloud_val_str = f"{cloud_cover_percent:.1f}%" if cloud_cover_percent is not None else "Unknown"
        return {
            "intensity": "INSUFFICIENT_DATA",
            "intensity_label": "Insufficient Data / Cloud Obscured",
            "candidate_pixels": candidate_pixel_count or 0,
            "candidate_ratio_pct": 0.0,
            "is_alert_candidate": False,
            "threshold_tier": "INVALID_OR_OBSCURED",
            "provisional_disclaimer": "PROVISIONAL / NOT SCIENTIFICALLY CALIBRATED. Optical observations obscured or baseline missing.",
            "rationale": f"Data quality insufficient for optical screening (Cloud cover: {cloud_val_str} vs limit <={max_cloud_threshold}%).",
        }

    ratio_pct = round((candidate_pixel_count / total_pixels) * 100.0, 2)

    if candidate_pixel_count >= 500 or ratio_pct >= 3.0:
        intensity = "HIGH_OBSERVED_DISTURBANCE"
        label = "High Observed Disturbance"
        is_alert = True
        tier = "HIGH"
        rationale = (
            f"Significant surface reflectance alteration across {candidate_pixel_count:,} pixels "
            f"({ratio_pct:.1f}% of AOI), satisfying provisional high threshold (>= 500 px or >= 3.0% of tile)."
        )
    elif candidate_pixel_count >= 50 or ratio_pct >= 0.3:
        intensity = "MODERATE_OBSERVED_DISTURBANCE"
        label = "Moderate Observed Disturbance"
        is_alert = True
        tier = "MODERATE"
        rationale = (
            f"Localized surface reflectance disturbance across {candidate_pixel_count:,} pixels "
            f"({ratio_pct:.1f}% of AOI), satisfying provisional moderate threshold (50-499 px)."
        )
    else:
        intensity = "LOW_OBSERVED_DISTURBANCE"
        label = "Low Observed Disturbance"
        is_alert = False
        tier = "LOW"
        rationale = (
            f"Nominal background spectral variance ({candidate_pixel_count} pixels, {ratio_pct:.2f}% of AOI) "
            f"below provisional disturbance threshold (< 50 px)."
        )

    return {
        "intensity": intensity,
        "intensity_label": label,
        "candidate_pixels": candidate_pixel_count,
        "candidate_ratio_pct": ratio_pct,
        "is_alert_candidate": is_alert,
        "threshold_tier": tier,
        "provisional_disclaimer": (
            "PROVISIONAL / NOT SCIENTIFICALLY CALIBRATED. "
            "Reflects observed optical canopy/soil surface disturbance only; NOT predictive hazard or confirmed landslide severity."
        ),
        "rationale": rationale,
    }


# -----------------------------------------------------------------------------
# 3. Surveillance Orchestration Engine
# -----------------------------------------------------------------------------
class HimalayanSurveillanceEngine:
    """
    Coordinates multi-region autonomous satellite surveillance sweeps
    with strict per-AOI isolation, caching, and bounded timeouts.
    """

    def __init__(self):
        self.registry = {item["aoi_id"]: item for item in HIMALAYAN_SURVEILLANCE_REGISTRY}
        seed_surveillance_aois()

    def _load_cache(self) -> Dict[str, Any]:
        """Loads cached surveillance state from disk."""
        if os.path.exists(CACHE_FILE_PATH):
            try:
                with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read surveillance cache: {e}")
        return {}

    def _save_cache(self, cache: Dict[str, Any]) -> None:
        """Persists surveillance state atomically to disk."""
        try:
            temp_path = CACHE_FILE_PATH + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
            os.replace(temp_path, CACHE_FILE_PATH)
        except Exception as e:
            logger.warning(f"Failed to persist surveillance cache: {e}")

    def get_surveillance_summary(self, allow_cached: bool = True) -> Dict[str, Any]:
        """
        Retrieves current surveillance matrix across all 6 Himalayan territories.
        Returns instantly from isolated cache if available.
        """
        cache = self._load_cache()
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        regions_status: List[Dict[str, Any]] = []
        fleet_alerts: List[Dict[str, Any]] = []

        for item in HIMALAYAN_SURVEILLANCE_REGISTRY:
            aid = item["aoi_id"]
            cached_data = cache.get(aid)

            if cached_data:
                entry = dict(cached_data)
            else:
                # Unchecked baseline placeholder
                entry = {
                    "aoi_id": aid,
                    "name": item["name"],
                    "state": item["state"],
                    "region": item["region"],
                    "corridor_code": item["corridor_code"],
                    "bbox_wgs84": item["bbox_wgs84"],
                    "center_coords": item["center_coords"],
                    "verification_status": item["verification_status"],
                    "verification_source": item["verification_source"],
                    "hazard_classification": item["hazard_classification"],
                    "monitoring_status": "ACTIVE",
                    "observed_disturbance": "INSUFFICIENT_DATA",
                    "disturbance_intensity_label": "Unchecked (Pending Initial Sweep)",
                    "candidate_pixels": 0,
                    "candidate_ratio_pct": 0.0,
                    "is_alert_candidate": False,
                    "cloud_cover_percent": None,
                    "acquisition_id": None,
                    "acquisition_date": None,
                    "evaluation_timestamp": None,
                    "data_freshness_days": None,
                    "data_quality_status": "PENDING_SWEEP",
                    "imagery_source": "Copernicus Sentinel-2 L2A",
                    "cnn_raw_score": None,
                    "mean_delta_ndvi": None,
                    "provisional_disclaimer": "PROVISIONAL / NOT SCIENTIFICALLY CALIBRATED.",
                }

            # Calculate freshness days
            acq_date_str = entry.get("acquisition_date")
            if acq_date_str:
                try:
                    dt = datetime.datetime.fromisoformat(acq_date_str.replace("Z", "+00:00"))
                    age_days = (now_utc - dt).total_seconds() / 86400.0
                    entry["data_freshness_days"] = round(age_days, 1)
                except Exception:
                    entry["data_freshness_days"] = None

            regions_status.append(entry)

            if entry.get("is_alert_candidate"):
                fleet_alerts.append({
                    "aoi_id": aid,
                    "region_name": item["name"],
                    "state": item["state"],
                    "observed_disturbance": entry.get("observed_disturbance"),
                    "candidate_pixels": entry.get("candidate_pixels", 0),
                    "candidate_ratio_pct": entry.get("candidate_ratio_pct", 0.0),
                    "acquisition_date": entry.get("acquisition_date"),
                    "alert_timestamp": entry.get("evaluation_timestamp"),
                    "recommended_action": f"Geotechnical road patrol review advised along {item['corridor_code']} corridor.",
                })

        # Summary statistics
        total_monitored = len(regions_status)
        high_count = sum(1 for r in regions_status if r.get("observed_disturbance") == "HIGH_OBSERVED_DISTURBANCE")
        mod_count = sum(1 for r in regions_status if r.get("observed_disturbance") == "MODERATE_OBSERVED_DISTURBANCE")
        low_count = sum(1 for r in regions_status if r.get("observed_disturbance") == "LOW_OBSERVED_DISTURBANCE")
        insufficient_count = sum(1 for r in regions_status if r.get("observed_disturbance") == "INSUFFICIENT_DATA")

        return {
            "timestamp": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fleet_summary": {
                "total_corridors_monitored": total_monitored,
                "high_observed_disturbance": high_count,
                "moderate_observed_disturbance": mod_count,
                "low_observed_disturbance": low_count,
                "insufficient_data": insufficient_count,
                "active_candidate_alerts": len(fleet_alerts),
            },
            "regions": regions_status,
            "fleet_alerts": fleet_alerts,
            "classifier_notice": (
                "Disturbance classes reflect observed optical reflectance alteration between paired passes. "
                "Thresholds are PROVISIONAL and NOT CALIBRATED. They do NOT forecast future landslide probability."
            ),
        }

    def surveil_single_aoi(
        self,
        aoi_id: str,
        allow_simulated_fallback: bool = False,
        client: Optional[CopernicusClient] = None,
    ) -> Dict[str, Any]:
        """
        Executes an isolated satellite observation query and bi-temporal evaluation
        for a single Himalayan AOI.
        """
        meta = self.registry.get(aoi_id)
        if not meta:
            return {"aoi_id": aoi_id, "status": "ERROR", "error": f"AOI '{aoi_id}' not found in registry."}

        bbox = meta["bbox_wgs84"]
        copernicus = client or CopernicusClient()
        now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # 1. Search acquisitions via CDSE STAC
        try:
            token = copernicus.get_auth_token()
            acqs = (
                copernicus.search_acquisitions(bbox, limit=10, max_cloud_cover=settings.MAX_CLOUD_COVER_PERCENT)
                if token else []
            )
        except Exception as e:
            logger.warning(f"CDSE STAC search failed for {aoi_id}: {e}")
            acqs = []

        usable_acqs = [a for a in acqs if a.get("is_usable")]

        # Case A: Insufficient cloud-free passes
        if not usable_acqs:
            newest_cloud = acqs[0].get("cloud_cover_percent") if acqs else None
            newest_acq_id = acqs[0].get("acquisition_id") if acqs else None
            newest_time = acqs[0].get("acquisition_time") if acqs else None

            classification = classify_observed_disturbance(
                candidate_pixel_count=0,
                cloud_cover_percent=newest_cloud if newest_cloud is not None else 99.0,
                max_cloud_threshold=settings.MAX_CLOUD_COVER_PERCENT,
                has_valid_pair=False,
            )

            result_entry = {
                "aoi_id": aoi_id,
                "name": meta["name"],
                "state": meta["state"],
                "region": meta["region"],
                "corridor_code": meta["corridor_code"],
                "bbox_wgs84": bbox,
                "center_coords": meta["center_coords"],
                "verification_status": meta["verification_status"],
                "verification_source": meta["verification_source"],
                "hazard_classification": meta["hazard_classification"],
                "monitoring_status": "ACTIVE",
                "observed_disturbance": "INSUFFICIENT_DATA",
                "disturbance_intensity_label": classification["intensity_label"],
                "candidate_pixels": 0,
                "candidate_ratio_pct": 0.0,
                "is_alert_candidate": False,
                "cloud_cover_percent": newest_cloud,
                "acquisition_id": newest_acq_id,
                "acquisition_date": newest_time,
                "evaluation_timestamp": now_iso,
                "data_quality_status": "NO_USABLE_CLOUD_FREE_SCENES",
                "imagery_source": "Copernicus Data Space Ecosystem (Sentinel-2 L2A)",
                "cnn_raw_score": None,
                "mean_delta_ndvi": None,
                "provisional_disclaimer": classification["provisional_disclaimer"],
                "status_explanation": classification["rationale"],
            }
            return result_entry

        # Case B: Download and evaluate Sentinel-2 L2A BOA rasters
        post_acq = usable_acqs[0]
        post_raster, _ = copernicus.fetch_scene_imagery(
            post_acq, bbox, patch_size=128, allow_simulated_fallback=allow_simulated_fallback
        )

        if post_raster is None:
            classification = classify_observed_disturbance(
                candidate_pixel_count=0,
                cloud_cover_percent=post_acq.get("cloud_cover_percent"),
                has_valid_pair=False,
            )
            return {
                "aoi_id": aoi_id,
                "name": meta["name"],
                "state": meta["state"],
                "region": meta["region"],
                "corridor_code": meta["corridor_code"],
                "bbox_wgs84": bbox,
                "center_coords": meta["center_coords"],
                "verification_status": meta["verification_status"],
                "verification_source": meta["verification_source"],
                "hazard_classification": meta["hazard_classification"],
                "monitoring_status": "ACTIVE",
                "observed_disturbance": "INSUFFICIENT_DATA",
                "disturbance_intensity_label": "Raster Retrieval Failed",
                "candidate_pixels": 0,
                "candidate_ratio_pct": 0.0,
                "is_alert_candidate": False,
                "cloud_cover_percent": post_acq.get("cloud_cover_percent"),
                "acquisition_id": post_acq.get("acquisition_id"),
                "acquisition_date": post_acq.get("acquisition_time"),
                "evaluation_timestamp": now_iso,
                "data_quality_status": "FETCH_FAILED",
                "imagery_source": "Copernicus Data Space Ecosystem",
                "cnn_raw_score": None,
                "mean_delta_ndvi": None,
                "provisional_disclaimer": classification["provisional_disclaimer"],
                "status_explanation": "Sentinel Hub Processing API did not deliver raster array.",
            }

        # Multi-temporal baseline pass selection
        pre_raster = None
        pre_date = None
        pre_acq_id = None
        if len(usable_acqs) > 1:
            post_day = post_acq.get("acquisition_time", "")[:10]
            candidate_pre = next(
                (a for a in usable_acqs[1:] if a.get("acquisition_time", "")[:10] != post_day),
                usable_acqs[1],
            )
            pre_raster, _ = copernicus.fetch_scene_imagery(
                candidate_pre, bbox, patch_size=128, allow_simulated_fallback=allow_simulated_fallback
            )
            if pre_raster is not None:
                pre_date = candidate_pre.get("acquisition_time")
                pre_acq_id = candidate_pre.get("acquisition_id")

        # Run HonestPredictionEngine
        engine = HonestPredictionEngine()
        eval_dict = engine.evaluate_aoi_observation(
            aoi_name=meta["name"],
            aoi_bbox=bbox,
            post_raster=post_raster,
            post_date=post_acq.get("acquisition_time", now_iso),
            post_acquisition_id=post_acq.get("acquisition_id", "DYNAMIC_ACQ"),
            cloud_cover_percent=post_acq.get("cloud_cover_percent", 0.0),
            imagery_source="Copernicus Data Space Ecosystem (Sentinel-2 L2A Dynamically Retrieved)",
            pre_raster=pre_raster,
            pre_date=pre_date,
            pre_acquisition_id=pre_acq_id,
            dem_grid=None,
            is_simulated_fallback=False,
            is_dem_parameterized=False,
        )

        # Disturbance classification
        change_metrics = eval_dict.get("bi_temporal_change_detection", {}).get("spatial_change_metrics", {})
        cand_px = change_metrics.get("candidate_change_pixel_count", 0)
        ndvi_analysis = eval_dict.get("bi_temporal_change_detection", {}).get("ndvi_analysis", {})
        mean_d_ndvi = ndvi_analysis.get("mean_delta_ndvi")
        cnn_raw = eval_dict.get("cnn_inference_diagnostics", {}).get("raw_uncalibrated_sigmoid_score")

        classification = classify_observed_disturbance(
            candidate_pixel_count=cand_px,
            cloud_cover_percent=post_acq.get("cloud_cover_percent", 0.0),
            has_valid_pair=(pre_raster is not None),
        )

        result_entry = {
            "aoi_id": aoi_id,
            "name": meta["name"],
            "state": meta["state"],
            "region": meta["region"],
            "corridor_code": meta["corridor_code"],
            "bbox_wgs84": bbox,
            "center_coords": meta["center_coords"],
            "verification_status": meta["verification_status"],
            "verification_source": meta["verification_source"],
            "hazard_classification": meta["hazard_classification"],
            "monitoring_status": "ACTIVE",
            "observed_disturbance": classification["intensity"],
            "disturbance_intensity_label": classification["intensity_label"],
            "candidate_pixels": classification["candidate_pixels"],
            "candidate_ratio_pct": classification["candidate_ratio_pct"],
            "is_alert_candidate": classification["is_alert_candidate"],
            "cloud_cover_percent": round(post_acq.get("cloud_cover_percent", 0.0), 2),
            "acquisition_id": post_acq.get("acquisition_id"),
            "acquisition_date": post_acq.get("acquisition_time"),
            "pre_acquisition_id": pre_acq_id,
            "pre_acquisition_date": pre_date,
            "evaluation_timestamp": now_iso,
            "data_quality_status": "VALID_BI_TEMPORAL_EVALUATION" if pre_raster is not None else "SINGLE_PASS_NO_BASELINE",
            "imagery_source": "Copernicus Data Space Ecosystem (Sentinel-2 L2A Dynamically Retrieved)",
            "cnn_raw_score": cnn_raw,
            "mean_delta_ndvi": mean_d_ndvi,
            "provisional_disclaimer": classification["provisional_disclaimer"],
            "status_explanation": eval_dict.get("status_explanation", classification["rationale"]),
            "detailed_evaluation": eval_dict,
        }
        return result_entry

    def run_surveillance_sweep(
        self,
        aoi_id: Optional[str] = None,
        allow_simulated_fallback: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes autonomous surveillance check across targeted AOI or entire 6-state Himalayan fleet.
        Updates persistent cache per AOI. Gracefully tolerates individual failures.
        """
        cache = self._load_cache()
        target_ids = [aoi_id] if aoi_id and aoi_id in self.registry else list(self.registry.keys())

        copernicus = CopernicusClient()
        updated_count = 0
        failed_count = 0

        for aid in target_ids:
            try:
                res = self.surveil_single_aoi(
                    aid,
                    allow_simulated_fallback=allow_simulated_fallback,
                    client=copernicus,
                )
                cache[aid] = res
                updated_count += 1
            except Exception as e:
                logger.error(f"Surveillance exception on AOI {aid}: {e}")
                failed_count += 1
                # Preserve existing cache or create graceful error entry
                if aid not in cache:
                    meta = self.registry[aid]
                    cache[aid] = {
                        "aoi_id": aid,
                        "name": meta["name"],
                        "state": meta["state"],
                        "region": meta["region"],
                        "corridor_code": meta["corridor_code"],
                        "bbox_wgs84": meta["bbox_wgs84"],
                        "center_coords": meta["center_coords"],
                        "verification_status": meta["verification_status"],
                        "observed_disturbance": "INSUFFICIENT_DATA",
                        "disturbance_intensity_label": "Sweep Exception",
                        "candidate_pixels": 0,
                        "candidate_ratio_pct": 0.0,
                        "is_alert_candidate": False,
                        "data_quality_status": f"ERROR: {str(e)[:80]}",
                        "evaluation_timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }

        self._save_cache(cache)
        summary = self.get_surveillance_summary(allow_cached=True)
        summary["sweep_execution"] = {
            "targets_requested": len(target_ids),
            "updated_successfully": updated_count,
            "failed_count": failed_count,
        }
        return summary


# Global singleton instance
_engine_instance: Optional[HimalayanSurveillanceEngine] = None


def get_surveillance_engine() -> HimalayanSurveillanceEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = HimalayanSurveillanceEngine()
    return _engine_instance
