"""
GeoGuard Landslide Detection System - Configuration Module.

Loads settings from environment variables (.env) with strict validation.
Ensures unconfirmed highway corridor coordinates are not invented.
"""

import os
import json
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def load_env_file() -> None:
    """Loads environment variables from .env file if present without external dependencies."""
    candidate_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    for env_path in candidate_paths:
        if os.path.isfile(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k not in os.environ and v:
                                os.environ[k] = v
                logger.info(f"Loaded configuration from {env_path}")
            except Exception as e:
                logger.warning(f"Could not read {env_path}: {e}")
            break


load_env_file()


def parse_bbox(bbox_str: Optional[str]) -> Optional[List[float]]:
    """Parses a comma-separated bounding box string: 'min_lon,min_lat,max_lon,max_lat'."""
    if not bbox_str or not bbox_str.strip():
        return None
    try:
        parts = [float(x.strip()) for x in bbox_str.split(",")]
        if len(parts) != 4:
            logger.warning(f"Invalid bbox string '{bbox_str}'. Expected 4 coordinates.")
            return None
        min_lon, min_lat, max_lon, max_lat = parts
        if min_lon >= max_lon or min_lat >= max_lat:
            logger.warning(f"Inverted bbox coordinates: {parts}")
            return None
        return [min_lon, min_lat, max_lon, max_lat]
    except Exception as e:
        logger.warning(f"Failed to parse bbox string '{bbox_str}': {e}")
        return None


class Settings:
    # --- Project Paths ---
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    REPORTS_DIR: str = os.path.join(BASE_DIR, "reports")
    LOGS_DIR: str = os.path.join(BASE_DIR, "logs")

    MODEL_PATH: str = os.getenv("MODEL_PATH", os.path.join(MODELS_DIR, "cnn_model.keras"))
    PREPROCESSING_CONFIG_PATH: str = os.getenv(
        "PREPROCESSING_CONFIG_PATH", os.path.join(MODELS_DIR, "preprocessing_config.json")
    )
    DATABASE_PATH: str = os.getenv(
        "DATABASE_PATH", os.path.join(DATA_DIR, "geoguard_predictions.db")
    )

    # --- AOI Configuration ---
    # The actual highway location must NOT be invented.
    # It must be provided via HIGHWAY_AOI_BBOX in .env.
    HIGHWAY_CORRIDOR_NAME: str = os.getenv(
        "HIGHWAY_CORRIDOR_NAME", "UNCONFIGURED_HIGHWAY_CORRIDOR"
    )
    HIGHWAY_AOI_BBOX: Optional[List[float]] = parse_bbox(os.getenv("HIGHWAY_AOI_BBOX", ""))

    # Explicitly labeled test bounding box used ONLY for unit/integration testing
    TEST_AOI_BBOX: List[float] = [76.95, 12.00, 77.05, 12.10]
    TEST_AOI_NAME: str = "TEST_DEVELOPMENT_AOI"

    # --- Copernicus Data Space Ecosystem (CDSE) ---
    CDSE_CLIENT_ID: Optional[str] = os.getenv("CDSE_CLIENT_ID", None)
    CDSE_CLIENT_SECRET: Optional[str] = os.getenv("CDSE_CLIENT_SECRET", None)
    CDSE_TOKEN_URL: str = os.getenv(
        "CDSE_TOKEN_URL",
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
    )
    CDSE_STAC_URL: str = os.getenv(
        "CDSE_STAC_URL", "https://stac.dataspace.copernicus.eu/v1/search"
    )
    CDSE_SH_PROCESS_URL: str = os.getenv(
        "CDSE_SH_PROCESS_URL", "https://sh.dataspace.copernicus.eu/api/v1/process"
    )

    # --- Pipeline & Model Parameters ---
    PATCH_SIZE: int = 128
    INPUT_CHANNELS: int = 4
    SELECTED_BANDS: List[str] = ["B02", "B03", "B04", "B08"]

    # Recommended decision threshold from validation tuning analysis (was 0.50, tuned to 0.40)
    DECISION_THRESHOLD: float = float(os.getenv("DECISION_THRESHOLD", "0.40"))
    LANDSLIDE_PIXEL_THRESHOLD: int = int(os.getenv("LANDSLIDE_PIXEL_THRESHOLD", "100"))

    MAX_CLOUD_COVER_PERCENT: float = float(os.getenv("MAX_CLOUD_COVER_PERCENT", "30.0"))
    MAX_INVALID_PIXEL_RATIO: float = float(os.getenv("MAX_INVALID_PIXEL_RATIO", "0.20"))
    MAX_TILE_REQUEST_DIM: int = int(os.getenv("MAX_TILE_REQUEST_DIM", "1500"))

    # Polling & Scheduler
    POLLING_INTERVAL_MINUTES: int = int(os.getenv("POLLING_INTERVAL_MINUTES", "15"))

    def get_active_aoi(self, allow_test_fallback: bool = True) -> Tuple[str, Optional[List[float]], bool]:
        """
        Returns (aoi_name, bbox, is_test_aoi).
        If highway corridor is unconfigured and allow_test_fallback is False, returns None for bbox.
        """
        if self.HIGHWAY_AOI_BBOX is not None:
            return self.HIGHWAY_CORRIDOR_NAME, self.HIGHWAY_AOI_BBOX, False
        if allow_test_fallback:
            return self.TEST_AOI_NAME, self.TEST_AOI_BBOX, True
        return self.HIGHWAY_CORRIDOR_NAME, None, False


settings = Settings()
