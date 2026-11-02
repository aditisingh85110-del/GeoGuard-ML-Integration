"""
=============================================================================
GEOGUARD AI — SATELLITE INTELLIGENCE API CLIENT
Module: modules/satellite_client.py
Purpose: Safe HTTP client interfacing with the GeoGuard ML Satellite Microservice
=============================================================================
Target Backend: FastAPI ML Microservice (default: http://127.0.0.1:8000)
Endpoints Connected:
  - GET  /health
  - GET  /api/v2/aois
  - GET  /api/v2/aois/{aoi_id}
  - POST /api/v2/monitoring/judge-evaluate
  - GET  /api/v2/monitoring/judge-demo
  - GET  /satellite-status
  - GET  /latest-prediction

Safety Guarantees:
  1. Explicit per-endpoint timeouts prevent frontend freezing.
  2. Connection/HTTP/JSON errors are returned as structured error objects;
     API failures are NEVER converted into false 'LOW_OBSERVED_RISK' or success.
  3. Chamoli baseline imagery is strictly quarantined and NEVER silently
     substituted for dynamic or failing Indian AOIs.
  4. 'INSUFFICIENT_DATA' is strictly preserved and never conflated with stability.
=============================================================================
"""

import os
import logging
from typing import Dict, Any, List, Optional
import requests

logger = logging.getLogger("geoguard_satellite_client")

# Configurable backend URL with standard local microservice default
DEFAULT_BACKEND_URL = os.getenv("GEOGUARD_ML_API_URL", "http://127.0.0.1:8000").rstrip("/")


class SatelliteAPIClient:
    """
    HTTP client for communicating with the GeoGuard FastAPI ML microservice.
    Provides structured error responses, explicit timeouts, and strict
    separation between historical baselines and dynamic satellite acquisitions.
    """

    def __init__(self, base_url: Optional[str] = None):
        configured_url = base_url or os.getenv("GEOGUARD_ML_API_URL", DEFAULT_BACKEND_URL)
        self.base_url = configured_url.rstrip("/")

    # -------------------------------------------------------------------------
    # 1. Health Check
    # -------------------------------------------------------------------------
    def check_health(self, timeout: float = 3.0) -> Dict[str, Any]:
        """
        Queries GET /health to verify the ML service, CNN models, and database.
        Returns a structured dictionary indicating whether the service is reachable.
        """
        url = f"{self.base_url}/health"
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "is_online": True,
                    "status": data.get("status", "healthy"),
                    "service": data.get("service", "GeoGuard ML Service"),
                    "model_loaded": data.get("model_loaded", False),
                    "model_version": data.get("model_version", "Unknown"),
                    "database_connected": data.get("database_connected", False),
                    "timestamp": data.get("timestamp"),
                    "error": None,
                    "url": self.base_url,
                }
            else:
                return {
                    "is_online": False,
                    "status": "degraded",
                    "service": "GeoGuard ML Service",
                    "model_loaded": False,
                    "model_version": "Unknown",
                    "database_connected": False,
                    "timestamp": None,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                    "url": self.base_url,
                }
        except requests.ConnectionError:
            return {
                "is_online": False,
                "status": "offline",
                "service": "GeoGuard ML Service (Unreachable)",
                "model_loaded": False,
                "model_version": "None",
                "database_connected": False,
                "timestamp": None,
                "error": f"Connection refused at {self.base_url}. Ensure ml_service is running.",
                "url": self.base_url,
            }
        except requests.Timeout:
            return {
                "is_online": False,
                "status": "timeout",
                "service": "GeoGuard ML Service (Timeout)",
                "model_loaded": False,
                "model_version": "None",
                "database_connected": False,
                "timestamp": None,
                "error": f"Health check timed out after {timeout:.1f}s.",
                "url": self.base_url,
            }
        except Exception as e:
            return {
                "is_online": False,
                "status": "error",
                "service": "GeoGuard ML Service",
                "model_loaded": False,
                "model_version": "None",
                "database_connected": False,
                "timestamp": None,
                "error": str(e),
                "url": self.base_url,
            }

    # -------------------------------------------------------------------------
    # 2. Registered AOIs (India-Wide Network)
    # -------------------------------------------------------------------------
    def list_aois(
        self,
        state: Optional[str] = None,
        status: Optional[str] = None,
        timeout: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Queries GET /api/v2/aois to retrieve registered landslide monitoring corridors.
        """
        url = f"{self.base_url}/api/v2/aois"
        params = {}
        if state:
            params["state"] = state
        if status:
            params["status"] = status

        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                aois = data.get("aois", [])
                return {
                    "success": True,
                    "total_count": data.get("total_count", len(aois)),
                    "aois": aois,
                    "error": None,
                }
            return {
                "success": False,
                "total_count": 0,
                "aois": [],
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            }
        except Exception as e:
            logger.warning(f"Failed to fetch AOIs from ML backend: {e}")
            return {
                "success": False,
                "total_count": 0,
                "aois": [],
                "error": str(e),
            }

    def get_aoi(self, aoi_id: str, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Queries GET /api/v2/aois/{aoi_id} to retrieve a single AOI's bounding box and metadata.
        """
        url = f"{self.base_url}/api/v2/aois/{aoi_id}"
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return {
                    "success": True,
                    "aoi": resp.json(),
                    "error": None,
                }
            return {
                "success": False,
                "aoi": None,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            }
        except Exception as e:
            return {
                "success": False,
                "aoi": None,
                "error": str(e),
            }

    # -------------------------------------------------------------------------
    # 3. Judge Evaluation (On-Demand Multi-Evidence Screening)
    # -------------------------------------------------------------------------
    def evaluate_aoi(
        self,
        aoi_id: Optional[str] = None,
        allow_simulated_fallback: bool = False,
        timeout: float = 45.0,
    ) -> Dict[str, Any]:
        """
        Calls POST /api/v2/monitoring/judge-evaluate with an AOI identifier.
        
        - If aoi_id is None, 'demo', or 'chamoli': evaluates the Chamoli disaster corridor baseline.
        - If aoi_id is a registered AOI ID (e.g. 'IN-UK-DEHRADUN-01'): performs live CDSE STAC
          retrieval and Sentinel-2 L2A bi-temporal screening.
        
        Strict Safety Contract:
          If retrieval or processing fails, an explicit error dict is returned.
          Under no circumstances is Chamoli imagery substituted or an error converted
          to LOW_OBSERVED_RISK.
        """
        url = f"{self.base_url}/api/v2/monitoring/judge-evaluate"
        params: Dict[str, Any] = {
            "allow_simulated_fallback": str(allow_simulated_fallback).lower()
        }
        if aoi_id and aoi_id not in ["demo", "chamoli"]:
            params["aoi_id"] = aoi_id
        elif aoi_id:
            params["aoi_id"] = aoi_id

        try:
            resp = requests.post(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                data["success"] = True
                return data
            elif resp.status_code == 404:
                return {
                    "success": False,
                    "risk_screening_status": "AOI_NOT_FOUND",
                    "final_candidate_status": "AOI_NOT_FOUND",
                    "status_explanation": f"AOI '{aoi_id}' is not registered in the ML database.",
                    "data_provenance": {
                        "imagery_source": "None",
                        "is_simulated_fallback": False,
                        "data_quality_status": "NOT_FOUND",
                    },
                    "limitations_and_missing_evidence": [f"AOI '{aoi_id}' was not found."],
                    "error": f"HTTP 404: AOI not found",
                }
            else:
                return {
                    "success": False,
                    "risk_screening_status": "BACKEND_ERROR",
                    "final_candidate_status": "BACKEND_ERROR",
                    "status_explanation": f"ML Service returned HTTP {resp.status_code}: {resp.text[:250]}",
                    "data_provenance": {
                        "imagery_source": "Unavailable",
                        "is_simulated_fallback": False,
                        "data_quality_status": "ERROR",
                    },
                    "limitations_and_missing_evidence": [f"Backend processing failure (HTTP {resp.status_code})."],
                    "error": f"HTTP {resp.status_code}",
                }
        except requests.ConnectionError as e:
            return {
                "success": False,
                "risk_screening_status": "CONNECTION_OFFLINE",
                "final_candidate_status": "CONNECTION_OFFLINE",
                "status_explanation": f"Unable to reach ML Backend at {self.base_url}. Ensure the FastAPI server is running on port 8000.",
                "data_provenance": {
                    "imagery_source": "Unavailable (Connection Offline)",
                    "is_simulated_fallback": False,
                    "data_quality_status": "OFFLINE",
                },
                "limitations_and_missing_evidence": [
                    "Connection refused to ML microservice.",
                    "No inference or change detection could be executed."
                ],
                "error": str(e),
            }
        except requests.Timeout as e:
            return {
                "success": False,
                "risk_screening_status": "EVALUATION_TIMEOUT",
                "final_candidate_status": "EVALUATION_TIMEOUT",
                "status_explanation": (
                    f"Satellite retrieval and evaluation timed out after {timeout:.0f}s. "
                    "Copernicus STAC or Sentinel Hub may be experiencing latency or rate limiting."
                ),
                "data_provenance": {
                    "imagery_source": "Copernicus Data Space Ecosystem (Timeout)",
                    "is_simulated_fallback": False,
                    "data_quality_status": "TIMEOUT",
                },
                "limitations_and_missing_evidence": [
                    f"CDSE STAC query or raster download exceeded {timeout:.0f}s threshold."
                ],
                "error": str(e),
            }
        except Exception as e:
            return {
                "success": False,
                "risk_screening_status": "UNEXPECTED_CLIENT_ERROR",
                "final_candidate_status": "UNEXPECTED_CLIENT_ERROR",
                "status_explanation": f"Client encountered an unexpected error: {str(e)}",
                "data_provenance": {
                    "imagery_source": "Unavailable",
                    "is_simulated_fallback": False,
                    "data_quality_status": "CLIENT_ERROR",
                },
                "limitations_and_missing_evidence": [str(e)],
                "error": str(e),
            }

    # -------------------------------------------------------------------------
    # 4. Judge-Ready Demonstration Cache
    # -------------------------------------------------------------------------
    def get_judge_demo(self, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Queries GET /api/v2/monitoring/judge-demo to fetch the verified Chamoli baseline.
        """
        url = f"{self.base_url}/api/v2/monitoring/judge-demo"
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                data["success"] = True
                return data
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                "risk_screening_status": "ERROR",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "risk_screening_status": "CONNECTION_OFFLINE",
            }

    # -------------------------------------------------------------------------
    # 5. Satellite & Macro Prediction Helpers
    # -------------------------------------------------------------------------
    def get_satellite_status(self, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Queries GET /satellite-status to check CDSE STAC connection and credentials status.
        """
        url = f"{self.base_url}/satellite-status"
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return {"success": True, "data": resp.json()}
            return {"success": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_latest_prediction(self, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Queries GET /latest-prediction for the most recent corridor prediction.
        """
        url = f"{self.base_url}/latest-prediction"
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return {"success": True, "data": resp.json()}
            return {"success": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # -------------------------------------------------------------------------
    # 6. Autonomous Himalayan Surveillance Fleet
    # -------------------------------------------------------------------------
    def get_surveillance_summary(self, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Queries GET /api/v2/surveillance/summary to fetch the 6-state Himalayan surveillance matrix.
        """
        url = f"{self.base_url}/api/v2/surveillance/summary"
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                data["success"] = True
                return data
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                "regions": [],
                "fleet_summary": {},
                "fleet_alerts": [],
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "regions": [],
                "fleet_summary": {},
                "fleet_alerts": [],
            }

    def trigger_surveillance_sweep(
        self,
        aoi_id: Optional[str] = None,
        allow_simulated_fallback: bool = False,
        timeout: float = 65.0,
    ) -> Dict[str, Any]:
        """
        Calls POST /api/v2/surveillance/sweep to trigger an autonomous surveillance check.
        """
        url = f"{self.base_url}/api/v2/surveillance/sweep"
        params: Dict[str, Any] = {
            "allow_simulated_fallback": str(allow_simulated_fallback).lower()
        }
        if aoi_id:
            params["aoi_id"] = aoi_id

        try:
            resp = requests.post(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                data["success"] = True
                return data
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                "regions": [],
                "fleet_summary": {},
                "fleet_alerts": [],
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "regions": [],
                "fleet_summary": {},
                "fleet_alerts": [],
            }


# -----------------------------------------------------------------------------
# Module Singleton & Ergonomic Functional Exports
# -----------------------------------------------------------------------------
_client_instance: Optional[SatelliteAPIClient] = None


def get_satellite_client(base_url: Optional[str] = None) -> SatelliteAPIClient:
    """Returns the singleton SatelliteAPIClient instance."""
    global _client_instance
    if _client_instance is None or (base_url and _client_instance.base_url != base_url.rstrip("/")):
        _client_instance = SatelliteAPIClient(base_url=base_url)
    return _client_instance


def check_health(base_url: Optional[str] = None) -> Dict[str, Any]:
    return get_satellite_client(base_url).check_health()


def list_aois(base_url: Optional[str] = None, state: Optional[str] = None, status: Optional[str] = None) -> Dict[str, Any]:
    return get_satellite_client(base_url).list_aois(state=state, status=status)


def get_aoi(aoi_id: str, base_url: Optional[str] = None) -> Dict[str, Any]:
    return get_satellite_client(base_url).get_aoi(aoi_id)


def evaluate_aoi(
    aoi_id: Optional[str] = None,
    allow_simulated_fallback: bool = False,
    base_url: Optional[str] = None,
    timeout: float = 45.0,
) -> Dict[str, Any]:
    return get_satellite_client(base_url).evaluate_aoi(
        aoi_id=aoi_id,
        allow_simulated_fallback=allow_simulated_fallback,
        timeout=timeout,
    )


def get_judge_demo(base_url: Optional[str] = None) -> Dict[str, Any]:
    return get_satellite_client(base_url).get_judge_demo()


def get_surveillance_summary(base_url: Optional[str] = None) -> Dict[str, Any]:
    return get_satellite_client(base_url).get_surveillance_summary()


def trigger_surveillance_sweep(
    aoi_id: Optional[str] = None,
    allow_simulated_fallback: bool = False,
    base_url: Optional[str] = None,
    timeout: float = 65.0,
) -> Dict[str, Any]:
    return get_satellite_client(base_url).trigger_surveillance_sweep(
        aoi_id=aoi_id,
        allow_simulated_fallback=allow_simulated_fallback,
        timeout=timeout,
    )
