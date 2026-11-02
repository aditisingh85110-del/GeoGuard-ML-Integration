"""
GeoGuard Landslide Detection System — Terrain & Topography Context Module.
Provides DEM elevation and slope analysis to contextualize optical change candidates.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def classify_slope_hazard(slope_deg: float) -> str:
    """Classifies slope angle into geomorphological hazard categories."""
    if slope_deg < 5.0:
        return "FLAT_LOW_INITIATION_HAZARD"
    elif slope_deg < 15.0:
        return "GENTLE_LOW_INITIATION_HAZARD"
    elif slope_deg < 30.0:
        return "MODERATE_SLOPE_HAZARD"
    elif slope_deg < 45.0:
        return "STEEP_HIGH_INITIATION_HAZARD"
    else:
        return "VERY_STEEP_ROCKFALL_AVALANCHE_HAZARD"


class TerrainContextAnalyzer:
    """
    Computes topographic slope and elevation context for monitored AOIs.
    """

    def __init__(self, default_pixel_spacing_m: float = 10.0):
        self.pixel_spacing_m = default_pixel_spacing_m

    def compute_slope_from_dem(self, dem_grid: np.ndarray) -> np.ndarray:
        """
        Computes slope angle in degrees using Horn's method / 2nd order central difference.
        """
        if dem_grid.ndim != 2:
            raise ValueError(f"Expected 2D DEM grid, got shape {dem_grid.shape}")

        dz_dy, dz_dx = np.gradient(dem_grid, self.pixel_spacing_m, self.pixel_spacing_m)
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)
        return slope_deg.astype(np.float32)

    def analyze_terrain(
        self,
        bbox: List[float],
        dem_grid: Optional[np.ndarray] = None,
        dem_source: Optional[str] = None,
        is_parameterized: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyzes terrain context for a given AOI bbox.
        If DEM grid is not supplied or unavailable, safely returns structured UNAVAILABLE context.
        If DEM grid is parameterized/synthetic, clearly labels dem_source as PARAMETERIZED.
        """
        if dem_grid is None:
            return {
                "status": "UNAVAILABLE",
                "dem_source": None,
                "is_parameterized": False,
                "elevation_context": {
                    "min_elevation_m": None,
                    "mean_elevation_m": None,
                    "max_elevation_m": None,
                },
                "slope_context": {
                    "mean_slope_degrees": None,
                    "max_slope_degrees": None,
                    "hazard_classification": "UNKNOWN_TERRAIN_CONTEXT",
                },
                "note": "Digital Elevation Model (DEM) data was not provided for this pass. Optical change detection continues without topographic constraint.",
            }

        slope_grid = self.compute_slope_from_dem(dem_grid)
        mean_elev = float(np.mean(dem_grid))
        min_elev = float(np.min(dem_grid))
        max_elev = float(np.max(dem_grid))

        mean_slope = float(np.mean(slope_grid))
        max_slope = float(np.max(slope_grid))
        hazard = classify_slope_hazard(mean_slope)

        default_source = (
            "PARAMETERIZED_TOPOGRAPHIC_MODEL (Alpine Valley Profile)"
            if is_parameterized
            else "Copernicus GLO-30 DEM / Topographic Grid"
        )
        resolved_source = dem_source or default_source

        return {
            "status": "AVAILABLE",
            "dem_source": resolved_source,
            "is_parameterized": is_parameterized,
            "elevation_context": {
                "min_elevation_m": round(min_elev, 1),
                "mean_elevation_m": round(mean_elev, 1),
                "max_elevation_m": round(max_elev, 1),
            },
            "slope_context": {
                "mean_slope_degrees": round(mean_slope, 1),
                "max_slope_degrees": round(max_slope, 1),
                "hazard_classification": hazard,
            },
            "interpretation": (
                f"Mean terrain slope of {mean_slope:.1f}° classified as {hazard}. "
                "Steep terrain (>15°) supports initiation; flatter valley terrain (<15°) can host debris deposition."
            ),
        }
