"""
GeoGuard Landslide Detection System - Radiometric Calibration & Conversion Module.

Converts Landslide4Sense benchmark HDF5 patches to true physical Bottom-of-Atmosphere
(BOA) surface reflectance [0.0, 1.0], harmonizing them with real Copernicus Sentinel-2 L2A observations.

Scientific Basis:
- The IARAI Landslide4Sense benchmark dataset creators pre-normalized Sentinel-2 raw Digital Numbers (DN)
  and ALOS PALSAR terrain data by dividing each band by its dataset-wide mean vector (Issue #4, #12 in official repo).
- Standard Sentinel-2 L2A surface reflectance is related to raw DN by:
      Reflectance = DN / 10000.0
- Therefore, the true physical BOA surface reflectance for any Landslide4Sense band c is:
      Reflectance_c = HDF5_c * (img_mean_c / 10000.0)
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Complete 14-channel mean vector published by the IARAI Landslide4Sense benchmark authors
IARAI_BAND_MEANS_ALL = [
    1111.81236406,  # B01 - Coastal Aerosol (DN)
    824.63171476,   # B02 - Blue (DN)
    663.41636217,   # B03 - Green (DN)
    445.17289745,   # B04 - Red (DN)
    645.85829260,   # B05 - Red Edge 1 (DN)
    1547.73508126,  # B06 - Red Edge 2 (DN)
    1960.44401001,  # B07 - Red Edge 3 (DN)
    1941.32229668,  # B08 - NIR (DN)
    674.07572865,   # B09 - Water Vapour (DN)
    9.04787384,     # B10 - SWIR Cirrus (DN)
    1113.98338755,  # B11 - SWIR 1 (DN)
    519.90397929,   # B12 - SWIR 2 (DN)
    20.29228266,    # B13 - Slope (degrees)
    772.83144788,   # B14 - DEM (elevation in meters)
]

# Core 4-channel subset: [B02, B03, B04, B08] at indices [1, 2, 3, 7] in Landslide4Sense
OPTICAL_4CH_INDICES = [1, 2, 3, 7]
OPTICAL_4CH_NAMES = ["B02", "B03", "B04", "B08"]

# Multiplicative scale factors: img_mean / 10000.0
# Converts HDF5 values -> Physical Sentinel-2 BOA Surface Reflectance [0.0, 1.0]
OPTICAL_4CH_BOA_SCALE_FACTORS = np.array(
    [IARAI_BAND_MEANS_ALL[idx] / 10000.0 for idx in OPTICAL_4CH_INDICES],
    dtype=np.float32,
)  # ~ [0.082463, 0.066342, 0.044517, 0.194132]

# Valid physical BOA reflectance bounds for non-cloud terrestrial Earth observation
PHYSICAL_BOA_BOUNDS = {
    "B02": {"min": 0.001, "max": 0.85, "typical_mean_range": (0.02, 0.20)},
    "B03": {"min": 0.001, "max": 0.85, "typical_mean_range": (0.02, 0.20)},
    "B04": {"min": 0.001, "max": 0.85, "typical_mean_range": (0.01, 0.25)},
    "B08": {"min": 0.001, "max": 0.95, "typical_mean_range": (0.08, 0.60)},
}


def convert_l4s_h5_to_boa(
    patch: np.ndarray,
    scale_factors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Converts a Landslide4Sense 4-channel patch (B02, B03, B04, B08)
    from benchmark HDF5 normalized units to physical BOA reflectance.

    Args:
        patch: np.ndarray of shape (H, W, 4) or (N, H, W, 4)
        scale_factors: optional custom 4-element scale array

    Returns:
        np.ndarray with same shape in physical BOA reflectance [0.0, 1.0]
    """
    scales = scale_factors if scale_factors is not None else OPTICAL_4CH_BOA_SCALE_FACTORS
    scales = np.asarray(scales, dtype=np.float32)

    if patch.ndim == 3:
        if patch.shape[-1] != 4:
            raise ValueError(f"Expected 4 channels in last axis, got {patch.shape}")
        return (patch * scales).astype(np.float32)
    elif patch.ndim == 4:
        if patch.shape[-1] != 4:
            raise ValueError(f"Expected 4 channels in last axis, got {patch.shape}")
        return (patch * scales.reshape(1, 1, 1, 4)).astype(np.float32)
    else:
        raise ValueError(f"Expected 3D or 4D array, got ndim={patch.ndim}")


def validate_satellite_radiometry(
    patch: np.ndarray,
    max_invalid_pixel_ratio: float = 0.15,
    max_cloud_reflectance_threshold: float = 0.65,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Scientifically verifies that an input patch is a valid optical Sentinel-2
    Bottom-of-Atmosphere (BOA) raster suitable for CNN inference.

    Rejects:
    - Out-of-distribution unit-mismatched data
    - Dense clouds or snow saturation
    - High missing/nodata/corrupted pixel ratios

    Returns:
        (is_valid, status_code, diagnostics_dict)
    """
    if patch.ndim != 3 or patch.shape[-1] != 4:
        return False, "INVALID_SHAPE", {"error": f"Expected shape (H, W, 4), got {patch.shape}"}

    H, W, C = patch.shape
    total_pixels = H * W

    # 1. Check NaN / Inf / Zero-fill corrupted pixels
    nan_mask = np.isnan(patch).any(axis=-1)
    inf_mask = np.isinf(patch).any(axis=-1)
    nodata_mask = (patch == 0).all(axis=-1)
    invalid_mask = nan_mask | inf_mask | nodata_mask
    invalid_ratio = float(np.sum(invalid_mask) / total_pixels)

    if invalid_ratio > max_invalid_pixel_ratio:
        return False, "INSUFFICIENT_DATA", {
            "reason": f"High invalid pixel ratio: {invalid_ratio:.1%} > {max_invalid_pixel_ratio:.1%}",
            "invalid_ratio": invalid_ratio,
        }

    # 2. Check band physical distributions
    band_diagnostics = {}
    valid_pixels = patch[~invalid_mask]
    if len(valid_pixels) == 0:
        return False, "INSUFFICIENT_DATA", {"reason": "Patch contains 100% invalid pixels"}

    for i, bname in enumerate(OPTICAL_4CH_NAMES):
        b_vals = valid_pixels[:, i]
        b_mean = float(np.mean(b_vals))
        b_min = float(np.min(b_vals))
        b_max = float(np.max(b_vals))

        band_diagnostics[bname] = {
            "mean": round(b_mean, 4),
            "min": round(b_min, 4),
            "max": round(b_max, 4),
        }

        # Check if values appear to be raw DN (e.g. 500 - 4000) or Landslide4Sense HDF5 unscaled
        if b_mean > 2.0:
            return False, "RADIOMETRIC_MISMATCH", {
                "reason": f"Band {bname} mean ({b_mean:.2f}) exceeds physical BOA reflectance ceiling (1.0). "
                          "Data appears to be uncalibrated raw DN or unscaled HDF5.",
                "band_diagnostics": band_diagnostics,
            }

    # 3. Check for cloud saturation (B02 and B04 > cloud threshold)
    b02_vals = valid_pixels[:, 0]
    b04_vals = valid_pixels[:, 2]
    cloud_mask = (b02_vals > max_cloud_reflectance_threshold) & (b04_vals > max_cloud_reflectance_threshold)
    cloud_ratio = float(np.sum(cloud_mask) / len(valid_pixels))

    if cloud_ratio > 0.40:
        return False, "INSUFFICIENT_DATA", {
            "reason": f"Heavy cloud contamination detected ({cloud_ratio:.1%} of valid pixels saturated)",
            "cloud_ratio": round(cloud_ratio, 4),
            "band_diagnostics": band_diagnostics,
        }

    return True, "VALID", {
        "invalid_ratio": round(invalid_ratio, 4),
        "cloud_ratio": round(cloud_ratio, 4),
        "band_diagnostics": band_diagnostics,
    }
