"""
GeoGuard Landslide Detection System - Spatial Tiling & Native 10m Resolution Engine.

Ensures that Sentinel-2 satellite data is requested and processed at its native 10m ground
sampling distance (GSD), preventing spatial downsampling artifacts (such as the 85.7m degradation
identified in Phase 6).

Key Capabilities:
1. Calculates exact native 10m pixel dimensions (W, H) for arbitrary geographic bounding boxes.
2. Derives single-patch bounding boxes matching the exact 1.28 km x 1.28 km footprint of Landslide4Sense.
3. Implements sliding-window overlapping patch extraction with coordinate tracking and georeferencing.
4. Preserves EPSG:4326 CRS coordinates, centroids, and pixel resolution metrics for every patch.
"""

import math
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Constants for WGS84 ellipsoid approximations
METERS_PER_DEGREE_LAT = 111132.95  # meters per degree latitude at equator
NATIVE_SENTINEL2_RESOLUTION_METERS = 10.0  # 10m native GSD for B02, B03, B04, B08
STANDARD_PATCH_PIXELS = 128  # 128x128 pixels
STANDARD_PATCH_SPAN_METERS = STANDARD_PATCH_PIXELS * NATIVE_SENTINEL2_RESOLUTION_METERS  # 1,280 meters (1.28 km)


def meters_per_degree_lon(latitude_deg: float) -> float:
    """Computes ground distance in meters per degree longitude at a given latitude."""
    lat_rad = math.radians(latitude_deg)
    return 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)


def compute_bbox_dimensions_meters(bbox: List[float]) -> Tuple[float, float, float]:
    """
    Computes width, height (in meters), and center latitude for a WGS84 bounding box.
    bbox: [min_lon, min_lat, max_lon, max_lat]
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2.0
    delta_lat = max_lat - min_lat
    delta_lon = max_lon - min_lon

    height_m = delta_lat * METERS_PER_DEGREE_LAT
    width_m = delta_lon * meters_per_degree_lon(center_lat)
    return width_m, height_m, center_lat


def calculate_native_10m_pixel_dims(
    bbox: List[float],
    target_res_m: float = NATIVE_SENTINEL2_RESOLUTION_METERS,
    min_pixels: int = STANDARD_PATCH_PIXELS,
) -> Tuple[int, int, float, float]:
    """
    Calculates the exact raster width and height required to achieve native target resolution.

    Returns:
        (width_pixels, height_pixels, effective_res_x_m, effective_res_y_m)
    """
    width_m, height_m, _ = compute_bbox_dimensions_meters(bbox)

    width_px = max(min_pixels, int(round(width_m / target_res_m)))
    height_px = max(min_pixels, int(round(height_m / target_res_m)))

    eff_res_x = width_m / width_px
    eff_res_y = height_m / height_px

    return width_px, height_px, eff_res_x, eff_res_y


def compute_128px_patch_bbox(
    center_lon: float,
    center_lat: float,
    span_meters: float = STANDARD_PATCH_SPAN_METERS,
) -> List[float]:
    """
    Computes the exact WGS84 bounding box for a single 128x128 patch covering
    1.28 km x 1.28 km around a given geographic center point.
    """
    m_per_deg_lon = meters_per_degree_lon(center_lat)
    delta_lat = (span_meters / 2.0) / METERS_PER_DEGREE_LAT
    delta_lon = (span_meters / 2.0) / m_per_deg_lon

    min_lon = round(center_lon - delta_lon, 6)
    max_lon = round(center_lon + delta_lon, 6)
    min_lat = round(center_lat - delta_lat, 6)
    max_lat = round(center_lat + delta_lat, 6)

    return [min_lon, min_lat, max_lon, max_lat]


def extract_sliding_window_patches(
    scene: np.ndarray,
    bbox: List[float],
    patch_size: int = STANDARD_PATCH_PIXELS,
    overlap_ratio: float = 0.20,
    max_invalid_ratio: float = 0.20,
) -> List[Dict[str, Any]]:
    """
    Extracts 128x128 patches from a satellite scene using a sliding window with overlap.
    Preserves exact geographic coordinates, bounding boxes, and centroids for each patch.

    Args:
        scene: np.ndarray of shape (H, W, 4)
        bbox: [min_lon, min_lat, max_lon, max_lat] in EPSG:4326
        patch_size: Spatial pixel dimensions (default 128)
        overlap_ratio: Fraction of overlap between adjacent patches (default 0.20 = 20%)
        max_invalid_ratio: Maximum fraction of invalid/corrupt pixels allowed

    Returns:
        List of patch dictionaries with metadata and array data.
    """
    H, W, C = scene.shape
    if C != 4:
        raise ValueError(f"Expected 4 channels, got {C}")

    min_lon, min_lat, max_lon, max_lat = bbox
    width_m, height_m, center_lat = compute_bbox_dimensions_meters(bbox)
    eff_res_x = width_m / W
    eff_res_y = height_m / H

    lon_per_pixel = (max_lon - min_lon) / W
    lat_per_pixel = (max_lat - min_lat) / H

    # Calculate stride in pixels
    stride = max(1, int(round(patch_size * (1.0 - overlap_ratio))))

    # Compute row and column start indices ensuring the scene is fully covered
    row_starts = list(range(0, max(1, H - patch_size + 1), stride))
    if not row_starts or (H >= patch_size and row_starts[-1] + patch_size < H):
        row_starts.append(H - patch_size)

    col_starts = list(range(0, max(1, W - patch_size + 1), stride))
    if not col_starts or (W >= patch_size and col_starts[-1] + patch_size < W):
        col_starts.append(W - patch_size)

    # Handle edge case where scene is smaller than patch_size
    if H < patch_size or W < patch_size:
        logger.warning(f"Scene dimensions ({H}x{W}) are smaller than patch size ({patch_size}x{patch_size})")
        return []

    patches = []
    patch_idx = 0

    for r_start in sorted(set(row_starts)):
        r_end = r_start + patch_size

        for c_start in sorted(set(col_starts)):
            c_end = c_start + patch_size

            patch_arr = scene[r_start:r_end, c_start:c_end, :]

            # Geographic bounding box of this patch
            p_min_lon = min_lon + c_start * lon_per_pixel
            p_max_lon = min_lon + c_end * lon_per_pixel
            # Latitude decreases as raster row increases
            p_max_lat = max_lat - r_start * lat_per_pixel
            p_min_lat = max_lat - r_end * lat_per_pixel

            centroid_lon = (p_min_lon + p_max_lon) / 2.0
            centroid_lat = (p_min_lat + p_max_lat) / 2.0

            # Measure invalid / nodata pixels
            nan_inf_mask = np.isnan(patch_arr).any(axis=-1) | np.isinf(patch_arr).any(axis=-1)
            zero_fill_mask = (patch_arr == 0).all(axis=-1)
            invalid_mask = nan_inf_mask | zero_fill_mask
            invalid_ratio = float(np.sum(invalid_mask) / (patch_size * patch_size))

            is_usable = invalid_ratio <= max_invalid_ratio

            patches.append({
                "patch_index": patch_idx,
                "row_start": r_start,
                "row_end": r_end,
                "col_start": c_start,
                "col_end": c_end,
                "patch_arr": patch_arr,
                "bbox_wgs84": [
                    round(p_min_lon, 6),
                    round(p_min_lat, 6),
                    round(p_max_lon, 6),
                    round(p_max_lat, 6),
                ],
                "centroid": {
                    "latitude": round(centroid_lat, 6),
                    "longitude": round(centroid_lon, 6),
                },
                "spatial_metadata": {
                    "patch_pixels": [patch_size, patch_size],
                    "spatial_coverage_km": [
                        round((p_max_lon - p_min_lon) * meters_per_degree_lon(centroid_lat) / 1000.0, 3),
                        round((p_max_lat - p_min_lat) * METERS_PER_DEGREE_LAT / 1000.0, 3),
                    ],
                    "effective_resolution_meters": [round(eff_res_x, 2), round(eff_res_y, 2)],
                    "crs": "EPSG:4326",
                },
                "invalid_ratio": round(invalid_ratio, 4),
                "is_usable": is_usable,
            })
            patch_idx += 1

    logger.info(f"Extracted {len(patches)} native-scale patches ({patch_size}x{patch_size}) with {overlap_ratio:.0%} overlap.")
    return patches
