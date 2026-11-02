"""
India Boundary Spatial Validator.
Provides pure Python ray-casting point-in-polygon and bounding box containment
validation against simplified boundary polygons of India (mainland + islands).
Zero external C-library dependencies (no GDAL or Shapely required).
"""

from typing import List, Tuple, Dict, Any, Optional

# Simplified boundary polygon coordinates for Mainland India (Longitude, Latitude)
# Ordered counter-clockwise starting from Sir Creek / Kutch, Gujarat
MAINLAND_INDIA_POLYGON: List[Tuple[float, float]] = [
    (68.12, 23.70),  # Sir Creek / Kutch, Gujarat
    (68.70, 24.20),  # Great Rann of Kutch border
    (70.50, 24.50),  # Barmer border, Rajasthan
    (70.20, 27.20),  # Jaisalmer western border
    (71.50, 28.50),  # Bikaner border
    (73.80, 30.00),  # Punjab border (Firozpur)
    (74.80, 31.80),  # Amritsar / Wagah
    (74.30, 32.50),  # Jammu border
    (74.00, 34.00),  # Line of Control, J&K (Poonch/Uri)
    (74.50, 35.50),  # Northern J&K / Gilgit border
    (76.50, 37.10),  # Indira Col / Siachen / Karakoram
    (78.00, 35.50),  # Aksai Chin / Ladakh border
    (79.00, 34.00),  # Pangong Tso / Demchok
    (78.80, 32.50),  # Spiti / Kinnaur, HP border with Tibet
    (79.30, 31.00),  # Uttarkashi / Chamoli, Uttarakhand border
    (80.50, 30.50),  # Pithoragarh / Lipulekh (tri-junction)
    (80.20, 28.80),  # Indo-Nepal border (Pilibhit/Lakhimpur)
    (83.00, 27.50),  # Gorakhpur / Raxaul border
    (88.10, 26.80),  # Darjeeling / Nepal border
    (88.10, 27.80),  # Sikkim / Nepal border (Kanchenjunga)
    (88.70, 28.10),  # Sikkim / Tibet border (Nathu La)
    (88.90, 27.30),  # Sikkim / Bhutan border
    (90.00, 26.80),  # Assam / Bhutan border
    (91.80, 27.80),  # Tawang, Arunachal border
    (94.00, 28.80),  # Upper Subansiri / Siang, Arunachal
    (97.00, 28.30),  # Anjaw / Kibithu
    (97.40, 28.00),  # Easternmost point (Dong / Kibithu)
    (96.50, 27.00),  # Patkai hills (Arunachal/Myanmar)
    (95.00, 25.50),  # Nagaland / Myanmar border
    (94.00, 24.00),  # Manipur / Myanmar border
    (93.30, 22.50),  # Mizoram / Myanmar border
    (92.50, 21.90),  # Southernmost point of Mizoram
    (92.20, 23.00),  # Mizoram / Bangladesh border
    (91.20, 23.50),  # Tripura / Bangladesh border
    (91.80, 25.20),  # Meghalaya / Bangladesh border
    (89.80, 25.50),  # Western Meghalaya / Garo hills
    (88.50, 26.00),  # North Bengal / Siliguri corridor
    (88.00, 24.00),  # Malda / Murshidabad (Bangladesh border)
    (89.00, 21.60),  # Sundarbans / Bay of Bengal coast
    (87.00, 21.50),  # Odisha coast (Dhamra/Chandipur)
    (85.80, 19.80),  # Puri / Chilika lake
    (83.30, 17.70),  # Visakhapatnam coast
    (81.00, 16.00),  # Krishna / Godavari delta
    (80.30, 13.10),  # Chennai coast
    (79.80, 10.80),  # Point Calimere / Nagapattinam
    (79.30, 9.30),   # Rameswaram / Palk Strait
    (77.55, 8.08),   # Kanyakumari (Southern tip of mainland)
    (76.90, 8.50),   # Thiruvananthapuram coast, Kerala
    (76.20, 9.90),   # Kochi coast, Kerala
    (75.00, 12.50),  # Mangaluru coast, Karnataka
    (73.80, 15.40),  # Goa coast
    (72.80, 19.00),  # Mumbai coast, Maharashtra
    (72.80, 21.00),  # Surat / Gulf of Khambhat
    (72.00, 21.50),  # Bhavnagar / Saurashtra
    (70.00, 20.80),  # Diu / Somnath coast
    (69.00, 22.30),  # Dwarka, Gujarat
    (68.12, 23.70),  # Closing the loop at Koteshwar
]

# Andaman & Nicobar Islands Bounding Polygon
ANDAMAN_NICOBAR_POLYGON: List[Tuple[float, float]] = [
    (92.00, 6.50),
    (94.50, 6.50),
    (94.50, 14.00),
    (92.00, 14.00),
    (92.00, 6.50),
]

# Lakshadweep Islands Bounding Polygon
LAKSHADWEEP_POLYGON: List[Tuple[float, float]] = [
    (71.50, 8.00),
    (74.50, 8.00),
    (74.50, 12.50),
    (71.50, 12.50),
    (71.50, 8.00),
]

ALL_INDIA_POLYGONS = [
    MAINLAND_INDIA_POLYGON,
    ANDAMAN_NICOBAR_POLYGON,
    LAKSHADWEEP_POLYGON,
]


def point_in_polygon(lon: float, lat: float, polygon: List[Tuple[float, float]]) -> bool:
    """
    Ray-casting algorithm (even-odd rule) to test if point (lon, lat) is inside polygon.
    Returns True if inside, False otherwise.
    """
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if min(p1y, p2y) < lat <= max(p1y, p2y):
            if lon <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or lon <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def is_coordinate_in_india(lon: float, lat: float) -> bool:
    """Checks whether a single WGS84 point (lon, lat) falls within Indian territory."""
    for poly in ALL_INDIA_POLYGONS:
        if point_in_polygon(lon, lat, poly):
            return True
    return False


def validate_aoi_bbox_in_india(bbox: List[float]) -> Tuple[bool, str]:
    """
    Validates that a bounding box [min_lon, min_lat, max_lon, max_lat] is within India:
    1. Validates coordinate format, ordering, and reasonable extent (< 5 degrees span).
    2. Validates that the bounding box center is within Indian territory polygon.
    3. Validates that at least one corner or significant fraction lies within Indian territory.
    Returns:
        (is_valid: bool, error_message: str)
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False, "Bounding box must be a list of 4 numbers: [min_lon, min_lat, max_lon, max_lat]."

    min_lon, min_lat, max_lon, max_lat = bbox

    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        return False, f"Longitude values must be between -180 and 180. Got min_lon={min_lon}, max_lon={max_lon}."
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        return False, f"Latitude values must be between -90 and 90. Got min_lat={min_lat}, max_lat={max_lat}."

    if min_lon >= max_lon:
        return False, f"min_lon ({min_lon}) must be strictly less than max_lon ({max_lon})."
    if min_lat >= max_lat:
        return False, f"min_lat ({min_lat}) must be strictly less than max_lat ({max_lat})."

    # Restrict single AOI bounding box to realistic corridor scale (< 2.5 degrees ~ 275 km)
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat
    if lon_span > 2.5 or lat_span > 2.5:
        return False, f"AOI extent too large ({lon_span:.2f} deg x {lat_span:.2f} deg). Max allowable extent for monitored corridor is 2.5 degrees."

    # Center coordinate
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0

    # Test center point
    if not is_coordinate_in_india(center_lon, center_lat):
        return False, (
            f"AOI center coordinate ({center_lon:.4f}°E, {center_lat:.4f}°N) does not fall within "
            f"Indian territorial boundary polygon. Please verify coordinates."
        )

    # Test corners
    corners = [
        ("SW", min_lon, min_lat),
        ("SE", max_lon, min_lat),
        ("NW", min_lon, max_lat),
        ("NE", max_lon, max_lat),
    ]
    corners_in = sum(1 for _, cx, cy in corners if is_coordinate_in_india(cx, cy))
    if corners_in == 0:
        return False, (
            f"AOI bounding box corners fall completely outside Indian territorial boundary polygon."
        )

    return True, "Valid Indian Area of Interest."
