"""
Spatial Landslide Event Clustering and Deduplication.
Prevents overlapping 128x128 patches from inflating landslide event counts.
Groups proximate or overlapping positive patch detections into distinct,
macro-scale landslide events using connected-component spatial clustering.
"""

import math
from typing import List, Dict, Any, Tuple


def haversine_distance_meters(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Computes great-circle distance between two WGS84 points in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def bboxes_overlap(b1: List[float], b2: List[float], tolerance_deg: float = 0.002) -> bool:
    """
    Checks if two bounding boxes [min_lon, min_lat, max_lon, max_lat] overlap
    or are contiguous within tolerance_deg (~200 meters).
    """
    if b1[2] + tolerance_deg < b2[0] or b2[2] + tolerance_deg < b1[0]:
        return False
    if b1[3] + tolerance_deg < b2[1] or b2[3] + tolerance_deg < b1[1]:
        return False
    return True


def cluster_landslide_patches(
    positive_patches: List[Dict[str, Any]],
    max_cluster_distance_meters: float = 1400.0,
) -> List[Dict[str, Any]]:
    """
    Clusters overlapping and spatially contiguous positive landslide patches into
    distinct geographic landslide events.

    Args:
        positive_patches: List of patch dictionaries with keys:
            - patch_index: int
            - center_coords: {"lat": float, "lon": float}
            - bbox: [min_lon, min_lat, max_lon, max_lat]
            - probability: float (uncalibrated model score)
        max_cluster_distance_meters: Maximum center-to-center distance for linking patches.
            Default 1,400m corresponds to adjacent overlapping 1.28km patches (640m stride).

    Returns:
        List of clustered distinct landslide events, each containing:
            - event_id: str (e.g. "EVENT-01")
            - patch_count: int
            - constituent_patch_indices: List[int]
            - centroid: {"lat": float, "lon": float}
            - bounding_box: [min_lon, min_lat, max_lon, max_lat]
            - max_model_score: float (highest raw sigmoid score in cluster)
            - mean_model_score: float
            - risk_severity: str ("HIGH" or "MODERATE")
    """
    if not positive_patches:
        return []

    n = len(positive_patches)
    # Adjacency graph for connected components
    adj = {i: [] for i in range(n)}

    for i in range(n):
        p1 = positive_patches[i]
        c1 = p1.get("center_coords", {})
        lon1, lat1 = c1.get("lon", 0.0), c1.get("lat", 0.0)
        b1 = p1.get("bbox", [lon1, lat1, lon1, lat1])

        for j in range(i + 1, n):
            p2 = positive_patches[j]
            c2 = p2.get("center_coords", {})
            lon2, lat2 = c2.get("lon", 0.0), c2.get("lat", 0.0)
            b2 = p2.get("bbox", [lon2, lat2, lon2, lat2])

            # Check overlap or proximity
            dist = haversine_distance_meters(lon1, lat1, lon2, lat2)
            overlap = bboxes_overlap(b1, b2)

            if overlap or dist <= max_cluster_distance_meters:
                adj[i].append(j)
                adj[j].append(i)

    # Find connected components using Breadth-First Search
    visited = set()
    clusters = []

    for i in range(n):
        if i not in visited:
            component = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(component)

    # Format each cluster into a distinct event
    events = []
    for idx, comp in enumerate(clusters, start=1):
        patches_in_comp = [positive_patches[k] for k in comp]
        patch_indices = [p.get("patch_index", k) for k, p in zip(comp, patches_in_comp)]

        lons = [p["center_coords"]["lon"] for p in patches_in_comp if "center_coords" in p]
        lats = [p["center_coords"]["lat"] for p in patches_in_comp if "center_coords" in p]
        scores = [p.get("probability", 0.0) for p in patches_in_comp]

        # Envelope bounding box
        all_min_lons = [p["bbox"][0] for p in patches_in_comp if "bbox" in p and len(p["bbox"]) == 4]
        all_min_lats = [p["bbox"][1] for p in patches_in_comp if "bbox" in p and len(p["bbox"]) == 4]
        all_max_lons = [p["bbox"][2] for p in patches_in_comp if "bbox" in p and len(p["bbox"]) == 4]
        all_max_lats = [p["bbox"][3] for p in patches_in_comp if "bbox" in p and len(p["bbox"]) == 4]

        event_bbox = [
            min(all_min_lons) if all_min_lons else min(lons),
            min(all_min_lats) if all_min_lats else min(lats),
            max(all_max_lons) if all_max_lons else max(lons),
            max(all_max_lats) if all_max_lats else max(lats),
        ]

        max_score = float(max(scores)) if scores else 0.0
        mean_score = float(sum(scores) / len(scores)) if scores else 0.0
        patch_count = len(patches_in_comp)

        # High risk if multiple overlapping patches detect the landslide or peak score is high
        risk_severity = "HIGH" if (max_score >= 0.60 or patch_count >= 2) else "MODERATE"

        events.append(
            {
                "event_id": f"LS-EVENT-{idx:02d}",
                "patch_count": patch_count,
                "constituent_patch_indices": patch_indices,
                "centroid": {
                    "lon": round(sum(lons) / len(lons), 6),
                    "lat": round(sum(lats) / len(lats), 6),
                },
                "bounding_box": [round(c, 6) for c in event_bbox],
                "max_model_score": round(max_score, 4),
                "mean_model_score": round(mean_score, 4),
                "risk_severity": risk_severity,
                "notice": "Raw uncalibrated CNN model score. Multi-patch spatial cluster.",
            }
        )

    return events
