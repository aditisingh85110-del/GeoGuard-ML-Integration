"""
GeoGuard Landslide Detection System - SQLite Database Module.

Provides persistent storage for:
- Satellite acquisition inference history
- AOI macro predictions and high-risk coordinate sets
- Scheduler polling and catalog discovery logs
"""

import os
import json
import sqlite3
import datetime
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "geoguard_predictions.db"
)


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Creates a sqlite3 connection with row dictionary access."""
    path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initializes tables in the SQLite database if they do not exist."""
    path = db_path or DEFAULT_DB_PATH
    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                acquisition_id TEXT,
                prediction_class TEXT NOT NULL,
                probability REAL NOT NULL,
                decision_threshold REAL NOT NULL,
                acquisition_timestamp TEXT,
                processing_timestamp TEXT NOT NULL,
                cloud_cover_percent REAL,
                aoi_name TEXT NOT NULL,
                aoi_bbox TEXT,
                data_status TEXT NOT NULL,
                is_simulated INTEGER NOT NULL DEFAULT 0,
                model_version TEXT NOT NULL,
                total_patches INTEGER NOT NULL DEFAULT 0,
                landslide_patches INTEGER NOT NULL DEFAULT 0,
                high_risk_coordinates_json TEXT,
                patch_details_json TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_predictions_acq_id 
            ON predictions(acquisition_id)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_predictions_proc_time 
            ON predictions(processing_timestamp DESC)
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS polling_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checked_at TEXT NOT NULL,
                status TEXT NOT NULL,
                latest_catalog_acquisition_id TEXT,
                message TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_polling_logs_checked_at 
            ON polling_logs(checked_at DESC)
            """
        )

        # --- Phase 8 Multi-AOI Tables ---
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS monitored_aois (
                aoi_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                state TEXT NOT NULL,
                region TEXT NOT NULL,
                bbox_wgs84_json TEXT NOT NULL,
                polygon_geojson TEXT,
                monitoring_status TEXT NOT NULL DEFAULT 'ACTIVE',
                last_acquisition_id TEXT,
                last_acquisition_time TEXT,
                last_quality_status TEXT NOT NULL DEFAULT 'UNCHECKED',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_aois_status 
            ON monitored_aois(monitoring_status)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_aois_state 
            ON monitored_aois(state)
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_acquisitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                aoi_id TEXT NOT NULL,
                acquisition_id TEXT NOT NULL,
                acquisition_time TEXT,
                cloud_cover REAL,
                processing_status TEXT NOT NULL,
                total_patches_extracted INTEGER DEFAULT 0,
                landslide_patches_detected INTEGER DEFAULT 0,
                landslide_events_clustered INTEGER DEFAULT 0,
                processed_at TEXT NOT NULL,
                UNIQUE(aoi_id, acquisition_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_processed_acq_lookup 
            ON processed_acquisitions(aoi_id, acquisition_id)
            """
        )
        conn.commit()
    logger.info(f"Database initialized at: {path}")


def save_prediction(pred_data: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """
    Saves an AOI macro prediction result to the database.
    Expects dictionary format produced by SatellitePipeline.process_aoi or standardized dict.
    """
    path = db_path or DEFAULT_DB_PATH
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Extract fields gracefully from pipeline response structure
    pred_meta = pred_data.get("prediction", {})
    sat_meta = pred_data.get("satellite", {})
    aoi_meta = pred_data.get("aoi", {})
    agg_meta = pred_data.get("aggregation", {})

    prediction_class = pred_meta.get("overall_class", pred_data.get("prediction_class", "UNKNOWN"))
    probability = float(pred_meta.get("max_probability", pred_data.get("probability", 0.0)))
    decision_threshold = float(pred_meta.get("decision_threshold", pred_data.get("decision_threshold", 0.40)))
    model_version = str(pred_meta.get("model_version", pred_data.get("model_version", "GeoGuard_CNN_v1")))

    acquisition_id = sat_meta.get("acquisition_id", pred_data.get("acquisition_id"))
    acquisition_timestamp = sat_meta.get("acquisition_time", pred_data.get("acquisition_timestamp", now_iso))
    cloud_cover_percent = sat_meta.get("cloud_cover_percent", pred_data.get("cloud_cover_percent"))
    processing_timestamp = pred_data.get("processing_timestamp", now_iso)

    aoi_name = aoi_meta.get("name", pred_data.get("aoi_name", "UNKNOWN_AOI"))
    aoi_bbox = json.dumps(aoi_meta.get("bbox", pred_data.get("aoi_bbox", [])))

    data_status = pred_data.get("data_status", "VALID")
    mode = sat_meta.get("mode", "")
    is_simulated = 1 if (data_status == "SIMULATED" or mode == "SIMULATED" or aoi_meta.get("is_test_aoi", False)) else 0

    total_patches = int(agg_meta.get("total_patches_evaluated", pred_data.get("total_patches", 1)))
    landslide_patches = int(agg_meta.get("landslide_patches", pred_data.get("landslide_patches", 1 if prediction_class == "LANDSLIDE DETECTED" else 0)))

    high_risk_coords = agg_meta.get("high_risk_coordinates", pred_data.get("high_risk_coordinates", []))
    high_risk_json = json.dumps(high_risk_coords)

    patch_details = pred_data.get("patch_details", [])
    patch_details_json = json.dumps(patch_details)

    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (
                acquisition_id, prediction_class, probability, decision_threshold,
                acquisition_timestamp, processing_timestamp, cloud_cover_percent,
                aoi_name, aoi_bbox, data_status, is_simulated, model_version,
                total_patches, landslide_patches, high_risk_coordinates_json,
                patch_details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                acquisition_id,
                prediction_class,
                probability,
                decision_threshold,
                acquisition_timestamp,
                processing_timestamp,
                cloud_cover_percent,
                aoi_name,
                aoi_bbox,
                data_status,
                is_simulated,
                model_version,
                total_patches,
                landslide_patches,
                high_risk_json,
                patch_details_json,
            ),
        )
        conn.commit()
        inserted_id = cursor.lastrowid

    logger.info(f"Saved prediction #{inserted_id}: {prediction_class} (prob={probability:.4f}) for acq={acquisition_id}")
    return inserted_id


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Converts a sqlite3.Row from predictions table into a standardized dictionary."""
    d = dict(row)
    # Parse JSON fields safely
    for field in ["aoi_bbox", "high_risk_coordinates_json", "patch_details_json"]:
        if field in d and d[field]:
            try:
                parsed_name = field.replace("_json", "")
                d[parsed_name] = json.loads(d[field])
            except Exception:
                pass
    return d


def get_latest_prediction(db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves the most recent macro prediction from the database."""
    path = db_path or DEFAULT_DB_PATH
    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM predictions 
            ORDER BY id DESC LIMIT 1
            """
        )
        row = cursor.fetchone()
        if not row:
            return None
        return row_to_dict(row)


def get_prediction_history(
    limit: int = 50, offset: int = 0, db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves a paginated list of historical predictions ordered newest first."""
    path = db_path or DEFAULT_DB_PATH
    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM predictions 
            ORDER BY id DESC 
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        rows = cursor.fetchall()
        return [row_to_dict(r) for r in rows]


def get_prediction_count(db_path: Optional[str] = None) -> int:
    """Returns the total number of saved predictions."""
    path = db_path or DEFAULT_DB_PATH
    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        return cursor.fetchone()[0]


def is_acquisition_processed(
    acquisition_id: Optional[str],
    aoi_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> bool:
    """
    Checks whether an acquisition ID has already been processed and saved.
    If aoi_id is provided, checks both processed_acquisitions table and predictions table.
    """
    if not acquisition_id:
        return False
    path = db_path or DEFAULT_DB_PATH
    with get_connection(path) as conn:
        cursor = conn.cursor()
        if aoi_id:
            cursor.execute(
                "SELECT 1 FROM processed_acquisitions WHERE aoi_id = ? AND acquisition_id = ? LIMIT 1",
                (aoi_id, acquisition_id),
            )
            if cursor.fetchone() is not None:
                return True
        cursor.execute(
            "SELECT 1 FROM predictions WHERE acquisition_id = ? LIMIT 1",
            (acquisition_id,),
        )
        return cursor.fetchone() is not None


def aoi_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Converts a sqlite3.Row from monitored_aois table into a dictionary."""
    d = dict(row)
    if "bbox_wgs84_json" in d and d["bbox_wgs84_json"]:
        try:
            d["bbox_wgs84"] = json.loads(d["bbox_wgs84_json"])
        except Exception:
            d["bbox_wgs84"] = []
    else:
        d["bbox_wgs84"] = []
    if "polygon_geojson" in d and d["polygon_geojson"]:
        try:
            d["polygon_geojson"] = json.loads(d["polygon_geojson"])
        except Exception:
            pass
    return d


def create_aoi(aoi_data: Dict[str, Any], db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Registers a new Area of Interest in the monitored_aois table.
    """
    path = db_path or DEFAULT_DB_PATH
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    aoi_id = aoi_data["aoi_id"]
    name = aoi_data["name"]
    state = aoi_data["state"]
    region = aoi_data["region"]
    bbox = aoi_data["bbox_wgs84"]
    bbox_json = json.dumps(bbox)
    polygon_json = json.dumps(aoi_data.get("polygon_geojson")) if aoi_data.get("polygon_geojson") else None
    status = aoi_data.get("monitoring_status", "ACTIVE")

    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO monitored_aois (
                aoi_id, name, state, region, bbox_wgs84_json,
                polygon_geojson, monitoring_status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (aoi_id, name, state, region, bbox_json, polygon_json, status, now_iso, now_iso),
        )
        conn.commit()

    logger.info(f"Registered monitored AOI: {aoi_id} ({name}, {state})")
    return get_aoi(aoi_id, db_path=path)  # type: ignore


def get_aoi(aoi_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves an AOI by its unique aoi_id."""
    path = db_path or DEFAULT_DB_PATH
    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM monitored_aois WHERE aoi_id = ?", (aoi_id,))
        row = cursor.fetchone()
        return aoi_row_to_dict(row) if row else None


def list_aois(
    state: Optional[str] = None,
    status: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Lists registered AOIs with optional state and monitoring_status filters."""
    path = db_path or DEFAULT_DB_PATH
    query = "SELECT * FROM monitored_aois WHERE 1=1"
    params: List[Any] = []
    if state:
        query += " AND state = ?"
        params.append(state)
    if status:
        query += " AND monitoring_status = ?"
        params.append(status)
    query += " ORDER BY aoi_id ASC"

    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        return [aooi_dict for aooi_dict in [aoi_row_to_dict(r) for r in rows]]


def update_aoi(
    aoi_id: str,
    update_data: Dict[str, Any],
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Updates fields of an existing AOI."""
    path = db_path or DEFAULT_DB_PATH
    existing = get_aoi(aoi_id, db_path=path)
    if not existing:
        return None

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    fields: List[str] = []
    values: List[Any] = []

    if "name" in update_data and update_data["name"] is not None:
        fields.append("name = ?")
        values.append(update_data["name"])
    if "state" in update_data and update_data["state"] is not None:
        fields.append("state = ?")
        values.append(update_data["state"])
    if "region" in update_data and update_data["region"] is not None:
        fields.append("region = ?")
        values.append(update_data["region"])
    if "bbox_wgs84" in update_data and update_data["bbox_wgs84"] is not None:
        fields.append("bbox_wgs84_json = ?")
        values.append(json.dumps(update_data["bbox_wgs84"]))
    if "polygon_geojson" in update_data and update_data["polygon_geojson"] is not None:
        fields.append("polygon_geojson = ?")
        values.append(json.dumps(update_data["polygon_geojson"]))
    if "monitoring_status" in update_data and update_data["monitoring_status"] is not None:
        fields.append("monitoring_status = ?")
        values.append(update_data["monitoring_status"])

    fields.append("updated_at = ?")
    values.append(now_iso)

    values.append(aoi_id)
    sql = f"UPDATE monitored_aois SET {', '.join(fields)} WHERE aoi_id = ?"

    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, tuple(values))
        conn.commit()

    return get_aoi(aoi_id, db_path=path)


def delete_aoi(
    aoi_id: str,
    hard_delete: bool = False,
    db_path: Optional[str] = None,
) -> bool:
    """
    Deactivates (sets status='ARCHIVED') or hard-deletes an AOI.
    Default soft-delete preserves historical data integrity.
    """
    path = db_path or DEFAULT_DB_PATH
    existing = get_aoi(aoi_id, db_path=path)
    if not existing:
        return False

    with get_connection(path) as conn:
        cursor = conn.cursor()
        if hard_delete:
            cursor.execute("DELETE FROM monitored_aois WHERE aoi_id = ?", (aoi_id,))
        else:
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            cursor.execute(
                "UPDATE monitored_aois SET monitoring_status = 'ARCHIVED', updated_at = ? WHERE aoi_id = ?",
                (now_iso, aoi_id),
            )
        conn.commit()
    return True


def record_processed_acquisition(
    record_data: Dict[str, Any],
    db_path: Optional[str] = None,
) -> int:
    """
    Records that a Sentinel-2 acquisition was processed for a specific AOI to prevent re-processing.
    """
    path = db_path or DEFAULT_DB_PATH
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    aoi_id = record_data["aoi_id"]
    acq_id = record_data["acquisition_id"]
    acq_time = record_data.get("acquisition_time")
    cloud_cover = record_data.get("cloud_cover")
    status = record_data.get("processing_status", "SUCCESS")
    total_patches = record_data.get("total_patches_extracted", 0)
    landslide_patches = record_data.get("landslide_patches_detected", 0)
    landslide_events = record_data.get("landslide_events_clustered", 0)

    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO processed_acquisitions (
                aoi_id, acquisition_id, acquisition_time, cloud_cover,
                processing_status, total_patches_extracted, landslide_patches_detected,
                landslide_events_clustered, processed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                aoi_id, acq_id, acq_time, cloud_cover,
                status, total_patches, landslide_patches,
                landslide_events, now_iso
            ),
        )
        conn.commit()
        return cursor.lastrowid


def list_processed_acquisitions(
    aoi_id: str,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Lists all processed acquisitions for an AOI ordered newest first."""
    path = db_path or DEFAULT_DB_PATH
    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM processed_acquisitions 
            WHERE aoi_id = ? 
            ORDER BY id DESC
            """,
            (aoi_id,),
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def update_aoi_acquisition_status(
    aoi_id: str,
    acquisition_id: str,
    acquisition_time: Optional[str],
    quality_status: str,
    db_path: Optional[str] = None,
) -> bool:
    """Updates the last_acquisition_id, time, and data-quality status for an AOI."""
    path = db_path or DEFAULT_DB_PATH
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE monitored_aois 
            SET last_acquisition_id = ?,
                last_acquisition_time = ?,
                last_quality_status = ?,
                updated_at = ?
            WHERE aoi_id = ?
            """,
            (acquisition_id, acquisition_time, quality_status, now_iso, aoi_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def log_polling_event(
    status: str,
    latest_catalog_id: Optional[str] = None,
    message: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """Logs a scheduler check event into polling_logs."""
    path = db_path or DEFAULT_DB_PATH
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO polling_logs (checked_at, status, latest_catalog_acquisition_id, message)
            VALUES (?, ?, ?, ?)
            """,
            (now_iso, status, latest_catalog_id, message),
        )
        conn.commit()
        return cursor.lastrowid


def get_latest_polling_log(db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves the most recent polling log record."""
    path = db_path or DEFAULT_DB_PATH
    with get_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM polling_logs 
            ORDER BY id DESC LIMIT 1
            """
        )
        row = cursor.fetchone()
        return dict(row) if row else None
