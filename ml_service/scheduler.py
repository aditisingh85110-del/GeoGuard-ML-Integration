"""
GeoGuard Landslide Detection System - Autonomous Satellite Monitoring Scheduler.

Periodically polls the Copernicus Data Space Ecosystem (CDSE) STAC catalogue
for new Sentinel-2 Level-2A satellite acquisitions over the monitored highway corridor.

Features:
- Configurable polling interval (default 15 minutes).
- Deduplication: Never reprocesses previously ingested acquisitions.
- Data integrity: If no new acquisition exists or conditions are cloudy, retains
  the last valid prediction and its true historical acquisition timestamp.
- Cloud screening: Flags scenes exceeding maximum cloud tolerance (default 30%).
- Graceful error handling: Handles network outages and API downtime without crashing.
- Dual execution modes: Single-shot (--once) or continuous daemon loop.
"""

import os
import sys
import time
import signal
import logging
import argparse
import datetime
from typing import Dict, Any, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml_service.config import settings
from ml_service.src import database
from ml_service.src.predict import LandslidePredictor
from ml_service.src.satellite import SatellitePipeline, CopernicusClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Scheduler] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("geoguard_scheduler")


class SatelliteScheduler:
    """Autonomous monitoring daemon for Copernicus Sentinel-2 acquisitions."""

    def __init__(
        self,
        pipeline: Optional[SatellitePipeline] = None,
        polling_interval_minutes: int = settings.POLLING_INTERVAL_MINUTES,
        db_path: Optional[str] = None,
    ):
        database.init_db(db_path)
        self.db_path = db_path
        self.polling_interval_seconds = polling_interval_minutes * 60

        if pipeline is None:
            predictor = LandslidePredictor(
                model_path=settings.MODEL_PATH,
                config_path=settings.PREPROCESSING_CONFIG_PATH,
                decision_threshold=settings.DECISION_THRESHOLD,
            )
            copernicus = CopernicusClient()
            self.pipeline = SatellitePipeline(predictor=predictor, copernicus_client=copernicus)
        else:
            self.pipeline = pipeline

        self._running = False

    def poll_once(self) -> Dict[str, Any]:
        """
        Executes a single catalogue check and inference cycle.
        Returns a dictionary summary of the action taken.
        """
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.info(f"Initiating satellite catalogue polling cycle at {now_str}...")

        aoi_name, bbox, is_test_aoi = settings.get_active_aoi(allow_test_fallback=True)
        if bbox is None:
            msg = "Corridor bounding box is not configured. Specify HIGHWAY_AOI_BBOX in .env."
            logger.warning(msg)
            database.log_polling_event(
                status="UNCONFIGURED_AOI",
                message=msg,
                db_path=self.db_path,
            )
            return {"status": "skipped", "reason": "unconfigured_aoi", "message": msg}

        # 1. Query CDSE STAC catalogue for latest acquisitions
        try:
            acquisitions = self.pipeline.copernicus.search_acquisitions(bbox=bbox, limit=2)
        except Exception as e:
            msg = f"Network or API exception during CDSE STAC search: {e}"
            logger.error(msg)
            database.log_polling_event(
                status="ERROR",
                message=msg,
                db_path=self.db_path,
            )
            return {"status": "error", "reason": "catalogue_connection_error", "message": msg}

        if not acquisitions:
            msg = "No Sentinel-2 acquisitions returned by STAC catalogue for AOI."
            logger.info(msg)
            database.log_polling_event(
                status="NO_NEW_ACQUISITION_AVAILABLE",
                message=msg,
                db_path=self.db_path,
            )
            return {"status": "idle", "reason": "no_acquisitions_found", "message": msg}

        latest_acq = acquisitions[0]
        acq_id = latest_acq.get("acquisition_id")
        acq_time = latest_acq.get("acquisition_time")
        cloud_cover = latest_acq.get("cloud_cover_percent", 0.0)

        logger.info(
            f"Catalogue report: Latest scene is '{acq_id}' acquired at {acq_time} (Cloud cover: {cloud_cover}%)."
        )

        # 2. Check if already processed (deduplication)
        if database.is_acquisition_processed(acq_id, db_path=self.db_path):
            msg = (
                f"Acquisition '{acq_id}' has already been processed and saved. "
                "Retaining last valid prediction. No duplicate computation needed."
            )
            logger.info(msg)
            database.log_polling_event(
                status="NO_NEW_ACQUISITION_AVAILABLE",
                latest_catalog_id=acq_id,
                message=msg,
                db_path=self.db_path,
            )
            return {
                "status": "idle",
                "reason": "already_processed",
                "acquisition_id": acq_id,
                "message": msg,
            }

        # 3. Cloud cover filtering
        if cloud_cover is not None and cloud_cover > settings.MAX_CLOUD_COVER_PERCENT:
            msg = (
                f"New acquisition '{acq_id}' detected, but cloud cover ({cloud_cover}%) "
                f"exceeds maximum threshold of {settings.MAX_CLOUD_COVER_PERCENT}%. "
                "Optical landslide detection cannot penetrate dense cloud cover. "
                "Preserving last valid unclouded prediction."
            )
            logger.warning(msg)
            database.log_polling_event(
                status="EXCESSIVE_CLOUD",
                latest_catalog_id=acq_id,
                message=msg,
                db_path=self.db_path,
            )
            return {
                "status": "skipped",
                "reason": "excessive_cloud_cover",
                "acquisition_id": acq_id,
                "cloud_cover_percent": cloud_cover,
                "message": msg,
            }

        # 4. Process new acquisition through end-to-end pipeline
        logger.info(f"Processing newly discovered acquisition '{acq_id}'...")
        try:
            result = self.pipeline.process_aoi(
                bbox=bbox,
                aoi_name=aoi_name,
                acquisition_id=acq_id,
            )

            if result.get("status") == "success":
                pred_id = database.save_prediction(result, db_path=self.db_path)
                overall_class = result.get("prediction", {}).get("overall_class")
                max_prob = result.get("prediction", {}).get("max_probability")

                msg = (
                    f"Successfully processed new acquisition '{acq_id}' -> "
                    f"Prediction: {overall_class} (Probability: {max_prob:.4f}, Record ID: #{pred_id})."
                )
                logger.info(msg)
                database.log_polling_event(
                    status="NEW_ACQUISITION_PROCESSED",
                    latest_catalog_id=acq_id,
                    message=msg,
                    db_path=self.db_path,
                )
                return {
                    "status": "success",
                    "action": "processed_new_acquisition",
                    "prediction_id": pred_id,
                    "acquisition_id": acq_id,
                    "overall_class": overall_class,
                    "max_probability": max_prob,
                    "message": msg,
                }
            else:
                err_msg = f"Failed to process acquisition '{acq_id}': {result.get('message')}"
                logger.error(err_msg)
                database.log_polling_event(
                    status="PROCESSING_ERROR",
                    latest_catalog_id=acq_id,
                    message=err_msg,
                    db_path=self.db_path,
                )
                return {"status": "error", "reason": "pipeline_failure", "message": err_msg}

        except Exception as e:
            err_msg = f"Unexpected exception processing acquisition '{acq_id}': {e}"
            logger.error(err_msg)
            database.log_polling_event(
                status="ERROR",
                latest_catalog_id=acq_id,
                message=err_msg,
                db_path=self.db_path,
            )
            return {"status": "error", "reason": "unexpected_exception", "message": err_msg}

    def start(self):
        """Starts continuous polling loop."""
        self._running = True

        def handle_signal(sig, frame):
            logger.info("Received termination signal. Stopping scheduler gracefully...")
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        logger.info(
            f"GeoGuard Satellite Scheduler started. Polling interval: "
            f"{settings.POLLING_INTERVAL_MINUTES} minutes ({self.polling_interval_seconds}s)."
        )

        while self._running:
            try:
                self.poll_once()
            except Exception as e:
                logger.error(f"Unhandled error in scheduler loop: {e}")

            if not self._running:
                break

            logger.info(f"Sleeping for {settings.POLLING_INTERVAL_MINUTES} minutes until next catalogue check...")
            # Sleep in 1-second chunks to respond immediately to SIGINT
            for _ in range(self.polling_interval_seconds):
                if not self._running:
                    break
                time.sleep(1)

        logger.info("GeoGuard Satellite Scheduler terminated cleanly.")


def main():
    parser = argparse.ArgumentParser(description="GeoGuard Satellite Monitoring Scheduler")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Execute a single check cycle and exit immediately (useful for testing or cron jobs).",
    )
    args = parser.parse_args()

    scheduler = SatelliteScheduler()
    if args.once:
        res = scheduler.poll_once()
        print("\n--- Scheduler Single Run Summary ---")
        for k, v in res.items():
            print(f"  {k}: {v}")
    else:
        scheduler.start()


if __name__ == "__main__":
    main()
