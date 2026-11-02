"""
GeoGuard Landslide Detection System
Inference Module for Single-Patch and Batch Prediction.

Loads the trained CNN model once and provides high-performance predictions.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Union, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
for p in [current_dir, parent_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import tensorflow as tf

try:
    from ml_service.src.preprocessing import SatellitePreprocessor
except ImportError:
    from src.preprocessing import SatellitePreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class LandslidePredictor:
    """Inference engine wrapping the trained CNN model and preprocessor."""

    def __init__(
        self,
        model_path: str = "ml_service/models/cnn_model.keras",
        config_path: str = "ml_service/models/preprocessing_config.json",
        decision_threshold: Optional[float] = None,
        model_version: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to saved .keras model file.
            config_path: Path to preprocessing configuration JSON.
            decision_threshold: Decision threshold for declaring LANDSLIDE DETECTED.
            model_version: Optional explicit version string.
        """
        self.model_path = model_path
        self.config_path = config_path

        if "v2" in model_path:
            self.model_version = model_version or "GeoGuard_CNN_v2"
            self.decision_threshold = decision_threshold if decision_threshold is not None else 0.25
        else:
            self.model_version = model_version or "GeoGuard_CNN_v1"
            self.decision_threshold = decision_threshold if decision_threshold is not None else 0.40

        self.model: Optional[tf.keras.Model] = None
        self.preprocessor: Optional[SatellitePreprocessor] = None

        self._load_resources()

    def _load_resources(self):
        """Loads model and preprocessing parameters once into memory."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Train the model first.")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Preprocessing config not found at {self.config_path}.")

        logger.info(f"Loading preprocessor from {self.config_path}...")
        self.preprocessor = SatellitePreprocessor.load_config(self.config_path)

        logger.info(f"Loading CNN model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        logger.info("CNN model and preprocessor loaded successfully.")

    def predict_patch(self, patch: np.ndarray) -> Dict[str, Any]:
        """
        Runs inference on a single 4-channel patch of shape (128, 128, 4).

        Returns:
            Dictionary with prediction results, class label, and raw probability.
        """
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Predictor resources are not initialized.")

        if patch.shape != (128, 128, 4):
            raise ValueError(f"Expected patch shape (128, 128, 4), got {patch.shape}")

        t0 = time.time()

        # Apply identical normalization
        norm_patch = self.preprocessor.transform(patch)
        input_tensor = np.expand_dims(norm_patch, axis=0)

        # Run model inference
        raw_prob = float(self.model.predict(input_tensor, verbose=0)[0][0])
        inference_ms = round((time.time() - t0) * 1000, 2)

        is_landslide = raw_prob >= self.decision_threshold
        pred_class = "LANDSLIDE DETECTED" if is_landslide else "NORMAL / STABLE"

        return {
            "prediction_class": pred_class,
            "probability": round(raw_prob, 4),
            "model_score": round(raw_prob, 4),
            "score_type": "RAW_UNRESOLVED_SIGMOID",
            "decision_threshold": self.decision_threshold,
            "inference_time_ms": inference_ms,
            "model_version": self.model_version,
        }
