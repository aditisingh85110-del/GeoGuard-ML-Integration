"""
GeoGuard Landslide Detection System
CNN Training Pipeline with Reproducibility, Checkpointing, and Early Stopping.
"""

import os
import sys
import json
import logging
import datetime
from typing import Optional, Dict, Any, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
for p in [current_dir, parent_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

try:
    from ml_service.src.data_loader import LandslideDatasetLoader, DEFAULT_SELECTED_BAND_INDICES, DEFAULT_PIXEL_THRESHOLD
    from ml_service.src.preprocessing import SatellitePreprocessor
    from ml_service.src.model import build_landslide_cnn
except ImportError:
    from src.data_loader import LandslideDatasetLoader, DEFAULT_SELECTED_BAND_INDICES, DEFAULT_PIXEL_THRESHOLD
    from src.preprocessing import SatellitePreprocessor
    from src.model import build_landslide_cnn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Sets random seeds across Python, NumPy, and TensorFlow for strict reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Reproducible random seed set to {seed}")


def run_training(
    data_dir: str = "ml_service/data/raw",
    model_save_path: str = "ml_service/models/cnn_model.keras",
    config_save_path: str = "ml_service/models/preprocessing_config.json",
    metadata_save_path: str = "ml_service/models/model_metadata.json",
    epochs: int = 35,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    pixel_threshold: int = DEFAULT_PIXEL_THRESHOLD,
    seed: int = 42,
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Executes the end-to-end training routine.
    """
    set_seed(seed)

    # 1. Load Train and Validation Splits
    loader = LandslideDatasetLoader(
        data_dir=data_dir,
        selected_bands=DEFAULT_SELECTED_BAND_INDICES,
        pixel_threshold=pixel_threshold,
    )

    logger.info("Loading training split...")
    X_train, y_train, meta_train = loader.load_split("train")

    logger.info("Loading validation split...")
    X_val, y_val, meta_val = loader.load_split("validation")

    # 2. Fit Preprocessor strictly on X_train (Zero Data Leakage)
    preprocessor = SatellitePreprocessor(band_names=["B2", "B3", "B4", "B8"])
    X_train_norm = preprocessor.fit_transform(X_train)
    X_val_norm = preprocessor.transform(X_val)

    # Save preprocessing parameters
    os.makedirs(os.path.dirname(os.path.abspath(config_save_path)), exist_ok=True)
    preprocessor.save_config(config_save_path)

    # 3. Compute Class Weights to balance loss gradients
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info(f"Computed Class Weights: {class_weight_dict}")

    # 4. Build Model
    model = build_landslide_cnn(
        input_shape=(128, 128, 4),
        learning_rate=learning_rate,
    )

    # 5. Define Callbacks
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    # 6. Fit Model
    logger.info(f"Starting model training for {epochs} max epochs (batch_size={batch_size})...")
    history = model.fit(
        X_train_norm,
        y_train,
        validation_data=(X_val_norm, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # 7. Collect and Save Metadata
    best_epoch = int(np.argmin(history.history["val_loss"]))
    metadata = {
        "model_version": "GeoGuard_CNN_v1",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_bands": ["B2", "B3", "B4", "B8"],
        "input_shape": [128, 128, 4],
        "pixel_threshold": pixel_threshold,
        "total_parameters": int(model.count_params()),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "class_distribution_train": {
            "landslide": int(np.sum(y_train == 1)),
            "normal": int(np.sum(y_train == 0)),
        },
        "class_distribution_val": {
            "landslide": int(np.sum(y_val == 1)),
            "normal": int(np.sum(y_val == 0)),
        },
        "class_weights": class_weight_dict,
        "best_epoch": best_epoch + 1,
        "best_val_loss": float(history.history["val_loss"][best_epoch]),
        "best_val_accuracy": float(history.history["val_accuracy"][best_epoch]),
        "history": {k: [float(val) for val in v] for k, v in history.history.items()},
    }

    with open(metadata_save_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to {metadata_save_path}")

    return model, metadata


if __name__ == "__main__":
    run_training()
