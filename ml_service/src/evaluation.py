"""
GeoGuard Landslide Detection System
Held-Out Test Set Evaluation Module.

Evaluates trained CNN model on unseen test samples:
- Computes Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.
- Focuses specifically on Landslide Recall (minimizing false negatives).
- Generates diagnostic report charts (Confusion Matrix, ROC Curve, Loss/Accuracy Curves).
- Saves evaluation results as JSON report.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Tuple, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
for p in [current_dir, parent_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless reporting
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

try:
    from ml_service.src.data_loader import LandslideDatasetLoader, DEFAULT_SELECTED_BAND_INDICES, DEFAULT_PIXEL_THRESHOLD
    from ml_service.src.preprocessing import SatellitePreprocessor
except ImportError:
    from src.data_loader import LandslideDatasetLoader, DEFAULT_SELECTED_BAND_INDICES, DEFAULT_PIXEL_THRESHOLD
    from src.preprocessing import SatellitePreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str = "ml_service/models/cnn_model.keras",
    config_path: str = "ml_service/models/preprocessing_config.json",
    metadata_path: str = "ml_service/models/model_metadata.json",
    data_dir: str = "ml_service/data/raw",
    reports_dir: str = "ml_service/reports",
    decision_threshold: float = 0.5,
    pixel_threshold: int = DEFAULT_PIXEL_THRESHOLD,
) -> Dict[str, Any]:
    """
    Evaluates the saved model on the held-out test split.
    """
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load Preprocessor and Model
    logger.info(f"Loading preprocessing config from {config_path}")
    preprocessor = SatellitePreprocessor.load_config(config_path)

    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    # 2. Load Held-Out Test Split
    loader = LandslideDatasetLoader(
        data_dir=data_dir,
        selected_bands=DEFAULT_SELECTED_BAND_INDICES,
        pixel_threshold=pixel_threshold,
    )

    logger.info("Loading held-out test split...")
    X_test, y_test, meta_test = loader.load_split("test")

    if len(X_test) == 0:
        raise RuntimeError("Test split contains 0 samples. Verify ml_service/data/raw/test directory.")

    # 3. Preprocess Test Data using strictly Training Statistics
    X_test_norm = preprocessor.transform(X_test)

    # 4. Predict
    logger.info(f"Running inference on {len(X_test)} test samples...")
    y_prob = model.predict(X_test_norm, verbose=0).flatten()
    y_pred = (y_prob >= decision_threshold).astype(int)

    # 5. Compute Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Per-class specifics
    normal_recall = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    landslide_recall = rec  # tp / (tp + fn)

    logger.info("=== HELD-OUT TEST EVALUATION RESULTS ===")
    logger.info(f"Test Samples: {len(y_test)} (Landslide={np.sum(y_test == 1)}, Normal={np.sum(y_test == 0)})")
    logger.info(f"Accuracy:        {acc:.4f} ({acc*100:.2f}%)")
    logger.info(f"Precision:       {prec:.4f}")
    logger.info(f"Recall (Landslide): {rec:.4f} (Crucial safety metric)")
    logger.info(f"Recall (Normal):    {normal_recall:.4f}")
    logger.info(f"F1-Score:        {f1:.4f}")
    logger.info(f"ROC-AUC:         {auc:.4f}")
    logger.info(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    clf_report = classification_report(
        y_test, y_pred, target_names=["NORMAL", "LANDSLIDE"], output_dict=True, zero_division=0
    )

    # 6. Plot and Save Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    cax = ax_cm.matshow(cm, cmap="Blues", alpha=0.8)
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
    fig_cm.colorbar(cax)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["NORMAL (0)", "LANDSLIDE (1)"])
    ax_cm.set_yticklabels(["NORMAL (0)", "LANDSLIDE (1)"])
    ax_cm.set_xlabel("Predicted Label", fontweight="bold")
    ax_cm.set_ylabel("True Ground Truth Label", fontweight="bold")
    ax_cm.set_title("Test Set Confusion Matrix", pad=20, fontweight="bold")
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    fig_cm.tight_layout()
    fig_cm.savefig(cm_path, dpi=150)
    plt.close(fig_cm)
    logger.info(f"Saved Confusion Matrix plot to {cm_path}")

    # 7. Plot and Save ROC Curve
    try:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
        ax_roc.plot(fpr, tpr, color="crimson", lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
        ax_roc.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate", fontweight="bold")
        ax_roc.set_ylabel("True Positive Rate (Recall)", fontweight="bold")
        ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True, alpha=0.3)
        roc_path = os.path.join(reports_dir, "roc_curve.png")
        fig_roc.tight_layout()
        fig_roc.savefig(roc_path, dpi=150)
        plt.close(fig_roc)
        logger.info(f"Saved ROC Curve plot to {roc_path}")
    except Exception as e:
        logger.warning(f"Could not plot ROC curve: {e}")

    # 8. Plot and Save Training Curves (from metadata)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                meta = json.load(f)
            hist = meta.get("history", {})
            if "loss" in hist and "val_loss" in hist:
                epochs_range = range(1, len(hist["loss"]) + 1)
                fig_train, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                # Loss
                ax1.plot(epochs_range, hist["loss"], "b-o", label="Training Loss")
                ax1.plot(epochs_range, hist["val_loss"], "r--s", label="Validation Loss")
                ax1.set_title("Training & Validation Loss", fontweight="bold")
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Loss (Binary Crossentropy)")
                ax1.grid(True, alpha=0.3)
                ax1.legend()

                # Accuracy
                if "accuracy" in hist and "val_accuracy" in hist:
                    ax2.plot(epochs_range, hist["accuracy"], "b-o", label="Training Accuracy")
                    ax2.plot(epochs_range, hist["val_accuracy"], "r--s", label="Validation Accuracy")
                    ax2.set_title("Training & Validation Accuracy", fontweight="bold")
                    ax2.set_xlabel("Epoch")
                    ax2.set_ylabel("Accuracy")
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()

                curves_path = os.path.join(reports_dir, "training_curves.png")
                fig_train.tight_layout()
                fig_train.savefig(curves_path, dpi=150)
                plt.close(fig_train)
                logger.info(f"Saved Training Curves plot to {curves_path}")
        except Exception as e:
            logger.warning(f"Could not plot training curves: {e}")

    # 9. Save Evaluation Report JSON
    report = {
        "evaluation_dataset": "Landslide4Sense Test Split (Held-Out)",
        "total_test_samples": int(len(y_test)),
        "class_counts": {
            "landslide": int(np.sum(y_test == 1)),
            "normal": int(np.sum(y_test == 0)),
        },
        "decision_threshold": decision_threshold,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall_landslide": rec,
            "recall_normal": normal_recall,
            "f1_score": f1,
            "roc_auc": auc,
        },
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "classification_report": clf_report,
    }

    report_json_path = os.path.join(reports_dir, "test_evaluation_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved evaluation report JSON to {report_json_path}")

    return report


if __name__ == "__main__":
    evaluate_model()
