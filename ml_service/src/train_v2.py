"""
GeoGuard Landslide Detection System - CNN v2 Training & Radiometric Pipeline.

Implements Phase 7 Radiometric Alignment and CNN v2 Development:
1. Ingests Landslide4Sense optical channels [B02, B03, B04, B08].
2. Calibrates raw HDF5 values to true physical BOA Surface Reflectance [0.0, 1.0].
3. Strictly fits normalization statistics on the training set (Zero Data Leakage).
4. Applies data augmentations (flips, rotations) strictly to training samples.
5. Handles class imbalance via balanced loss weighting.
6. Trains 4-channel CNN v2 with EarlyStopping and Checkpointing.
7. Tunes decision threshold strictly on the validation set.
8. Evaluates once on the held-out test split.
9. Exports all v2 model artifacts and diagnostic evaluation plots.
"""

import os
import sys
import json
import logging
import datetime
from typing import Dict, Any, Tuple, List, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
for p in [current_dir, parent_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

from ml_service.src.data_loader import LandslideDatasetLoader, DEFAULT_PIXEL_THRESHOLD
from ml_service.src.radiometric_calibration import (
    OPTICAL_4CH_NAMES,
    OPTICAL_4CH_INDICES,
    OPTICAL_4CH_BOA_SCALE_FACTORS,
    convert_l4s_h5_to_boa,
)
from ml_service.src.model import build_landslide_cnn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Sets reproducible random seeds across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Random seed set to {seed}")


def augment_training_data(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies spatial rotation and reflection augmentations strictly to training samples.
    Expands dataset by 3x (original, 90-deg rotation, horizontal flip).
    """
    np.random.seed(seed)
    augmented_X = [X]
    augmented_y = [y]

    # 1. 90-degree rotation
    X_rot90 = np.rot90(X, k=1, axes=(1, 2))
    augmented_X.append(X_rot90)
    augmented_y.append(y)

    # 2. Horizontal flip
    X_flip_h = np.flip(X, axis=2)
    augmented_X.append(X_flip_h)
    augmented_y.append(y)

    X_out = np.concatenate(augmented_X, axis=0)
    y_out = np.concatenate(augmented_y, axis=0)

    # Shuffle
    indices = np.arange(len(X_out))
    np.random.shuffle(indices)
    logger.info(f"Augmented training data from {len(X)} to {len(X_out)} samples.")
    return X_out[indices], y_out[indices]


def run_training_v2(
    data_dir: str = "ml_service/data/raw",
    model_save_path: str = "ml_service/models/cnn_model_v2.keras",
    config_save_path: str = "ml_service/models/preprocessing_config_v2.json",
    metadata_save_path: str = "ml_service/models/model_metadata_v2.json",
    reports_dir: str = "ml_service/reports",
    epochs: int = 40,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    pixel_threshold: int = DEFAULT_PIXEL_THRESHOLD,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes the complete training, threshold tuning, and held-out test evaluation for CNN v2.
    """
    set_seed(seed)
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load Data Splits
    loader = LandslideDatasetLoader(
        data_dir=data_dir,
        selected_bands=OPTICAL_4CH_INDICES,
        pixel_threshold=pixel_threshold,
    )

    logger.info("Loading Landslide4Sense train, val, and test splits...")
    X_train_raw, y_train, _ = loader.load_split("train")
    X_val_raw, y_val, _ = loader.load_split("validation")
    X_test_raw, y_test, _ = loader.load_split("test")

    logger.info(f"Loaded splits: Train={len(X_train_raw)}, Val={len(X_val_raw)}, Test={len(X_test_raw)}")

    # 2. Calibrate HDF5 arrays to true physical BOA Surface Reflectance [0.0, 1.0]
    logger.info("Applying IARAI benchmark scale factors to convert to Physical BOA Reflectance...")
    X_train_boa = convert_l4s_h5_to_boa(X_train_raw)
    X_val_boa = convert_l4s_h5_to_boa(X_val_raw)
    X_test_boa = convert_l4s_h5_to_boa(X_test_raw)

    # 3. Fit Normalization Statistics ONLY on X_train_boa (Zero Data Leakage)
    train_mean = np.mean(X_train_boa, axis=(0, 1, 2)).astype(np.float32)
    train_std = np.std(X_train_boa, axis=(0, 1, 2)).astype(np.float32)
    train_std = np.where(train_std < 1e-7, 1.0, train_std)

    logger.info("Fitted BOA Normalization Statistics (Train Only):")
    for bname, m, s in zip(OPTICAL_4CH_NAMES, train_mean, train_std):
        logger.info(f"  {bname}: Mean={m:.4f}, Std={s:.4f}")

    # Save Preprocessing Configuration v2
    prep_config_v2 = {
        "convention": "PHYSICAL_BOA_REFLECTANCE",
        "reflectance_range": [0.0, 1.0],
        "conversion_method": "IARAI_BENCHMARK_SCALE_FACTORS",
        "band_names": OPTICAL_4CH_NAMES,
        "input_channels": 4,
        "scale_factors_from_h5": [float(x) for x in OPTICAL_4CH_BOA_SCALE_FACTORS],
        "mean": [float(m) for m in train_mean],
        "std": [float(s) for s in train_std],
        "clip_range": [-5.0, 5.0],
    }
    with open(config_save_path, "w") as f:
        json.dump(prep_config_v2, f, indent=2)
    logger.info(f"Saved preprocessing config to {config_save_path}")

    # 4. Apply Normalization to all splits
    def normalize_boa(X: np.ndarray) -> np.ndarray:
        norm = (X - train_mean.reshape(1, 1, 1, 4)) / train_std.reshape(1, 1, 1, 4)
        return np.clip(norm, -5.0, 5.0).astype(np.float32)

    X_train_norm = normalize_boa(X_train_boa)
    X_val_norm = normalize_boa(X_val_boa)
    X_test_norm = normalize_boa(X_test_boa)

    # 5. Data Augmentation (Train only)
    X_train_aug, y_train_aug = augment_training_data(X_train_norm, y_train, seed=seed)

    # 6. Class Weighting
    classes = np.unique(y_train_aug)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_aug)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info(f"Class weights: {class_weight_dict}")

    # 7. Build Model
    model = build_landslide_cnn(
        input_shape=(128, 128, 4),
        learning_rate=learning_rate,
        dropout_rate=0.25,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
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

    # 8. Train Model
    logger.info(f"Training CNN v2 for up to {epochs} epochs...")
    history = model.fit(
        X_train_aug,
        y_train_aug,
        validation_data=(X_val_norm, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # 9. Plot and Save Training Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"], label="Train Loss", color="royalblue", lw=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss", color="crimson", lw=2)
    axes[0].set_title("CNN v2: Binary Cross-Entropy Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Train Accuracy", color="royalblue", lw=2)
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy", color="crimson", lw=2)
    axes[1].set_title("CNN v2: Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "training_curves_v2.png"), dpi=160)
    plt.close()

    # 10. Threshold Tuning (Validation Set ONLY)
    logger.info("Tuning decision threshold strictly on validation set...")
    val_probs = model.predict(X_val_norm, verbose=0).flatten()

    candidate_thresholds = np.linspace(0.10, 0.90, 17)
    threshold_results = []

    best_thresh = 0.50
    best_val_f1 = -1.0

    for thresh in candidate_thresholds:
        t = round(float(thresh), 2)
        v_pred = (val_probs >= t).astype(int)
        v_acc = float(accuracy_score(y_val, v_pred))
        v_prec = float(precision_score(y_val, v_pred, zero_division=0))
        v_rec = float(recall_score(y_val, v_pred, zero_division=0))
        v_f1 = float(f1_score(y_val, v_pred, zero_division=0))

        threshold_results.append({
            "threshold": t,
            "accuracy": round(v_acc, 4),
            "precision": round(v_prec, 4),
            "recall": round(v_rec, 4),
            "f1": round(v_f1, 4),
        })

        # Select threshold maximizing F1, prioritizing recall >= 0.70 to avoid false negatives
        if v_f1 > best_val_f1 or (v_f1 == best_val_f1 and v_rec > 0.70):
            best_val_f1 = v_f1
            best_thresh = t

    logger.info(f"Selected Optimal Decision Threshold: {best_thresh} (Val F1={best_val_f1:.4f})")

    # Plot Threshold Tuning Curve
    plt.figure(figsize=(8, 5))
    t_vals = [r["threshold"] for r in threshold_results]
    plt.plot(t_vals, [r["precision"] for r in threshold_results], label="Precision", color="forestgreen", lw=2, marker="o")
    plt.plot(t_vals, [r["recall"] for r in threshold_results], label="Recall (Landslide)", color="crimson", lw=2, marker="s")
    plt.plot(t_vals, [r["f1"] for r in threshold_results], label="F1 Score", color="darkorchid", lw=2, marker="^")
    plt.axvline(best_thresh, color="black", linestyle="--", label=f"Selected Threshold ({best_thresh})")
    plt.title("Validation Threshold Tuning: Precision, Recall, and F1", fontweight="bold")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Metric Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "threshold_tuning_validation_v2.png"), dpi=160)
    plt.close()

    # 11. Held-Out Test Evaluation (Evaluated ONCE on Untouched Test Set)
    logger.info(f"Evaluating CNN v2 on held-out test split ({len(X_test_norm)} samples) using threshold={best_thresh}...")
    test_probs = model.predict(X_test_norm, verbose=0).flatten()
    test_preds = (test_probs >= best_thresh).astype(int)

    test_acc = float(accuracy_score(y_test, test_preds))
    test_prec = float(precision_score(y_test, test_preds, zero_division=0))
    test_rec_landslide = float(recall_score(y_test, test_preds, pos_label=1, zero_division=0))
    test_rec_normal = float(recall_score(y_test, test_preds, pos_label=0, zero_division=0))
    test_f1 = float(f1_score(y_test, test_preds, zero_division=0))

    try:
        test_auc = float(roc_auc_score(y_test, test_probs))
    except Exception:
        test_auc = 0.50

    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()

    logger.info("=== HELD-OUT TEST RESULTS (CNN v2) ===")
    logger.info(f"Accuracy:         {test_acc:.4f}")
    logger.info(f"Precision:        {test_prec:.4f}")
    logger.info(f"Landslide Recall: {test_rec_landslide:.4f}")
    logger.info(f"Normal Recall:    {test_rec_normal:.4f}")
    logger.info(f"F1 Score:         {test_f1:.4f}")
    logger.info(f"ROC-AUC:          {test_auc:.4f}")
    logger.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"CNN v2 Confusion Matrix (Test Set, T={best_thresh})", fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["NORMAL (0)", "LANDSLIDE (1)"])
    plt.yticks(tick_marks, ["NORMAL (0)", "LANDSLIDE (1)"])

    thresh_val = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh_val else "black",
                     fontsize=14, fontweight="bold")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "confusion_matrix_v2.png"), dpi=160)
    plt.close()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"CNN v2 ROC (AUC = {test_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("CNN v2 Receiver Operating Characteristic (ROC)", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "roc_curve_v2.png"), dpi=160)
    plt.close()

    # 12. Save Model Metadata v2
    best_epoch = int(np.argmin(history.history["val_loss"]))
    metadata_v2 = {
        "model_version": "GeoGuard_CNN_v2",
        "convention": "PHYSICAL_BOA_REFLECTANCE",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_bands": OPTICAL_4CH_NAMES,
        "input_shape": [128, 128, 4],
        "pixel_threshold": pixel_threshold,
        "total_parameters": int(model.count_params()),
        "train_samples_original": int(len(X_train_raw)),
        "train_samples_augmented": int(len(X_train_aug)),
        "val_samples": int(len(X_val_raw)),
        "test_samples": int(len(X_test_raw)),
        "class_distribution_train": {
            "landslide": int(np.sum(y_train == 1)),
            "normal": int(np.sum(y_train == 0)),
        },
        "class_distribution_val": {
            "landslide": int(np.sum(y_val == 1)),
            "normal": int(np.sum(y_val == 0)),
        },
        "class_distribution_test": {
            "landslide": int(np.sum(y_test == 1)),
            "normal": int(np.sum(y_test == 0)),
        },
        "class_weights": class_weight_dict,
        "best_epoch": best_epoch + 1,
        "best_val_loss": float(history.history["val_loss"][best_epoch]),
        "best_val_accuracy": float(history.history["val_accuracy"][best_epoch]),
        "selected_decision_threshold": best_thresh,
        "threshold_tuning_validation": threshold_results,
        "test_evaluation_metrics": {
            "accuracy": round(test_acc, 4),
            "precision": round(test_prec, 4),
            "landslide_recall": round(test_rec_landslide, 4),
            "normal_recall": round(test_rec_normal, 4),
            "f1_score": round(test_f1, 4),
            "roc_auc": round(test_auc, 4),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        },
    }

    with open(metadata_save_path, "w") as f:
        json.dump(metadata_v2, f, indent=2)
    logger.info(f"Saved model metadata to {metadata_save_path}")

    return metadata_v2


if __name__ == "__main__":
    run_training_v2()
