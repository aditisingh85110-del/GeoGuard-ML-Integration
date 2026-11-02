"""
Script to execute PART E: Live Inference Compatibility Test for CNN v2 on real CDSE Sentinel-2 L2A raster.
"""

import os
import sys
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

from ml_service.src.radiometric_calibration import (
    OPTICAL_4CH_NAMES,
    validate_satellite_radiometry,
)

def run():
    print("=== PART E: LIVE INFERENCE COMPATIBILITY TEST FOR CNN V2 ===")

    # 1. Load Live CDSE Raster
    raw_cdse_path = "ml_service/reports/diagnostics_raw_cdse_raster.npy"
    raw_cdse = np.load(raw_cdse_path)  # Shape (128, 128, 4)
    print(f"Loaded live CDSE raster: shape={raw_cdse.shape}, dtype={raw_cdse.dtype}")

    # 2. Load Preprocessing Config v2
    with open("ml_service/models/preprocessing_config_v2.json", "r") as f:
        prep_cfg = json.load(f)

    mean_v2 = np.array(prep_cfg["mean"], dtype=np.float32)
    std_v2 = np.array(prep_cfg["std"], dtype=np.float32)

    # 3. Validate Radiometry Before Inference
    is_valid, status, diag = validate_satellite_radiometry(raw_cdse)
    print(f"Input Radiometric Validation Status: {status} (is_valid={is_valid})")
    print(f"Diagnostics: {json.dumps(diag, indent=2)}")

    if not is_valid:
        print(f"Input rejected by safety guardrails with status: {status}")
        return

    # 4. Compute Per-Band Normalized Statistics and Clipping
    live_norm = (raw_cdse - mean_v2.reshape(1, 1, 4)) / std_v2.reshape(1, 1, 4)
    unclipped_norm = live_norm.copy()
    clipped_norm = np.clip(live_norm, -5.0, 5.0)

    stats_comparison = {}
    print("\n=== PER-BAND DISTRIBUTION COMPARISON: TRAINING BOA VS LIVE CDSE ===")
    for i, b in enumerate(OPTICAL_4CH_NAMES):
        raw_b = raw_cdse[:, :, i]
        unclip_b = unclipped_norm[:, :, i]
        clip_b = clipped_norm[:, :, i]

        clip_neg_pct = float(np.mean(unclip_b <= -5.0) * 100)
        clip_pos_pct = float(np.mean(unclip_b >= 5.0) * 100)
        z_mean = float(unclip_b.mean())

        stats_comparison[b] = {
            "training_boa_mean": float(mean_v2[i]),
            "training_boa_std": float(std_v2[i]),
            "live_cdse_raw_mean": float(raw_b.mean()),
            "live_cdse_raw_std": float(raw_b.std()),
            "live_cdse_raw_min": float(raw_b.min()),
            "live_cdse_raw_max": float(raw_b.max()),
            "normalized_z_score_mean": z_mean,
            "normalized_min": float(clip_b.min()),
            "normalized_max": float(clip_b.max()),
            "clipped_at_minus_5_percent": clip_neg_pct,
            "clipped_at_plus_5_percent": clip_pos_pct,
        }

        print(f"Band {b}:")
        print(f"  Train BOA: mean={mean_v2[i]:.4f}, std={std_v2[i]:.4f}")
        print(f"  Live CDSE: mean={raw_b.mean():.4f}, std={raw_b.std():.4f}, range=[{raw_b.min():.4f}, {raw_b.max():.4f}]")
        print(f"  Z-score mean: {z_mean:.3f} std deviations from train mean")
        print(f"  Clipping: -5.0 clamp = {clip_neg_pct:.2f}%, +5.0 clamp = {clip_pos_pct:.2f}%")

    # 5. Load CNN v2 Model and Run Inference
    model_v2_path = "ml_service/models/cnn_model_v2.keras"
    print(f"\nLoading CNN v2 model from {model_v2_path}...")
    model_v2 = load_model(model_v2_path)

    input_tensor = np.expand_dims(clipped_norm, axis=0)  # (1, 128, 128, 4)

    # Inspect intermediate activations to verify active neurons
    layer_outputs = [layer.output for layer in model_v2.layers]
    activation_model = Model(inputs=model_v2.input, outputs=layer_outputs)
    activations = activation_model.predict(input_tensor, verbose=0)

    print("\n=== CNN V2 LAYER ACTIVATION AUDIT (LIVE CDSE INPUT) ===")
    activation_audit = []
    for layer, act in zip(model_v2.layers, activations):
        zero_pct = float(np.mean(act == 0) * 100) if act.ndim > 1 else 0.0
        info = {
            "layer_name": layer.name,
            "layer_type": layer.__class__.__name__,
            "shape": list(act.shape),
            "min": float(act.min()),
            "max": float(act.max()),
            "mean": float(act.mean()),
            "zero_percentage": zero_pct,
        }
        activation_audit.append(info)
        print(f"Layer {layer.name:<25} ({layer.__class__.__name__:<16}): shape={str(act.shape):<18} mean={act.mean():+.4f} zeros={zero_pct:5.1f}%")

    # Prediction probability
    prob = float(activations[-1][0, 0])
    with open("ml_service/models/model_metadata_v2.json") as f:
        meta_v2 = json.load(f)
    decision_threshold = meta_v2["selected_decision_threshold"]
    pred_class = "LANDSLIDE DETECTED" if prob >= decision_threshold else "NORMAL / STABLE"

    print(f"\n=== CNN V2 LIVE INFERENCE OUTCOME ===")
    print(f"Predicted Probability P(Landslide): {prob:.4f}")
    print(f"Decision Threshold:                 {decision_threshold:.4f}")
    print(f"Classification Result:              {pred_class}")

    # 6. Test Guardrails on Corrupted / Cloud / Invalid Input
    print("\n=== TESTING INSUFFICIENT_DATA GUARDRAIL ON CORRUPTED INPUTS ===")
    # Test 6a: 100% NaN/nodata patch
    corrupted_patch = np.zeros((128, 128, 4), dtype=np.float32)
    is_valid_c, status_c, diag_c = validate_satellite_radiometry(corrupted_patch)
    print(f"Corrupted zero-fill patch validation: status={status_c}, is_valid={is_valid_c}")

    # Test 6b: Heavy cloud patch (reflectance > 0.85 across optical bands)
    cloud_patch = np.full((128, 128, 4), 0.75, dtype=np.float32)
    is_valid_cloud, status_cloud, diag_cloud = validate_satellite_radiometry(cloud_patch)
    print(f"Cloud-saturated patch validation: status={status_cloud}, is_valid={is_valid_cloud}")

    # Test 6c: Uncalibrated raw DN input (mean ~ 1200.0)
    raw_dn_patch = np.full((128, 128, 4), 1200.0, dtype=np.float32)
    is_valid_dn, status_dn, diag_dn = validate_satellite_radiometry(raw_dn_patch)
    print(f"Uncalibrated raw DN patch validation: status={status_dn}, is_valid={is_valid_dn}")

    # Save complete live test results to JSON
    live_results = {
        "radiometric_validation": {
            "status": status,
            "is_valid": is_valid,
            "diagnostics": diag,
        },
        "per_band_comparison": stats_comparison,
        "prediction": {
            "probability": prob,
            "decision_threshold": decision_threshold,
            "prediction_class": pred_class,
            "model_version": "GeoGuard_CNN_v2",
        },
        "activation_audit": activation_audit,
        "guardrail_verification": {
            "zero_fill_test": {"status": status_c, "is_valid": is_valid_c},
            "cloud_saturation_test": {"status": status_cloud, "is_valid": is_valid_cloud},
            "raw_dn_out_of_bounds_test": {"status": status_dn, "is_valid": is_valid_dn},
        },
    }

    with open("ml_service/reports/part_e_live_inference_results.json", "w") as f:
        json.dump(live_results, f, indent=2)
    print("\nSaved ml_service/reports/part_e_live_inference_results.json successfully.")

if __name__ == "__main__":
    run()
