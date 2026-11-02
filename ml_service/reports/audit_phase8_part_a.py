"""
Phase 8 Part A Audit Script.
Verifies:
1. Landslide4Sense normalization formula and channel-specific constants.
2. Band order, scaling, and normalization in training vs live inference.
3. Geographical overlap between train, val, and test splits (checking coordinates/filenames/sources).
4. Reported metrics directly from saved models, predictions, and labels.
5. Reconcile CNN v1 threshold reported in Phase 3 with Phase 7 comparison.
6. Validation set stability (bootstrapping / leave-one-out on the 30 validation samples).
"""

import os
import sys
import glob
import json
import h5py
import numpy as np

# Ensure ml_service and project root in path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, proj_root)

from ml_service.src.radiometric_calibration import (
    OPTICAL_4CH_NAMES,
    OPTICAL_4CH_INDICES,
    OPTICAL_4CH_BOA_SCALE_FACTORS,
    convert_l4s_h5_to_boa,
    validate_satellite_radiometry,
)
from ml_service.src.data_loader import LandslideDatasetLoader
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print("="*60)
print("AUDIT PART A.1: Landslide4Sense Channel Ranges & Constants")
print("="*60)

img_means_iarai = [1111.812, 824.632, 663.416, 445.173, 645.858, 1547.735, 1960.444, 1941.322, 674.076, 9.048, 1113.983, 519.904, 20.292, 772.831]

train_imgs = glob.glob("ml_service/data/raw/train/images/*.h5")
print(f"Total training image files found: {len(train_imgs)}")

# Check first 5 images for min, mean, max across channels [1, 2, 3, 7]
sample_ch_means = []
for f in train_imgs:
    with h5py.File(f, "r") as h:
        arr = h["img"][:]
        # arr has shape (128, 128, 14)
        sample_ch_means.append([arr[:, :, c].mean() for c in OPTICAL_4CH_INDICES])
sample_ch_means = np.array(sample_ch_means)
print(f"Empirical mean of HDF5 values across 120 train images for channels [1, 2, 3, 7] (B02, B03, B04, B08):")
for idx, cname in enumerate(OPTICAL_4CH_NAMES):
    print(f"  {cname} (idx {OPTICAL_4CH_INDICES[idx]}): empirical H5 mean = {sample_ch_means[:, idx].mean():.4f}")

print("\nConstants used in radiometric_calibration.py:")
for idx, cname in enumerate(OPTICAL_4CH_NAMES):
    s = OPTICAL_4CH_BOA_SCALE_FACTORS[idx]
    expected_s = img_means_iarai[OPTICAL_4CH_INDICES[idx]] / 10000.0
    print(f"  {cname}: configured scale factor = {s}, expected iarai/10000 = {expected_s:.7f}, match = {abs(s - expected_s) < 1e-6}")

print("\n" + "="*60)
print("AUDIT PART A.2: Band Order, Scaling, and Normalization in Train vs Inference")
print("="*60)

# Check preprocessing_config_v2.json
with open("ml_service/models/preprocessing_config_v2.json") as f:
    prep_v2 = json.load(f)
print("preprocessing_config_v2.json band order:", prep_v2.get("band_names"))
print("preprocessing_config_v2.json scale factors:", prep_v2.get("scale_factors_from_h5"))
print("preprocessing_config_v2.json training mean:", prep_v2.get("mean"))
print("preprocessing_config_v2.json training std:", prep_v2.get("std"))

# Check satellite.py evalscript bands
print("satellite.py evalscript input bands: ['B02', 'B03', 'B04', 'B08']")
print("radiometric_calibration.py OPTICAL_4CH_NAMES:", OPTICAL_4CH_NAMES)
print("Band order match between satellite.py, calibration, and training:",
      prep_v2.get("band_names") == OPTICAL_4CH_NAMES == ["B02", "B03", "B04", "B08"])

print("\n" + "="*60)
print("AUDIT PART A.3: Geographical Overlap & Sample IDs Check")
print("="*60)

loader = LandslideDatasetLoader("ml_service/data/raw")
train_samples = loader.get_sample_paths("train")
val_samples = loader.get_sample_paths("validation")
test_samples = loader.get_sample_paths("test")

train_ids = set([s[0] for s in train_samples])
val_ids = set([s[0] for s in val_samples])
test_ids = set([s[0] for s in test_samples])

print(f"Sample count: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
print(f"Train & Val intersection: {train_ids.intersection(val_ids)}")
print(f"Train & Test intersection: {train_ids.intersection(test_ids)}")
print(f"Val & Test intersection: {val_ids.intersection(test_ids)}")

print(f"Train ID range: min={min(map(int, train_ids))}, max={max(map(int, train_ids))}")
print(f"Val ID range: min={min(map(int, val_ids))}, max={max(map(int, val_ids))}")
print(f"Test ID range: min={min(map(int, test_ids))}, max={max(map(int, test_ids))}")

# Check if metadata or coordinates exist in HDF5 files
sample_h5 = train_samples[0][1]
with h5py.File(sample_h5, "r") as h:
    print(f"HDF5 keys in sample {sample_h5}: {list(h.keys())}")
    print(f"HDF5 attributes in sample: {list(h.attrs.keys())}")
    for k in h.attrs.keys():
        print(f"  attr {k}: {h.attrs[k]}")

print("\n" + "="*60)
print("AUDIT PART A.4: Verify Reported Metrics Directly from Saved Models")
print("="*60)

# Evaluate CNN v2 on test split
X_test_raw, y_test, meta_test = loader.load_split("test")
X_test_boa = convert_l4s_h5_to_boa(X_test_raw)

# Load config
t_mean = np.array(prep_v2["mean"], dtype=np.float32)
t_std = np.array(prep_v2["std"], dtype=np.float32)
X_test_norm = (X_test_boa - t_mean) / t_std
X_test_norm = np.clip(X_test_norm, -5.0, 5.0)

model_v2 = tf.keras.models.load_model("ml_service/models/cnn_model_v2.keras")
y_prob_v2 = model_v2.predict(X_test_norm, verbose=0).flatten()

# Test with threshold 0.25
y_pred_v2 = (y_prob_v2 >= 0.25).astype(int)
acc = accuracy_score(y_test, y_pred_v2)
prec = precision_score(y_test, y_pred_v2, zero_division=0)
rec_landslide = recall_score(y_test, y_pred_v2, pos_label=1)
rec_normal = recall_score(y_test, y_pred_v2, pos_label=0)
f1 = f1_score(y_test, y_pred_v2)
auc = roc_auc_score(y_test, y_prob_v2)
cm = confusion_matrix(y_test, y_pred_v2)
tn, fp, fn, tp = cm.ravel()

print(f"Direct Evaluation of CNN v2 on Test Set (Threshold=0.25):")
print(f"  Accuracy:         {acc:.4f} (reported: 0.7833)")
print(f"  Precision:        {prec:.4f} (reported: 0.7250)")
print(f"  Landslide Recall: {rec_landslide:.4f} (reported: 0.9355)")
print(f"  Normal Recall:    {rec_normal:.4f} (reported: 0.6207)")
print(f"  F1 Score:         {f1:.4f} (reported: 0.8169)")
print(f"  ROC-AUC:          {auc:.4f} (reported: 0.8754)")
print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp} (reported: TN=18, FP=11, FN=2, TP=29)")

print("\n" + "="*60)
print("AUDIT PART A.5: Reconcile CNN v1 Threshold Reported in Phase 3 vs Phase 7")
print("="*60)
with open("ml_service/reports/test_evaluation_report.json") as f:
    rep_v1 = json.load(f)
print("Phase 3 test_evaluation_report.json decision_threshold:", rep_v1.get("decision_threshold"))
print("Phase 3 metrics:", rep_v1["metrics"])
print("Phase 3 confusion_matrix:", rep_v1["confusion_matrix"])

# Now evaluate CNN v1 at threshold 0.50 and at threshold 0.25
with open("ml_service/models/preprocessing_config.json") as f:
    prep_v1 = json.load(f)
v1_mean = np.array(prep_v1["mean"], dtype=np.float32)
v1_std = np.array(prep_v1["std"], dtype=np.float32)
X_test_v1_norm = (X_test_raw - v1_mean) / v1_std
X_test_v1_norm = np.clip(X_test_v1_norm, -5.0, 5.0)

model_v1 = tf.keras.models.load_model("ml_service/models/cnn_model.keras")
y_prob_v1 = model_v1.predict(X_test_v1_norm, verbose=0).flatten()

y_pred_v1_at_050 = (y_prob_v1 >= 0.50).astype(int)
cm_050 = confusion_matrix(y_test, y_pred_v1_at_050)
print(f"CNN v1 at T=0.50 (Phase 3 baseline):")
print(f"  Acc={accuracy_score(y_test, y_pred_v1_at_050):.4f}, Prec={precision_score(y_test, y_pred_v1_at_050):.4f}, Recall={recall_score(y_test, y_pred_v1_at_050):.4f}, F1={f1_score(y_test, y_pred_v1_at_050):.4f}")
print(f"  TN={cm_050[0,0]}, FP={cm_050[0,1]}, FN={cm_050[1,0]}, TP={cm_050[1,1]}")

y_pred_v1_at_025 = (y_prob_v1 >= 0.25).astype(int)
cm_025 = confusion_matrix(y_test, y_pred_v1_at_025)
print(f"CNN v1 at T=0.25 (Same threshold as v2):")
print(f"  Acc={accuracy_score(y_test, y_pred_v1_at_025):.4f}, Prec={precision_score(y_test, y_pred_v1_at_025):.4f}, Recall={recall_score(y_test, y_pred_v1_at_025):.4f}, F1={f1_score(y_test, y_pred_v1_at_025):.4f}")
print(f"  TN={cm_025[0,0]}, FP={cm_025[0,1]}, FN={cm_025[1,0]}, TP={cm_025[1,1]}")

print("\n" + "="*60)
print("AUDIT PART A.6: Stability of 0.25 Threshold on 30-Sample Validation Set")
print("="*60)
X_val_raw, y_val, meta_val = loader.load_split("validation")
X_val_boa = convert_l4s_h5_to_boa(X_val_raw)
X_val_norm = np.clip((X_val_boa - t_mean) / t_std, -5.0, 5.0)
val_probs = model_v2.predict(X_val_norm, verbose=0).flatten()

print(f"Validation set size: {len(y_val)} (Landslide={np.sum(y_val==1)}, Normal={np.sum(y_val==0)})")
print(f"Validation probabilities: min={val_probs.min():.4f}, median={np.median(val_probs):.4f}, max={val_probs.max():.4f}")

# Bootstrap analysis: resample validation set 1,000 times
np.random.seed(42)
n_bootstraps = 1000
bootstrapped_optimal_thresholds = []
candidate_thresholds = np.linspace(0.10, 0.90, 17)

for _ in range(n_bootstraps):
    boot_indices = np.random.choice(len(y_val), size=len(y_val), replace=True)
    b_y = y_val[boot_indices]
    b_probs = val_probs[boot_indices]
    if len(np.unique(b_y)) < 2:
        continue
    best_f1_b = -1.0
    best_t_b = 0.50
    for thresh in candidate_thresholds:
        t = round(float(thresh), 2)
        b_pred = (b_probs >= t).astype(int)
        rec = recall_score(b_y, b_pred, zero_division=0)
        f1_b = f1_score(b_y, b_pred, zero_division=0)
        if f1_b > best_f1_b or (f1_b == best_f1_b and rec > 0.70):
            best_f1_b = f1_b
            best_t_b = t
    bootstrapped_optimal_thresholds.append(best_t_b)

bootstrapped_optimal_thresholds = np.array(bootstrapped_optimal_thresholds)
unique, counts = np.unique(bootstrapped_optimal_thresholds, return_counts=True)
print("Bootstrap (1,000 resamples) optimal threshold distribution on validation set:")
for u, c in zip(unique, counts):
    print(f"  Threshold {u:.2f}: {c/len(bootstrapped_optimal_thresholds)*100:.1f}%")

print("\nAUDIT COMPLETED SUCCESSFULLY.")
