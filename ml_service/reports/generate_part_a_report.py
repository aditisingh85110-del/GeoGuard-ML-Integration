"""
Script to compute empirical statistics for Part A and generate comparison histograms.
"""

import os
import sys
import glob
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import h5py
import numpy as np
import matplotlib.pyplot as plt

from ml_service.src.radiometric_calibration import (
    OPTICAL_4CH_NAMES,
    OPTICAL_4CH_INDICES,
    OPTICAL_4CH_BOA_SCALE_FACTORS,
    convert_l4s_h5_to_boa,
)

def run():
    print("Collecting Landslide4Sense splits...")
    splits = ["train", "validation", "test"]
    data = {}
    stats = {"original_h5": {}, "physical_boa": {}}

    for split in splits:
        files = sorted(glob.glob(f"ml_service/data/raw/{split}/images/*.h5"))
        h5_list = []
        for fp in files:
            with h5py.File(fp, "r") as f:
                h5_list.append(f["img"][:, :, OPTICAL_4CH_INDICES])
        arr_h5 = np.stack(h5_list, axis=0)  # (N, 128, 128, 4)
        arr_boa = convert_l4s_h5_to_boa(arr_h5)
        data[split] = {"h5": arr_h5, "boa": arr_boa}
        print(f"Loaded {split}: {arr_h5.shape[0]} patches")

    # Compute comprehensive statistics for train split
    for i, b in enumerate(OPTICAL_4CH_NAMES):
        ch_h5 = data["train"]["h5"][:, :, :, i].flatten()
        ch_boa = data["train"]["boa"][:, :, :, i].flatten()

        p_h5 = np.percentile(ch_h5, [1, 5, 25, 50, 75, 95, 99])
        p_boa = np.percentile(ch_boa, [1, 5, 25, 50, 75, 95, 99])

        stats["original_h5"][b] = {
            "min": float(ch_h5.min()),
            "max": float(ch_h5.max()),
            "mean": float(ch_h5.mean()),
            "std": float(ch_h5.std()),
            "median": float(p_h5[3]),
            "p1": float(p_h5[0]),
            "p5": float(p_h5[1]),
            "p25": float(p_h5[2]),
            "p75": float(p_h5[4]),
            "p95": float(p_h5[5]),
            "p99": float(p_h5[6]),
        }
        stats["physical_boa"][b] = {
            "scale_factor_used": float(OPTICAL_4CH_BOA_SCALE_FACTORS[i]),
            "min": float(ch_boa.min()),
            "max": float(ch_boa.max()),
            "mean": float(ch_boa.mean()),
            "std": float(ch_boa.std()),
            "median": float(p_boa[3]),
            "p1": float(p_boa[0]),
            "p5": float(p_boa[1]),
            "p25": float(p_boa[2]),
            "p75": float(p_boa[4]),
            "p95": float(p_boa[5]),
            "p99": float(p_boa[6]),
        }

    with open("ml_service/reports/part_a_landslide4sense_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved ml_service/reports/part_a_landslide4sense_stats.json")

    # Load raw CDSE raster from Phase 6
    raw_cdse = np.load("ml_service/reports/diagnostics_raw_cdse_raster.npy")

    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    colors = ["royalblue", "forestgreen", "crimson", "darkorchid"]

    for i, b in enumerate(OPTICAL_4CH_NAMES):
        # Row 1: Original HDF5
        axes[0, i].hist(data["train"]["h5"][:, :, :, i].flatten(), bins=40, color=colors[i], alpha=0.75, density=True)
        axes[0, i].set_title(f"Original H5 {b} (Train)\nMean={stats['original_h5'][b]['mean']:.3f}, Std={stats['original_h5'][b]['std']:.3f}", fontsize=11)
        if i == 0:
            axes[0, i].set_ylabel("Original HDF5\nDensity", fontsize=11, fontweight="bold")
        axes[0, i].grid(True, alpha=0.3)

        # Row 2: Converted Physical BOA Reflectance
        axes[1, i].hist(data["train"]["boa"][:, :, :, i].flatten(), bins=40, color=colors[i], alpha=0.75, density=True)
        axes[1, i].set_title(f"Calibrated BOA {b} (Train)\nMean={stats['physical_boa'][b]['mean']:.3f}, Std={stats['physical_boa'][b]['std']:.3f}", fontsize=11)
        if i == 0:
            axes[1, i].set_ylabel("Physical BOA\nDensity", fontsize=11, fontweight="bold")
        axes[1, i].grid(True, alpha=0.3)

        # Row 3: Live CDSE Sentinel-2 BOA Reflectance
        cdse_b = raw_cdse[:, :, i].flatten()
        axes[2, i].hist(cdse_b, bins=40, color="darkorange", alpha=0.75, density=True)
        axes[2, i].set_title(f"Live CDSE L2A {b} (Observed)\nMean={cdse_b.mean():.3f}, Std={cdse_b.std():.3f}", fontsize=11)
        if i == 0:
            axes[2, i].set_ylabel("Live CDSE\nDensity", fontsize=11, fontweight="bold")
        axes[2, i].set_xlabel("Surface Reflectance [0, 1]", fontsize=10)
        axes[2, i].grid(True, alpha=0.3)

    plt.suptitle("Radiometric Harmonization: Original HDF5 vs Calibrated BOA vs Live CDSE Sentinel-2", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig("ml_service/reports/part_a_histograms_comparison.png", dpi=160)
    plt.close()
    print("Saved ml_service/reports/part_a_histograms_comparison.png")

if __name__ == "__main__":
    run()
