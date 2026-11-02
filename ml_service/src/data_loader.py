"""
GeoGuard Landslide Detection System
Dataset Loader for Landslide4Sense multispectral satellite imagery patches.

Key features:
- Reads HDF5 (.h5) patches and segmentation masks.
- Extracts selected compatible Sentinel-2 bands (default: B2, B3, B4, B8).
- Converts pixel masks into patch-level binary classification labels.
- Evaluates class distributions across configurable landslide-pixel thresholds.
- Provides leak-free training, validation, and testing splits.
"""

import os
import glob
import logging
from typing import Tuple, List, Optional, Dict, Any

import h5py
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Complete band specification for Landslide4Sense
ALL_BANDS = [
    "B1 - Coastal aerosol",      # idx 0
    "B2 - Blue",                 # idx 1 (Sentinel-2 10m)
    "B3 - Green",                # idx 2 (Sentinel-2 10m)
    "B4 - Red",                  # idx 3 (Sentinel-2 10m)
    "B5 - Red Edge 1",           # idx 4
    "B6 - Red Edge 2",           # idx 5
    "B7 - Red Edge 3",           # idx 6
    "B8 - NIR",                  # idx 7 (Sentinel-2 10m)
    "B9 - Water vapour",         # idx 8
    "B10 - SWIR Cirrus",         # idx 9
    "B11 - SWIR 1",              # idx 10
    "B12 - SWIR 2",              # idx 11
    "B13 - ALOS PALSAR Slope",   # idx 12 (Radar terrain slope)
    "B14 - ALOS PALSAR DEM"      # idx 13 (Radar elevation)
]

# Core 4-band subset available natively at 10m resolution from Copernicus Sentinel-2
DEFAULT_SELECTED_BAND_INDICES = [1, 2, 3, 7]  # B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
DEFAULT_SELECTED_BAND_NAMES = ["B2", "B3", "B4", "B8"]

# Default landslide-pixel threshold for declaring a 128x128 patch positive
# 100 pixels represents ~0.61% of the patch (~1 hectare footprint at 10m GSD)
DEFAULT_PIXEL_THRESHOLD = 100


class LandslideDatasetLoader:
    """Loader and patch-label generator for Landslide4Sense dataset."""

    def __init__(
        self,
        data_dir: str,
        selected_bands: Optional[List[int]] = None,
        pixel_threshold: int = DEFAULT_PIXEL_THRESHOLD,
    ):
        """
        Args:
            data_dir: Path to raw dataset directory (e.g. ml_service/data/raw)
            selected_bands: List of channel indices to load (default: [1, 2, 3, 7])
            pixel_threshold: Minimum positive pixels in mask to label patch as LANDSLIDE (1).
        """
        self.data_dir = data_dir
        self.selected_bands = selected_bands if selected_bands is not None else DEFAULT_SELECTED_BAND_INDICES
        self.pixel_threshold = pixel_threshold
        logger.info(
            f"Initialized LandslideDatasetLoader (Bands: {self.selected_bands}, "
            f"Pixel Threshold: {self.pixel_threshold})"
        )

    def get_sample_paths(self, split: str = "train") -> List[Tuple[str, str, str]]:
        """
        Returns list of tuples: (sample_id, image_path, mask_path).
        """
        img_dir = os.path.join(self.data_dir, split, "images")
        mask_dir = os.path.join(self.data_dir, split, "annotations")

        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        img_files = glob.glob(os.path.join(img_dir, "*.h5"))
        # Sort numerically by sample number
        img_files.sort(
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[1])
        )

        samples = []
        for img_path in img_files:
            fname = os.path.basename(img_path)
            num = fname.split("_")[1].split(".")[0]
            mask_path = os.path.join(mask_dir, f"mask_{num}.h5")
            if os.path.exists(mask_path):
                samples.append((num, img_path, mask_path))
            else:
                logger.warning(f"Missing mask for image {fname}, skipping.")

        return samples

    def load_single_sample(
        self, img_path: str, mask_path: str
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Loads a single image patch and its mask.
        Returns:
            image_subset: ndarray of shape (128, 128, len(selected_bands))
            raw_mask: ndarray of shape (128, 128)
            binary_label: int (1 for landslide, 0 for stable)
        """
        with h5py.File(img_path, "r") as f_img:
            img = f_img["img"][:]

        with h5py.File(mask_path, "r") as f_mask:
            mask = f_mask["mask"][:]

        # Handle NaNs or infinite values if present
        if np.isnan(img).any() or np.isinf(img).any():
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        # Slice selected bands
        img_subset = img[:, :, self.selected_bands].astype(np.float32)

        # Determine binary label from positive pixel threshold
        pos_pixels = int(np.sum(mask == 1))
        label = 1 if pos_pixels >= self.pixel_threshold else 0

        return img_subset, mask, label

    def load_split(
        self, split: str = "train", max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Loads all samples for a given split.
        Returns:
            X: ndarray of shape (N, 128, 128, C)
            y: ndarray of shape (N,)
            metadata: pd.DataFrame with sample metrics
        """
        pairs = self.get_sample_paths(split)
        if max_samples is not None:
            pairs = pairs[:max_samples]

        X_list = []
        y_list = []
        meta_list = []

        logger.info(f"Loading {len(pairs)} samples for split '{split}'...")
        for sample_id, img_path, mask_path in pairs:
            img_subset, mask, label = self.load_single_sample(img_path, mask_path)
            pos_pixels = int(np.sum(mask == 1))

            X_list.append(img_subset)
            y_list.append(label)
            meta_list.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "pos_pixels": pos_pixels,
                    "pos_pct": float(pos_pixels / mask.size * 100),
                    "label": label,
                    "class_name": "LANDSLIDE" if label == 1 else "NORMAL",
                }
            )

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        metadata = pd.DataFrame(meta_list)

        logger.info(
            f"Loaded {len(X)} patches for '{split}'. "
            f"Class balance: Landslide={np.sum(y == 1)} ({np.mean(y == 1)*100:.1f}%), "
            f"Normal={np.sum(y == 0)} ({np.mean(y == 0)*100:.1f}%)"
        )
        return X, y, metadata

    @staticmethod
    def evaluate_threshold_distribution(
        masks: List[np.ndarray], thresholds: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Evaluates how different landslide pixel thresholds affect class balance.
        """
        if thresholds is None:
            thresholds = [1, 16, 50, 100, 164, 328, 500, 1000]

        total_samples = len(masks)
        pos_pixel_counts = [int(np.sum(m == 1)) for m in masks]

        results = []
        for t in thresholds:
            pos_count = sum(1 for c in pos_pixel_counts if c >= t)
            neg_count = total_samples - pos_count
            pct_patch = (t / 16384) * 100
            results.append(
                {
                    "threshold_pixels": t,
                    "patch_area_pct": round(pct_patch, 3),
                    "positive_samples": pos_count,
                    "negative_samples": neg_count,
                    "positive_ratio": round(pos_count / total_samples, 4),
                    "negative_ratio": round(neg_count / total_samples, 4),
                }
            )
        return pd.DataFrame(results)
