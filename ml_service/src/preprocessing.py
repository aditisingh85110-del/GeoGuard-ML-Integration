"""
GeoGuard Landslide Detection System
Preprocessing Module for Satellite Imagery.

Handles:
- Channel-wise Z-score normalization computed strictly from training split.
- Exporting and loading preprocessing parameters to ensure zero data leakage.
- Spatial rotational and reflective data augmentations for training.
- True-color RGB and False-color (NIR-R-G) visual rendering with percentile stretching.
"""

import os
import json
import logging
from typing import Tuple, Optional, Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)


class SatellitePreprocessor:
    """Handles scaling, normalization, and augmentation of satellite patches."""

    def __init__(
        self,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        clip_range: Optional[Tuple[float, float]] = (-5.0, 5.0),
        band_names: Optional[List[str]] = None,
    ):
        """
        Args:
            mean: List of channel mean values.
            std: List of channel standard deviation values.
            clip_range: (min_val, max_val) after normalization to suppress extreme outliers.
            band_names: List of human-readable band names.
        """
        self.mean = np.array(mean, dtype=np.float32) if mean is not None else None
        self.std = np.array(std, dtype=np.float32) if std is not None else None
        self.clip_range = clip_range
        self.band_names = band_names or ["B2", "B3", "B4", "B8"]

    def fit(self, X_train: np.ndarray) -> "SatellitePreprocessor":
        """
        Computes per-channel mean and standard deviation strictly from training data.
        X_train shape: (N, H, W, C)
        """
        if X_train.ndim != 4:
            raise ValueError(f"Expected 4D array (N, H, W, C), got shape {X_train.shape}")

        C = X_train.shape[-1]
        # Compute mean and std over samples, height, width for each channel
        mean = np.mean(X_train, axis=(0, 1, 2))
        std = np.std(X_train, axis=(0, 1, 2))

        # Avoid division by zero
        std = np.where(std < 1e-7, 1.0, std)

        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

        logger.info("Fitted SatellitePreprocessor:")
        for i in range(C):
            name = self.band_names[i] if i < len(self.band_names) else f"Ch_{i}"
            logger.info(f"  Channel {i} ({name}): Mean={self.mean[i]:.4f}, Std={self.std[i]:.4f}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies z-score standardization: (X - mean) / std.
        X shape: (N, H, W, C) or (H, W, C)
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Preprocessor has not been fitted or loaded with parameters.")

        is_single = X.ndim == 3
        if is_single:
            X = np.expand_dims(X, axis=0)

        # Broadcast subtract and divide along channel dimension
        normalized = (X - self.mean) / (self.std + 1e-8)

        if self.clip_range is not None:
            normalized = np.clip(normalized, self.clip_range[0], self.clip_range[1])

        return normalized[0] if is_single else normalized

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fits on training data and returns transformed array."""
        self.fit(X_train)
        return self.transform(X_train)

    def save_config(self, filepath: str) -> None:
        """Saves mean, std, and band metadata to a JSON file."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot save uninitialized preprocessor.")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        config = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "band_names": self.band_names,
            "clip_range": list(self.clip_range) if self.clip_range else None,
            "input_channels": len(self.mean),
        }
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved preprocessing config to {filepath}")

    @classmethod
    def load_config(cls, filepath: str) -> "SatellitePreprocessor":
        """Loads preprocessing parameters from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessing config not found: {filepath}")

        with open(filepath, "r") as f:
            config = json.load(f)

        return cls(
            mean=config["mean"],
            std=config["std"],
            clip_range=tuple(config["clip_range"]) if config.get("clip_range") else None,
            band_names=config.get("band_names"),
        )

    @staticmethod
    def augment_patch(patch: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Randomly applies D4 dihedral group transformations (flips and 90 deg rotations).
        Preserves geometric validity of overhead satellite imagery.
        """
        rng = np.random.default_rng(seed)
        # Random horizontal flip
        if rng.random() > 0.5:
            patch = np.fliplr(patch)
        # Random vertical flip
        if rng.random() > 0.5:
            patch = np.flipud(patch)
        # Random 90 degree rotation
        k = rng.integers(0, 4)
        if k > 0:
            patch = np.rot90(patch, k=k, axes=(0, 1))
        return patch

    @staticmethod
    def to_rgb_display(
        patch_4ch: np.ndarray,
        p_min: float = 2.0,
        p_max: float = 98.0,
    ) -> np.ndarray:
        """
        Converts a 4-channel [B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)] patch
        into a contrast-stretched 3-channel uint8 RGB image for visualization.
        """
        # Patch channels: idx 0 = B2 (Blue), idx 1 = B3 (Green), idx 2 = B4 (Red)
        r = patch_4ch[:, :, 2]
        g = patch_4ch[:, :, 1]
        b = patch_4ch[:, :, 0]
        rgb = np.stack([r, g, b], axis=-1)

        # Percentile stretch for satellite reflectance
        rgb_stretched = np.zeros_like(rgb, dtype=np.float32)
        for c in range(3):
            low = np.percentile(rgb[:, :, c], p_min)
            high = np.percentile(rgb[:, :, c], p_max)
            if high - low > 1e-5:
                stretched = (rgb[:, :, c] - low) / (high - low)
            else:
                stretched = rgb[:, :, c] - low
            rgb_stretched[:, :, c] = np.clip(stretched, 0.0, 1.0)

        return (rgb_stretched * 255).astype(np.uint8)

    @staticmethod
    def to_false_color_display(
        patch_4ch: np.ndarray,
        p_min: float = 2.0,
        p_max: float = 98.0,
    ) -> np.ndarray:
        """
        Converts 4-channel patch into standard Color-Infrared (CIR) composite:
        NIR (B8) -> Red channel
        Red (B4) -> Green channel
        Green (B3) -> Blue channel
        Landslide scars appear distinctive cyan/grey against bright red vegetation.
        """
        nir = patch_4ch[:, :, 3]  # B8
        red = patch_4ch[:, :, 2]  # B4
        green = patch_4ch[:, :, 1]  # B3
        cir = np.stack([nir, red, green], axis=-1)

        cir_stretched = np.zeros_like(cir, dtype=np.float32)
        for c in range(3):
            low = np.percentile(cir[:, :, c], p_min)
            high = np.percentile(cir[:, :, c], p_max)
            if high - low > 1e-5:
                stretched = (cir[:, :, c] - low) / (high - low)
            else:
                stretched = cir[:, :, c] - low
            cir_stretched[:, :, c] = np.clip(stretched, 0.0, 1.0)

        return (cir_stretched * 255).astype(np.uint8)
