"""
GeoGuard Landslide Detection System — Bi-Temporal Change Detection Module.
Experimental Prototype for Hackathon Demonstration.

Computes bi-temporal spectral differences between aligned Sentinel-2 L2A optical acquisitions:
- Per-band reflectance differences (B02, B03, B04, B08)
- Pre-event and post-event NDVI
- NDVI differential (ΔNDVI = NDVI_post - NDVI_pre)
- Mean visible brightness change (ΔVisible)
- Combined spectral change indicator mask (Vegetation loss + mineral soil exposure)
- Spatial change metrics and candidate classification
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class BiTemporalChangeDetector:
    """
    Analyzes paired pre-event and post-event Sentinel-2 Level-2A BOA reflectance rasters.
    """

    def __init__(
        self,
        ndvi_drop_threshold: float = -0.15,
        visible_gain_threshold: float = 0.03,
        min_candidate_pixels: int = 50,
    ):
        """
        Args:
            ndvi_drop_threshold: Minimum drop in NDVI required to flag vegetation loss.
            visible_gain_threshold: Minimum increase in visible reflectance indicating exposed soil/rock.
            min_candidate_pixels: Minimum contiguous/total anomalous pixels required for candidate status.
        """
        self.ndvi_drop_threshold = ndvi_drop_threshold
        self.visible_gain_threshold = visible_gain_threshold
        self.min_candidate_pixels = min_candidate_pixels

    @staticmethod
    def calculate_ndvi(raster: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Calculates Normalized Difference Vegetation Index (NDVI) from 4-band BOA raster.
        Channel order: [B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)].
        NDVI = (B08 - B04) / (B08 + B04 + eps)
        """
        b04 = raster[:, :, 2]
        b08 = raster[:, :, 3]
        denom = b08 + b04 + eps
        ndvi = (b08 - b04) / denom
        return np.clip(ndvi, -1.0, 1.0)

    def analyze_pair(
        self,
        pre_raster: np.ndarray,
        post_raster: np.ndarray,
        pre_date: str,
        post_date: str,
        aoi_bbox: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Performs bi-temporal change analysis between pre-event and post-event rasters.
        Both rasters must have identical shape (H, W, 4) in physical BOA reflectance [0.0, 1.0].
        """
        if pre_raster.shape != post_raster.shape:
            raise ValueError(
                f"Raster shape mismatch: pre={pre_raster.shape}, post={post_raster.shape}"
            )
        if pre_raster.shape[-1] != 4:
            raise ValueError(f"Expected 4 channels, got {pre_raster.shape[-1]}")

        H, W, _ = pre_raster.shape
        total_pixels = H * W

        # 1. Per-band reflectance difference (ΔB = Post - Pre)
        delta_bands = post_raster - pre_raster
        band_diff_means = {
            "B02_delta_mean": round(float(np.mean(delta_bands[:, :, 0])), 5),
            "B03_delta_mean": round(float(np.mean(delta_bands[:, :, 1])), 5),
            "B04_delta_mean": round(float(np.mean(delta_bands[:, :, 2])), 5),
            "B08_delta_mean": round(float(np.mean(delta_bands[:, :, 3])), 5),
        }

        # 2. NDVI calculation
        pre_ndvi = self.calculate_ndvi(pre_raster)
        post_ndvi = self.calculate_ndvi(post_raster)
        delta_ndvi = post_ndvi - pre_ndvi

        mean_pre_ndvi = float(np.mean(pre_ndvi))
        mean_post_ndvi = float(np.mean(post_ndvi))
        mean_delta_ndvi = float(np.mean(delta_ndvi))

        # 3. Visible brightness change: average of ΔB02, ΔB03, ΔB04
        delta_visible = np.mean(delta_bands[:, :, :3], axis=-1)
        mean_delta_visible = float(np.mean(delta_visible))

        # 4. Landslide spectral change indicator mask:
        # Significant drop in chlorophyll (ΔNDVI < threshold) AND increase in exposed soil/rock brightness (ΔVisible > threshold)
        vegetation_loss_mask = delta_ndvi <= self.ndvi_drop_threshold
        soil_exposure_mask = delta_visible >= self.visible_gain_threshold
        candidate_change_mask = vegetation_loss_mask & soil_exposure_mask

        candidate_pixel_count = int(np.sum(candidate_change_mask))
        candidate_pixel_ratio = round(float(candidate_pixel_count / total_pixels), 4)

        is_significant_change = candidate_pixel_count >= self.min_candidate_pixels

        # Determine preliminary change status
        if is_significant_change:
            change_status = "SIGNIFICANT_SPECTRAL_CHANGE_DETECTED"
            summary_desc = (
                f"Detected sharp vegetation loss (ΔNDVI <= {self.ndvi_drop_threshold}) and bare soil exposure "
                f"across {candidate_pixel_count} pixels ({candidate_pixel_ratio:.1%} of AOI)."
            )
        else:
            change_status = "NO_SIGNIFICANT_CHANGE"
            summary_desc = (
                f"No significant mass-wasting spectral signature observed between {pre_date} and {post_date}. "
                f"Candidate pixels ({candidate_pixel_count}) below threshold ({self.min_candidate_pixels})."
            )

        return {
            "status": change_status,
            "temporal_metadata": {
                "pre_event_acquisition_date": pre_date,
                "post_event_acquisition_date": post_date,
            },
            "ndvi_analysis": {
                "pre_event_mean_ndvi": round(mean_pre_ndvi, 4),
                "post_event_mean_ndvi": round(mean_post_ndvi, 4),
                "mean_delta_ndvi": round(mean_delta_ndvi, 4),
                "ndvi_drop_threshold": self.ndvi_drop_threshold,
            },
            "reflectance_difference": {
                "mean_delta_visible": round(mean_delta_visible, 4),
                "per_band_deltas": band_diff_means,
            },
            "spatial_change_metrics": {
                "total_aoi_pixels": total_pixels,
                "candidate_change_pixel_count": candidate_pixel_count,
                "candidate_change_pixel_ratio": candidate_pixel_ratio,
                "min_pixels_threshold": self.min_candidate_pixels,
                "is_significant_change": is_significant_change,
            },
            "summary_description": summary_desc,
            "caveats": [
                "Optical change detection reflects land surface alteration but cannot distinguish landslides from agricultural clearing, intentional logging, or seasonal phenology without terrain context.",
                "Cloud shadows or subtle atmospheric differences between passes can induce minor localized NDVI fluctuations."
            ],
            # Raw matrices for visualization
            "_matrices": {
                "pre_ndvi": pre_ndvi,
                "post_ndvi": post_ndvi,
                "delta_ndvi": delta_ndvi,
                "candidate_mask": candidate_change_mask,
            }
        }

    def generate_visualization(
        self,
        pre_raster: np.ndarray,
        post_raster: np.ndarray,
        analysis_result: Dict[str, Any],
        output_path: str,
        title_prefix: str = "GeoGuard Bi-Temporal Change Detection",
    ):
        """
        Generates and saves a 4-panel visual inspection figure:
        Panel 1: Pre-event True Color RGB
        Panel 2: Post-event True Color RGB
        Panel 3: ΔNDVI Differential Heatmap
        Panel 4: Candidate Change Mask
        """
        matrices = analysis_result.get("_matrices", {})
        delta_ndvi = matrices.get("delta_ndvi")
        candidate_mask = matrices.get("candidate_mask")

        if delta_ndvi is None or candidate_mask is None:
            pre_ndvi = self.calculate_ndvi(pre_raster)
            post_ndvi = self.calculate_ndvi(post_raster)
            delta_ndvi = post_ndvi - pre_ndvi
            candidate_mask = (delta_ndvi <= self.ndvi_drop_threshold) & (
                np.mean(post_raster[:, :, :3] - pre_raster[:, :, :3], axis=-1) >= self.visible_gain_threshold
            )

        # True color normalization: B04 (Red), B03 (Green), B02 (Blue)
        pre_rgb = np.clip(pre_raster[:, :, [2, 1, 0]] / 0.30, 0.0, 1.0)
        post_rgb = np.clip(post_raster[:, :, [2, 1, 0]] / 0.30, 0.0, 1.0)

        pre_date = analysis_result["temporal_metadata"]["pre_event_acquisition_date"]
        post_date = analysis_result["temporal_metadata"]["post_event_acquisition_date"]
        status = analysis_result["status"]

        fig, axs = plt.subplots(1, 4, figsize=(18, 4.5), dpi=150)

        # Panel 1: Pre-Event True Color
        axs[0].imshow(pre_rgb)
        axs[0].set_title(f"Pre-Event Baseline\n({pre_date[:10]})", fontsize=10)
        axs[0].axis("off")

        # Panel 2: Post-Event True Color
        axs[1].imshow(post_rgb)
        axs[1].set_title(f"Post-Event Acquisition\n({post_date[:10]})", fontsize=10)
        axs[1].axis("off")

        # Panel 3: ΔNDVI Heatmap
        im_ndvi = axs[2].imshow(delta_ndvi, cmap="RdYlGn", vmin=-0.4, vmax=0.4)
        axs[2].set_title(f"ΔNDVI Differential\n(Mean: {analysis_result['ndvi_analysis']['mean_delta_ndvi']:+.3f})", fontsize=10)
        axs[2].axis("off")
        fig.colorbar(im_ndvi, ax=axs[2], fraction=0.046, pad=0.04)

        # Panel 4: Change Candidate Mask
        im_mask = axs[3].imshow(candidate_mask, cmap="Reds", vmin=0, vmax=1)
        cnt = analysis_result["spatial_change_metrics"]["candidate_change_pixel_count"]
        pct = analysis_result["spatial_change_metrics"]["candidate_change_pixel_ratio"] * 100
        axs[3].set_title(f"Change Candidate Mask\n({cnt} px / {pct:.1f}%)", fontsize=10)
        axs[3].axis("off")

        plt.suptitle(f"{title_prefix} — [{status}]", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved bi-temporal change visualization to {output_path}")
