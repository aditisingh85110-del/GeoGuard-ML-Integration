"""
Notebook Generator for GeoGuard ML Service.
Builds all 5 notebooks:
- 01_dataset_exploration.ipynb
- 02_preprocessing.ipynb
- 03_cnn_training.ipynb
- 04_model_evaluation.ipynb
- 05_satellite_inference_test.ipynb
"""

import json
import os

def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated notebook: {filepath}")

def build_exploration_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# GeoGuard: Satellite Landslide Detection\n",
                "## Notebook 01: Dataset Exploration & Band Analysis\n",
                "\n",
                "This notebook explores the **Landslide4Sense** benchmark dataset to inspect:\n",
                "1. Data structure, array dimensions, and value ranges.\n",
                "2. Channel definitions across all 14 multi-sensor bands.\n",
                "3. Why optical **Sentinel-2 B2, B3, B4, and B8** (10m resolution) are selected for the CNN.\n",
                "4. Why non-satellite ALOS PALSAR terrain bands (Slope and DEM) are omitted from the initial model.\n",
                "5. Ground-truth landslide segmentation masks and how pixel thresholds affect patch-level binary class balance."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys\n",
                "import h5py\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import pandas as pd\n",
                "\n",
                "PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
                "if PROJECT_ROOT not in sys.path:\n",
                "    sys.path.insert(0, PROJECT_ROOT)\n",
                "\n",
                "from src.data_loader import LandslideDatasetLoader, ALL_BANDS, DEFAULT_SELECTED_BAND_INDICES\n",
                "from src.preprocessing import SatellitePreprocessor\n",
                "\n",
                "print(f\"Project root: {PROJECT_ROOT}\")\n",
                "print(f\"Number of total available bands: {len(ALL_BANDS)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Inspecting the 14 Bands in Landslide4Sense"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "band_table = []\n",
                "for idx, name in enumerate(ALL_BANDS):\n",
                "    selected = \"YES (CNN Input)\" if idx in DEFAULT_SELECTED_BAND_INDICES else \"No\"\n",
                "    band_table.append({\"Index\": idx, \"Band Name\": name, \"Selected for CNN\": selected})\n",
                "\n",
                "df_bands = pd.DataFrame(band_table)\n",
                "print(df_bands.to_string(index=False))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Visualizing Satellite Patches & Ground Truth Masks"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "raw_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')\n",
                "loader = LandslideDatasetLoader(data_dir=raw_dir, pixel_threshold=100)\n",
                "samples = loader.get_sample_paths('train')\n",
                "sample_id, img_p, mask_p = samples[0]\n",
                "with h5py.File(img_p, 'r') as f_img, h5py.File(mask_p, 'r') as f_mask:\n",
                "    img_all = f_img['img'][:]\n",
                "    mask = f_mask['mask'][:]\n",
                "\n",
                "patch_4ch = img_all[:, :, [1, 2, 3, 7]]\n",
                "rgb = SatellitePreprocessor.to_rgb_display(patch_4ch)\n",
                "cir = SatellitePreprocessor.to_false_color_display(patch_4ch)\n",
                "slope = img_all[:, :, 12]\n",
                "dem = img_all[:, :, 13]\n",
                "\n",
                "fig, axes = plt.subplots(1, 5, figsize=(20, 4))\n",
                "axes[0].imshow(rgb); axes[0].set_title(f\"True-Color RGB (#{sample_id})\"); axes[0].axis('off')\n",
                "axes[1].imshow(cir); axes[1].set_title(\"False-Color CIR (NIR-R-G)\"); axes[1].axis('off')\n",
                "pos_px = int(np.sum(mask == 1))\n",
                "axes[2].imshow(mask, cmap='gray'); axes[2].set_title(f\"Mask ({pos_px} px, {pos_px/mask.size*100:.1f}%)\"); axes[2].axis('off')\n",
                "axes[3].imshow(slope, cmap='magma'); axes[3].set_title(\"ALOS PALSAR Slope\"); axes[3].axis('off')\n",
                "axes[4].imshow(dem, cmap='terrain'); axes[4].set_title(\"ALOS PALSAR DEM\"); axes[4].axis('off')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Patch-Level Binary Labelling & Threshold Sensitivity Curve"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "masks = []\n",
                "for s_id, i_p, m_p in samples:\n",
                "    with h5py.File(m_p, 'r') as f_m:\n",
                "        masks.append(f_m['mask'][:])\n",
                "\n",
                "thresholds = [1, 16, 50, 100, 164, 328, 500, 1000]\n",
                "df_thresh = LandslideDatasetLoader.evaluate_threshold_distribution(masks, thresholds)\n",
                "print(df_thresh.to_string(index=False))\n",
                "\n",
                "plt.figure(figsize=(10, 5))\n",
                "plt.plot(df_thresh['threshold_pixels'], df_thresh['positive_ratio'] * 100, marker='o', color='crimson', label='% Landslide Patches')\n",
                "plt.plot(df_thresh['threshold_pixels'], df_thresh['negative_ratio'] * 100, marker='s', color='navy', label='% Stable Patches')\n",
                "plt.axvline(x=100, color='gray', linestyle='--', label='Selected Default Threshold (100 px = ~0.6%)')\n",
                "plt.xlabel('Landslide Pixel Threshold (pixels per 128x128 patch)')\n",
                "plt.ylabel('Percentage of Patches (%)')\n",
                "plt.title('Threshold Sensitivity Curve: Effect on Class Distribution')\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.legend()\n",
                "plt.show()"
            ]
        }
    ]
    create_notebook(cells, os.path.join("ml_service", "notebooks", "01_dataset_exploration.ipynb"))

def build_preprocessing_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# GeoGuard: Satellite Landslide Detection\n",
                "## Notebook 02: Preprocessing, Normalization & Data Preparation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
                "if PROJECT_ROOT not in sys.path:\n",
                "    sys.path.insert(0, PROJECT_ROOT)\n",
                "\n",
                "from src.data_loader import LandslideDatasetLoader\n",
                "from src.preprocessing import SatellitePreprocessor\n",
                "\n",
                "raw_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')\n",
                "loader = LandslideDatasetLoader(data_dir=raw_dir, pixel_threshold=100)\n",
                "X_train, y_train, meta_train = loader.load_split('train')\n",
                "X_val, y_val, meta_val = loader.load_split('validation')\n",
                "print(f\"X_train shape: {X_train.shape}, dtype: {X_train.dtype}\")\n",
                "print(f\"X_val shape: {X_val.shape}, dtype: {X_val.dtype}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 2. Computing Training Statistics (Zero Data Leakage)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "preprocessor = SatellitePreprocessor(band_names=['B2', 'B3', 'B4', 'B8'])\n",
                "preprocessor.fit(X_train)\n",
                "for i, bname in enumerate(preprocessor.band_names):\n",
                "    print(f\"Band {bname:2s}: Mean = {preprocessor.mean[i]:.4f}, Std = {preprocessor.std[i]:.4f}\")\n",
                "X_train_norm = preprocessor.transform(X_train)\n",
                "X_val_norm = preprocessor.transform(X_val)\n",
                "print(f\"Normalized train min/max: {X_train_norm.min():.2f} / {X_train_norm.max():.2f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 3. Comparing Channel Distributions Before & After Normalization"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(2, 4, figsize=(16, 6))\n",
                "band_names = ['B2 (Blue)', 'B3 (Green)', 'B4 (Red)', 'B8 (NIR)']\n",
                "for c in range(4):\n",
                "    axes[0, c].hist(X_train[:, :, :, c].flatten(), bins=40, color='royalblue', alpha=0.7)\n",
                "    axes[0, c].set_title(f\"Raw {band_names[c]}\")\n",
                "    axes[1, c].hist(X_train_norm[:, :, :, c].flatten(), bins=40, color='forestgreen', alpha=0.7)\n",
                "    axes[1, c].set_title(f\"Normalized {band_names[c]}\")\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 4. Data Augmentations (Rotation and Flip Invariance)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "sample_patch = X_train[0]\n",
                "fig, axes = plt.subplots(1, 4, figsize=(16, 4))\n",
                "axes[0].imshow(SatellitePreprocessor.to_rgb_display(sample_patch))\n",
                "axes[0].set_title(\"Original Patch\"); axes[0].axis('off')\n",
                "for i in range(1, 4):\n",
                "    aug = SatellitePreprocessor.augment_patch(sample_patch, seed=i*42)\n",
                "    axes[i].imshow(SatellitePreprocessor.to_rgb_display(aug))\n",
                "    axes[i].set_title(f\"Augmented Variant #{i}\"); axes[i].axis('off')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 5. Saving Preprocessing Metadata for Model Serving"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "config_path = os.path.join(PROJECT_ROOT, 'models', 'preprocessing_config.json')\n",
                "preprocessor.save_config(config_path)\n",
                "print(f\"Saved config to {config_path}\")"
            ]
        }
    ]
    create_notebook(cells, os.path.join("ml_service", "notebooks", "02_preprocessing.ipynb"))

def build_training_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# GeoGuard: Satellite Landslide Detection\n",
                "## Notebook 03: CNN Model Training & Validation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys\n",
                "import numpy as np\n",
                "import tensorflow as tf\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
                "if PROJECT_ROOT not in sys.path:\n",
                "    sys.path.insert(0, PROJECT_ROOT)\n",
                "\n",
                "from src.train import run_training\n",
                "from src.model import build_landslide_cnn\n",
                "\n",
                "print(f\"TensorFlow Version: {tf.__version__}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 2. Inspecting the Model Architecture"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model = build_landslide_cnn(input_shape=(128, 128, 4))\n",
                "model.summary()\n",
                "print(f\"Total trainable parameters: {model.count_params():,}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 3. Executing Model Training"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "data_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')\n",
                "trained_model, metadata = run_training(\n",
                "    data_dir=data_dir,\n",
                "    epochs=25,\n",
                "    batch_size=16,\n",
                "    learning_rate=1e-3,\n",
                "    seed=42\n",
                ")\n",
                "print(f\"Best Val Loss: {metadata['best_val_loss']:.4f}\")\n",
                "print(f\"Best Val Accuracy: {metadata['best_val_accuracy']:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 4. Plotting Loss and Accuracy Learning Curves"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "hist = metadata['history']\n",
                "epochs_range = range(1, len(hist['loss']) + 1)\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
                "ax1.plot(epochs_range, hist['loss'], 'b-o', label='Training Loss')\n",
                "ax1.plot(epochs_range, hist['val_loss'], 'r--s', label='Validation Loss')\n",
                "ax1.set_title('Training & Validation Loss', fontweight='bold')\n",
                "ax1.grid(True, alpha=0.3)\n",
                "ax1.legend()\n",
                "ax2.plot(epochs_range, hist['accuracy'], 'b-o', label='Training Accuracy')\n",
                "ax2.plot(epochs_range, hist['val_accuracy'], 'r--s', label='Validation Accuracy')\n",
                "ax2.set_title('Training & Validation Accuracy', fontweight='bold')\n",
                "ax2.grid(True, alpha=0.3)\n",
                "ax2.legend()\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        }
    ]
    create_notebook(cells, os.path.join("ml_service", "notebooks", "03_cnn_training.ipynb"))

def build_evaluation_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# GeoGuard: Satellite Landslide Detection\n",
                "## Notebook 04: Held-Out Test Evaluation & Diagnostic Metrics"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
                "if PROJECT_ROOT not in sys.path:\n",
                "    sys.path.insert(0, PROJECT_ROOT)\n",
                "\n",
                "from src.evaluation import evaluate_model\n",
                "from src.predict import LandslidePredictor\n",
                "from src.preprocessing import SatellitePreprocessor\n",
                "from src.data_loader import LandslideDatasetLoader"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 2. Running Evaluation on the Held-Out Test Set"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "report = evaluate_model(\n",
                "    model_path=os.path.join(PROJECT_ROOT, 'models', 'cnn_model.keras'),\n",
                "    config_path=os.path.join(PROJECT_ROOT, 'models', 'preprocessing_config.json'),\n",
                "    metadata_path=os.path.join(PROJECT_ROOT, 'models', 'model_metadata.json'),\n",
                "    data_dir=os.path.join(PROJECT_ROOT, 'data', 'raw'),\n",
                "    reports_dir=os.path.join(PROJECT_ROOT, 'reports'),\n",
                "    decision_threshold=0.40\n",
                ")\n",
                "print(\"=== HELD-OUT TEST METRICS (Threshold = 0.40) ===\")\n",
                "for k, v in report['metrics'].items():\n",
                "    print(f\"  {k:20s}: {v:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### 3. Displaying Evaluation Diagnostic Charts"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import matplotlib.image as mpimg\n",
                "reports_dir = os.path.join(PROJECT_ROOT, 'reports')\n",
                "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
                "axes[0].imshow(mpimg.imread(os.path.join(reports_dir, 'confusion_matrix.png'))); axes[0].axis('off')\n",
                "axes[1].imshow(mpimg.imread(os.path.join(reports_dir, 'roc_curve.png'))); axes[1].axis('off')\n",
                "axes[2].imshow(mpimg.imread(os.path.join(reports_dir, 'training_curves.png'))); axes[2].axis('off')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        }
    ]
    create_notebook(cells, os.path.join("ml_service", "notebooks", "04_model_evaluation.ipynb"))

def build_satellite_test_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# GeoGuard: Satellite Landslide Detection\n",
                "## Notebook 05: Copernicus Sentinel-2 Satellite Inference & AOI Tiling\n",
                "\n",
                "This notebook demonstrates the end-to-end satellite inference pipeline:\n",
                "1. **AOI Configuration**: Inspects the configured corridor without inventing unconfirmed coordinates.\n",
                "2. **Copernicus STAC Discovery**: Searches live Sentinel-2 acquisitions via CDSE STAC catalogue.\n",
                "3. **Scene Retrieval**: Acquires 4-channel optical bands (B02, B03, B04, B08) at 10m resolution (or runs in SIMULATED test mode if credentials are missing).\n",
                "4. **Visual Verification**: Displays True-Color RGB and Color-Infrared (CIR) composites.\n",
                "5. **Scene Tiling**: Divides the satellite scene into $128 \\times 128$ pixel patches with geographic tracking.\n",
                "6. **CNN Classification**: Runs the trained `cnn_model.keras` on valid patches.\n",
                "7. **Macro AOI Aggregation**: Aggregates patch results into a decision status."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
                "if PROJECT_ROOT not in sys.path:\n",
                "    sys.path.insert(0, PROJECT_ROOT)\n",
                "\n",
                "from config import settings\n",
                "from src.satellite import CopernicusClient, SatellitePipeline\n",
                "from src.predict import LandslidePredictor\n",
                "from src.preprocessing import SatellitePreprocessor\n",
                "\n",
                "print(f\"Highway Corridor Name: {settings.HIGHWAY_CORRIDOR_NAME}\")\n",
                "print(f\"Highway Corridor Bbox: {settings.HIGHWAY_AOI_BBOX} (None = unconfirmed, safe test fallback enabled)\")\n",
                "print(f\"Test AOI Bbox: {settings.TEST_AOI_BBOX}\")\n",
                "print(f\"CDSE Credentials configured: {CopernicusClient().has_credentials}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Live Copernicus STAC Catalogue Acquisition Discovery\n",
                "\n",
                "We query the public Copernicus Data Space Ecosystem STAC API (`https://stac.dataspace.copernicus.eu/v1/search`) to discover real Sentinel-2 Level-2A acquisitions covering the test AOI."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "client = CopernicusClient()\n",
                "acquisitions = client.search_acquisitions(bbox=settings.TEST_AOI_BBOX, limit=3)\n",
                "\n",
                "print(f\"Discovered {len(acquisitions)} live Sentinel-2 acquisitions:\")\n",
                "for a in acquisitions:\n",
                "    print(f\"  ID: {a['acquisition_id']}\")\n",
                "    print(f\"  Datetime: {a['acquisition_time']} | Cloud: {a['cloud_cover_percent']}% | Usable: {a['is_usable']}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Executing the Satellite Pipeline (Tiling, Inference, and Aggregation)\n",
                "\n",
                "We now invoke `SatellitePipeline.process_aoi()`. If live Copernicus credentials are absent, the system safely operates in `SIMULATED` test mode using verified real multispectral test patches."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pipeline = SatellitePipeline()\n",
                "result = pipeline.process_aoi(bbox=settings.TEST_AOI_BBOX, aoi_name=\"TEST_DEVELOPMENT_AOI\")\n",
                "\n",
                "print(\"=== AOI PROCESSING RESULT ===\")\n",
                "print(f\"Status: {result['status']} | Data Status: {result['data_status']}\")\n",
                "print(f\"Macro Decision: {result['prediction']['overall_class']}\")\n",
                "print(f\"Max Landslide Probability: {result['prediction']['max_probability']:.4f}\")\n",
                "print(f\"Satellite Source: {result['satellite']['source']} (Mode: {result['satellite']['mode']})\")\n",
                "print(f\"Acquisition ID: {result['satellite']['acquisition_id']}\")\n",
                "print(f\"Patches Evaluated: {result['aggregation']['total_patches_evaluated']} (Landslide={result['aggregation']['landslide_patches']}, Normal={result['aggregation']['normal_patches']})\")\n",
                "print(f\"High-Risk Coordinates: {result['aggregation']['high_risk_coordinates']}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Visualizing Individual Patch Predictions"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "patches = result.get('patch_details', [])\n",
                "if patches:\n",
                "    fig, axes = plt.subplots(len(patches), 2, figsize=(10, 4 * len(patches)))\n",
                "    if len(patches) == 1:\n",
                "        axes = np.expand_dims(axes, axis=0)\n",
                "        \n",
                "    for idx, p in enumerate(patches):\n",
                "        coords = p['center_coords']\n",
                "        pred_class = p['prediction_class']\n",
                "        prob = p['probability']\n",
                "        color = 'red' if pred_class == 'LANDSLIDE DETECTED' else 'green'\n",
                "        \n",
                "        axes[idx, 0].set_title(f\"Patch #{p['patch_index']} (Lat: {coords['lat']}, Lon: {coords['lon']})\")\n",
                "        axes[idx, 0].text(0.5, 0.5, f\"{pred_class}\\nP = {prob:.4f}\", fontsize=14, ha='center', va='center', color=color, fontweight='bold')\n",
                "        axes[idx, 0].axis('off')\n",
                "        \n",
                "        axes[idx, 1].text(0.1, 0.5, f\"Bounding Box:\\n{p['bbox']}\", fontsize=11)\n",
                "        axes[idx, 1].axis('off')\n",
                "    plt.tight_layout()\n",
                "    plt.show()"
            ]
        }
    ]
    create_notebook(cells, os.path.join("ml_service", "notebooks", "05_satellite_inference_test.ipynb"))

if __name__ == "__main__":
    build_exploration_notebook()
    build_preprocessing_notebook()
    build_training_notebook()
    build_evaluation_notebook()
    build_satellite_test_notebook()
