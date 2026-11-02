# GeoGuard — Satellite Image-Based Landslide Detection ML System

A modular, lightweight, and independent Machine Learning microservice providing **macro-level visual verification** for geotechnical IoT telemetry (soil moisture, precipitation, tilt, and vibration) using ESA Copernicus Sentinel-2 satellite imagery.

---

## 1. Executive Summary & Architecture

GeoGuard integrates two complementary sensor streams to protect critical highway corridors:
1. **Micro-Geotechnical IoT Telemetry**: Fast, localized in-situ sensors measuring subsurface pore-water pressure, volumetric soil moisture, 24-hour rainfall, geophone vibration, and inclinometer tilt.
2. **Macro-Visual Satellite Verification (This Service)**: Wide-area orbital optical verification using European Space Agency (ESA) Sentinel-2 multispectral imagery and a custom 4-channel Convolutional Neural Network (CNN).

```
+-------------------------------------------------------------------------+
|                  Copernicus Data Space Ecosystem (CDSE)                 |
|               Sentinel-2 Level-2A BOA Optical Observations              |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                 GeoGuard Autonomous Satellite Scheduler                 |
|       (Polls STAC Catalogue every 15 min; screens cloud cover <= 30%)    |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                 Tiling Engine & Preprocessing Pipeline                  |
|    - 128x128 Patch Extraction with Lat/Lon Geographic Tracking          |
|    - Band Alignment: [B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)]   |
|    - Z-Score Normalization using Training Baseline μ & σ                |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                   Custom 4-Channel Deep CNN Model                       |
|           3 Conv2D Blocks + BatchNorm + Dropout + GlobalAvgPool         |
|              Sigmoid Binary Classifier (Tuned Threshold = 0.40)         |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|              SQLite Prediction Store & Spatial Aggregator               |
|      Identifies high-risk GPS coordinates & macro corridor alert        |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                   FastAPI Microservice (REST API)                       |
|       /health | /latest-prediction | /predictions | /satellite-status   |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|             GeoGuard Website & React Native Mobile Dashboard            |
|   (Consumes REST endpoints without modifying internal frontend code)    |
+-------------------------------------------------------------------------+
```

---

## 2. Model Selection & Design Rationale

### Why a Custom 4-Channel CNN?
- **Spatial Inductive Bias**: Tabular models (Random Forests, XGBoost) treat pixels as independent features or rely on handcrafted textures. CNNs use 2D convolution kernels to naturally identify spatial morphological signatures—such as headscarps, tension cracks, debris fans, and denuded vegetation chutes.
- **Multispectral Bands (4 Optical Bands)**:
  - **B02 (Blue, 490 nm)**: Channel index 1 in Landslide4Sense. Native 10m resolution.
  - **B03 (Green, 560 nm)**: Channel index 2 in Landslide4Sense. Native 10m resolution.
  - **B04 (Red, 665 nm)**: Channel index 3 in Landslide4Sense. Native 10m resolution.
  - **B08 (Near-Infrared / NIR, 842 nm)**: Channel index 7 in Landslide4Sense. Native 10m resolution. Essential for vegetation chlorophyll reflectance drop during slope stripping.
  - *Omission of Radar & DEM*: While Landslide4Sense includes ALOS PALSAR radar and SRTM slope, these are **not** present in standard Sentinel-2 optical acquisitions. Incorporating them would prevent direct inference on live Sentinel-2 feeds.
- **Lightweight CPU Footprint**:
  - Total parameters: **149,569** (~584 KB in memory, 1.82 MB saved `.keras` file).
  - Can run inference on a low-cost cloud VM or local CPU in under **25 milliseconds per patch**.

---

## 3. Dataset Preprocessing & Verified Evaluation

### Dataset & Partitioning
- **Source**: Landslide4Sense benchmark dataset (derived from NASA & IBM geospatial challenges).
- **Patch Dimensions**: $128 \times 128$ pixels per patch (covering approx. $1.64\text{ km}^2$ per patch at 10m resolution).
- **Threshold Calibration**: Patches containing $\ge 100$ landslide pixels (~0.61% of patch area / ~1 hectare) are classified as positive (`LANDSLIDE DETECTED`). This yielded 57 positive and 63 negative training patches, avoiding severe class imbalance.
- **Data Leakage Prevention**: Split strictly into **120 training**, **30 validation**, and **60 held-out test** samples. The test set remained completely unseen during training and threshold calibration.

### Verified Test Set Performance (Untouched 60 Test Samples)
- **Overall Accuracy**: **75.00%** (45 / 60 correctly classified)
- **Precision**: **80.77%**
- **Landslide Recall**: **67.74%** at standard 0.50 threshold (21 / 31 landslides detected)
- **Normal/Stable Recall**: **82.76%** (24 / 29 stable patches detected)
- **F1-Score**: **73.68%**
- **ROC-AUC**: **0.8521**
- **Decision Threshold Optimization**: In landslide hazard monitoring, false negatives (missing an active landslide) carry severe human and structural costs. Validation set analysis demonstrated that adjusting the decision threshold from `0.50` to `0.40` increases landslide recall to **81.82%** with **90.00%** precision. Hence, `0.40` is configured as the default operational threshold.

### Normalization Baseline (Computed Strictly on Training Set)
```json
{
  "selected_bands": ["B02", "B03", "B04", "B08"],
  "band_indices": [1, 2, 3, 7],
  "means": [0.9098, 0.9972, 1.0374, 1.0323],
  "stds": [0.1672, 0.2284, 0.3960, 0.3295]
}
```

---

## 4. Copernicus Data Space Ecosystem (CDSE) Integration

- **Catalogue Discovery**: Uses the CDSE STAC v1 API (`https://stac.dataspace.copernicus.eu/v1/search`) querying the `sentinel-2-l2a` collection.
- **Authentication**: OAuth2 Client Credentials flow via `identity.dataspace.copernicus.eu`.
- **Safe Simulated Mode**: When CDSE credentials are not configured or during offline testing, the system automatically runs in `SIMULATED` test mode using verified Landslide4Sense test scenes, explicitly returning `data_status: "SIMULATED"` and `is_simulated: true` so test data is never mistaken for real observations.
- **Corridor AOI Protection**: The actual highway corridor is **never invented**. Set `HIGHWAY_AOI_BBOX` in `.env` when the highway team confirms coordinates. Until then, `TEST_DEVELOPMENT_AOI` is used for testing.

---

## 5. Scheduler: Polling Interval vs Satellite Revisit Rate

> [!IMPORTANT]
> **Understanding the 15-Minute Polling vs 5-Day Satellite Revisit**:
> - Sentinel-2 satellites (Sentinel-2A and Sentinel-2B) pass over the same geographical corridor approximately once every **5 days** (at the equator) or 2–3 days at higher latitudes.
> - Once the satellite captures an image, the ground station downlinks and processes the raw data into Level-2A Bottom-Of-Atmosphere (BOA) reflectance. This data is published to the CDSE catalogue at irregular times throughout the day.
> - **Why poll every 15 minutes?** Polling every 15 minutes ensures that as soon as the newly processed ESA product is uploaded to the catalogue, GeoGuard detects and processes it within minutes, rather than waiting days.
> - **Deduplication**: The scheduler verifies the scene's `acquisition_id` in SQLite. If no new scene is found, it preserves the last valid prediction and its original acquisition timestamp. It **never** fabricates fake data or updates timestamps on stale imagery.

---

## 6. Directory Structure

```
ml_service/
├── .env.example              # Template configuration for AOI and CDSE credentials
├── config.py                 # Central configuration loader
├── requirements.txt          # Python dependencies
├── app.py                    # FastAPI application server
├── scheduler.py              # Autonomous Copernicus polling daemon
├── API_INTEGRATION.md        # API reference for frontend website team
├── README.md                 # This documentation
├── data/
│   ├── geoguard_predictions.db # SQLite database storing prediction history
│   └── raw/                  # Landslide4Sense benchmark dataset partitions
├── models/
│   ├── cnn_model.keras       # Trained lightweight 4-channel CNN weights
│   ├── model_metadata.json   # Model architecture and training metadata
│   └── preprocessing_config.json # Channel means, stds, and band mappings
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_cnn_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_satellite_inference_test.ipynb
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_curves.png
│   └── test_evaluation_report.json
├── src/
│   ├── data_loader.py        # HDF5 dataset loader and band extractor
│   ├── preprocessing.py      # Z-score normalization and augmentations
│   ├── model.py              # Keras CNN model definition
│   ├── train.py              # CNN training loop with early stopping
│   ├── evaluation.py         # Multi-metric evaluation and plotting
│   ├── predict.py            # Single-patch and batch predictor
│   ├── satellite.py          # CDSE STAC client, tiling engine, spatial aggregator
│   ├── database.py           # SQLite storage engine and query helpers
│   └── schemas.py            # Pydantic v2 validation models
└── tests/
    ├── test_data_loader.py   # Dataset loading unit tests
    ├── test_model_pipeline.py # CNN inference and evaluation tests
    ├── test_satellite_pipeline.py # Tiling and Copernicus STAC tests
    ├── test_database.py      # SQLite operations and deduplication tests
    ├── test_api.py           # FastAPI endpoints integration tests
    └── test_scheduler.py     # Polling, cloud screening, and duplicate tests
```

---

## 7. Step-by-Step Running Instructions (Windows / Linux)

### Step 1: Install Dependencies
Open PowerShell or your terminal in the project root:
```powershell
pip install -r ml_service/requirements.txt
```

### Step 2: Configure Environment Variables
Copy the example environment file:
```powershell
cp ml_service/.env.example ml_service/.env
```
Edit `ml_service/.env` to configure your highway corridor bounding box and CDSE credentials (optional for simulated mode):
```ini
HIGHWAY_CORRIDOR_NAME=NH-58_Rishikesh_Devprayag
HIGHWAY_AOI_BBOX=78.25,30.05,78.60,30.20
CDSE_CLIENT_ID=your_client_id_here
CDSE_CLIENT_SECRET=your_client_secret_here
```

### Step 3: Run Full Test Suite
Verify that all unit and integration tests pass:
```powershell
python -m pytest ml_service/tests/
```

### Step 4: Start the FastAPI Microservice
Launch the server with Uvicorn:
```powershell
python -m uvicorn ml_service.app:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: **http://127.0.0.1:8000/docs**
- Health Check: **http://127.0.0.1:8000/health**
- Latest Prediction: **http://127.0.0.1:8000/latest-prediction**

### Step 5: Run the Autonomous Satellite Scheduler
To run a single catalogue check and exit:
```powershell
python ml_service/scheduler.py --once
```
To run continuously as a background daemon (polling every 15 minutes):
```powershell
python ml_service/scheduler.py
```

### Step 6: Interactive Jupyter Notebooks
Start Jupyter to inspect datasets, train models, or test inference interactively:
```powershell
jupyter notebook ml_service/notebooks/
```
Notebooks available:
- `01_dataset_exploration.ipynb`: Multispectral bands, pixel masks, and threshold balance.
- `02_preprocessing.ipynb`: Channel normalization, D4 dihedral augmentations, CIR composites.
- `03_cnn_training.ipynb`: Step-by-step training with learning curves.
- `04_model_evaluation.ipynb`: ROC-AUC, confusion matrix, and threshold optimization.
- `05_satellite_inference_test.ipynb`: Tiling scene arrays and running corridor predictions.
