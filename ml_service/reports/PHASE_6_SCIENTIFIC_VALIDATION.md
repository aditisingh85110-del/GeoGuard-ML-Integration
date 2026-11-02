# GeoGuard Phase 6: Scientific Validation of Real Satellite Inference

**Validation Execution Date**: 2026-11-01  
**Project**: GeoGuard Satellite Image-Based Landslide Detection ML System  
**Service Path**: `ml_service/`  
**Model Under Test**: `GeoGuard_CNN_v1` (`ml_service/models/cnn_model.keras`)  
**Report Status**: **SCIENTIFIC AUDIT COMPLETE — OPERATIONAL DEPLOYMENT BLOCKED PENDING RADIOMETRIC & SPATIAL REMEDIATION**

---

## 1. Executive Summary & Verdict

Phase 6 conducted a controlled, empirical scientific investigation into the inference behavior of `GeoGuard_CNN_v1` on real Sentinel-2 Level-2A satellite data acquired through the Copernicus Data Space Ecosystem (CDSE).

While OAuth2 authentication, STAC catalog discovery, and Sentinel Hub Processing API raster ingestion function properly from a software engineering perspective, **the end-to-end inference output cannot currently be certified as scientifically valid**.

### Summary of Scientific Findings

| Investigation Dimension | Empirical Finding | Scientific Verdict |
|:---|:---|:---|
| **Satellite Product Verification** | `S2B_MSIL2A_20261029T050929_N0513_R019_T43PGP_20261029T105708` acquired 2026-10-29T05:09:29Z. Cloud cover: 9.25%. | **VERIFIED REAL ACQUISITION** |
| **Monitored Highway AOI** | `HIGHWAY_AOI_BBOX` in `ml_service/.env` is unconfigured. Development placeholder in Southern Karnataka (`[76.95, 12.00, 77.05, 12.10]`) was used. | **NOT MONITORING HIGHWAY** |
| **Spatial Resolution & Scale** | AOI spans $10.9\text{ km} \times 11.1\text{ km}$. Output is $128 \times 128$ pixels, yielding $\approx 86\text{ m/pixel}$ (native Sentinel-2 is $10\text{ m/pixel}$). | **8.6x RESOLUTION DEGRADATION** |
| **Radiometric Normalization** | Training data (Landslide4Sense) has per-band means $\approx 1.0$. Live CDSE BOA surface reflectance has per-band means $0.057 - 0.288$. | **SEVERE RADIOMETRIC MISMATCH (3.6x - 16.0x)** |
| **Input Clipping & Z-Scores** | Normalized B02 inputs average $Z = -5.10$. **91.69% of all B02 pixels are clamped at $-5.0$**. Overall input tensor mean is $-3.43$. | **EXTREME OUT-OF-DISTRIBUTION INPUT** |
| **CNN Activation Behavior** | Extreme negative inputs cause **80.5% ReLU activation extinction** in Layer 1 Block 2 and 65.6% in the classification head. | **NETWORK NEURON SATURATION / SHUTDOWN** |
| **Prediction Significance ($P=0.1153$)** | Baseline output on an all-zeros input is $P=0.0924$. Output on live raster is $P=0.1153$. Output reflects class-prior bias, not terrain stability. | **ARTIFACT OF RELU EXTINCTION** |

> [!CAUTION]
> **OPERATIONAL VERDICT**: The system classification of `NORMAL / STABLE` ($P = 0.1153$) is **not** a valid detection of stable terrain. It is an artifact of neural network deactivation caused by feeding physical bottom-of-atmosphere reflectance $[0.0, 1.0]$ into a model trained on Landslide4Sense unit-centered inputs ($\mu \approx 1.0$). Operating this model on live CDSE data in production without radiometric calibration presents a severe risk of **false negatives (missed landslides)**.

---

## 2. Satellite Acquisition & Request Verification

### 2.1 Product Metadata
The satellite raster ingested during Phase 6 was retrieved from the official Copernicus Data Space Ecosystem using the Sentinel Hub Processing API:

- **Sentinel-2 Product Identifier**: `S2B_MSIL2A_20261029T050929_N0513_R019_T43PGP_20261029T105708`
- **Sensing / Acquisition Date & Time**: `2026-10-29T05:09:29.024000Z`
- **Data Ingestion / Verification Timestamp**: `2026-11-01T13:19:15Z`
- **Platform / Constellation**: Sentinel-2B (MultiSpectral Instrument, Level-2A BOA Surface Reflectance)
- **Relative Orbit**: R019
- **MGRS Tile ID**: `43PGP`
- **Product Cloud Cover Percentage**: `9.25%`
- **Data Format**: 4-channel IEEE 32-bit floating point GeoTIFF (`FLOAT32`)

### 2.2 Processing API Query Specification
To guarantee that the exact 2026-10-29 acquisition was processed (and not the 2017 catalog result found during earlier OData catalogue browsing), the Sentinel Hub Processing API request was strictly bounded:

```json
{
  "input": {
    "bounds": {
      "bbox": [76.95, 12.00, 77.05, 12.10],
      "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
    },
    "data": [{
      "type": "sentinel-2-l2a",
      "dataFilter": {
        "timeRange": {
          "from": "2026-10-29T00:00:00Z",
          "to": "2026-10-29T23:59:59Z"
        },
        "maxCloudCoverage": 30.0
      }
    }]
  },
  "output": {
    "width": 128,
    "height": 128,
    "responses": [{"format": {"type": "image/tiff"}}]
  },
  "evalscript": "//VERSION=3\nfunction setup() { return { input: ['B02', 'B03', 'B04', 'B08'], output: { bands: 4, sampleType: 'FLOAT32' } }; }\nfunction evaluatePixel(sample) { return [sample.B02, sample.B03, sample.B04, sample.B08]; }"
}
```

### 2.3 Corridor Relevance & Highway Coverage
- **Monitored Highway Corridor**: `HIGHWAY_AOI_BBOX` in `ml_service/.env` remains empty (`None`).
- **Target Coordinates Used**: Bounding box `[min_lon=76.95, min_lat=12.00, max_lon=77.05, max_lat=12.10]`.
- **Geographic Location**: Rural agricultural / scrub plains in Southern Karnataka, India (near Chamarajanagar).
- **Corridor Relevance Assessment**: **Zero coverage of the target highway corridor.** In accordance with project instructions, no fictitious highway coordinates were invented. This acquisition serves solely as an end-to-end technical probe.

---

## 3. Radiometric Normalization Mismatch Investigation

### 3.1 Empirical Comparison Table
The table below compares the Landslide4Sense training set distributions against the raw physical reflectance values returned by Copernicus CDSE Sentinel-2 L2A for the test patch:

| Channel | Sentinel-2 Band | Central $\lambda$ | Landslide4Sense Training $\mu$ | Landslide4Sense Training $\sigma$ | Live CDSE Raw $\mu$ | Live CDSE Raw Range $[\min, \max]$ | Shift ($Z$-score) | Scale Discrepancy ($\mu_{\text{train}} / \mu_{\text{CDSE}}$) | Clamped at $-5.0$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | **B02 (Blue)** | 490 nm | **0.9098** | 0.1672 | **0.0567** | $[0.0264, 0.2682]$ | **$-5.10\sigma$** | **16.05x** | **91.69%** |
| **1** | **B03 (Green)** | 560 nm | **0.9972** | 0.2284 | **0.0792** | $[0.0252, 0.2970]$ | **$-4.02\sigma$** | **12.59x** | **0.00%** |
| **2** | **B04 (Red)** | 665 nm | **1.0374** | 0.3960 | **0.0663** | $[0.0190, 0.3208]$ | **$-2.45\sigma$** | **15.65x** | **0.00%** |
| **3** | **B08 (NIR)** | 842 nm | **1.0323** | 0.3295 | **0.2883** | $[0.0000, 0.6354]$ | **$-2.26\sigma$** | **3.58x** | **0.00%** |

### 3.2 Root Cause Analysis
1. **Landslide4Sense Convention**:
   - In the benchmark Landslide4Sense dataset (Ghorbanzadeh et al., IEEE GRSL / IGARSS 2022), all 14 channels (including optical bands, slope, and elevation) were converted to float32 and pre-normalized around $1.0$ across the challenge dataset.
   - Consequently, the optical bands do not represent raw physical surface reflectance $[0.0, 1.0]$. Instead, every channel has a mean of approximately $1.0$.
2. **Copernicus CDSE L2A Convention**:
   - The Copernicus Sentinel Hub Processing API with `sampleType: 'FLOAT32'` outputs standard physical Bottom-of-Atmosphere (BOA) surface reflectance in $[0.0, 1.0]$.
   - In natural rural and mountainous terrain, optical surface reflectance typically ranges between $0.02$ and $0.15$ in the visible bands (B02, B03, B04) and between $0.20$ and $0.45$ in the near-infrared band (B08).
3. **The Normalization Breakdown**:
   When standard z-score normalization was applied:
   $$Z_c = \frac{X_c - \mu_{\text{train}, c}}{\sigma_{\text{train}, c}}$$
   - For B02, $(0.0567 - 0.9098) / 0.1672 = -5.101$. Because the preprocessor enforces an outlier clamp $[-5.0, +5.0]$, **91.69% of all B02 pixels were hard-clamped at $-5.0000$**.
   - For B03, values mapped to $[-4.26, -3.07]$.
   - For B04, values mapped to $[-2.57, -1.81]$.
   - For B08, values mapped to $[-3.13, -1.20]$.
   - The entire 4-channel input tensor has a global mean of **$-3.4302$**, representing an extreme negative shift into the unpopulated tail of the training distribution.

---

## 4. Controlled Inference Diagnostics & Activation Analysis

### 4.1 Diagnostic Artifacts Created
All raw and processed intermediate data have been serialized for reproducibility:
- **Raw CDSE 4-Band Raster**: `ml_service/reports/diagnostics_raw_cdse_raster.npy` (Shape: `(128, 128, 4)`, float32)
- **Normalized CNN Input Tensor**: `ml_service/reports/diagnostics_normalized_cdse_tensor.npy` (Shape: `(128, 128, 4)`, float32)
- **Distribution Comparison Record**: `ml_service/reports/diagnostics_distribution_comparison.json`
- **True-Color RGB Preview**: `ml_service/reports/diagnostics_cdse_rgb.png`
- **False-Color CIR Preview**: `ml_service/reports/diagnostics_cdse_cir.png`

### 4.2 Layer-by-Layer Activation Trace
To understand whether the prediction probability ($P = 0.1153$) indicates genuine absence of landslides or neuron starvation, layer-by-layer activations were inspected:

```text
=== LAYER ACTIVATION ANALYSIS ===
Layer satellite_patch_input (InputLayer):          shape=(1, 128, 128, 4),   mean=-3.4302, zeros=0.0%
Layer conv1_1 (Conv2D):                            shape=(1, 128, 128, 32),  mean=-0.0001, zeros=0.0%
Layer bn1_1 (BatchNormalization):                  shape=(1, 128, 128, 32),  mean=-0.1548, zeros=0.0%
Layer relu1_1 (Activation):                        shape=(1, 128, 128, 32),  mean=0.8699,  zeros=65.4%
Layer conv1_2 (Conv2D):                            shape=(1, 128, 128, 32),  mean=-0.6872, zeros=0.0%
Layer bn1_2 (BatchNormalization):                  shape=(1, 128, 128, 32),  mean=-0.8189, zeros=0.0%
Layer relu1_2 (Activation):                        shape=(1, 128, 128, 32),  mean=0.5118,  zeros=80.5%  <-- 80.5% NEURONS EXTINCTION
Layer pool1 (MaxPooling2D):                        shape=(1, 64, 64, 32),    mean=0.5301,  zeros=78.7%
Layer conv2_1 (Conv2D):                            shape=(1, 64, 64, 64),    mean=0.1616,  zeros=0.0%
Layer relu2_1 (Activation):                        shape=(1, 64, 64, 64),    mean=0.7218,  zeros=45.7%
Layer conv2_2 (Conv2D):                            shape=(1, 64, 64, 64),    mean=0.1791,  zeros=0.0%
Layer relu2_2 (Activation):                        shape=(1, 64, 64, 64),    mean=0.6870,  zeros=50.3%
Layer pool2 (MaxPooling2D):                        shape=(1, 32, 32, 64),    mean=0.7171,  zeros=48.7%
Layer conv3_1 (Conv2D):                            shape=(1, 32, 32, 128),   mean=-0.1249, zeros=0.0%
Layer relu3_1 (Activation):                        shape=(1, 32, 32, 128),   mean=0.5140,  zeros=51.3%
Layer pool3 (MaxPooling2D):                        shape=(1, 16, 16, 128),   mean=0.5484,  zeros=49.5%
Layer gap (GlobalAveragePooling2D):                shape=(1, 128),           mean=0.5484,  zeros=20.3%
Layer dense1 (Dense):                              shape=(1, 64),            mean=-0.4341, zeros=0.0%
Layer bn_dense (BatchNormalization):               shape=(1, 64),            mean=-0.4920, zeros=0.0%
Layer relu_dense (Activation):                     shape=(1, 64),            mean=0.4810,  zeros=65.6%  <-- 65.6% HEAD EXTINCTION
Layer landslide_probability (Dense Sigmoid):       shape=(1, 1),             probability=0.115317
```

### 4.3 Baseline Comparison Test
To establish whether $P = 0.1153$ is a meaningful signal or baseline bias, an unactivated neutral input (all zeros, corresponding to training mean $\mu$) was tested under identical conditions:

- **Model Output on All-Zeros Neutral Patch**: `0.0924` ($9.24\%$)
- **Model Output on Live CDSE Test Patch**: `0.1153` ($11.53\%$)
- **Model Final Dense Intercept Bias**: $\beta_0 \approx -2.037$ ($\sigma(-2.037) \approx 0.1153$)

### 4.4 Scientific Deduction
Because the input tensor is uniformly negative ($Z \approx -3.43$), the first convolution layer generates predominantly negative pre-activations. In Layer 1 Block 2, **80.5% of all feature channels are clamped to zero by the ReLU activation function**.

The network's deeper representation collapses, preventing meaningful feature extraction. The final dense layer is deprived of discriminatory spatial features and simply outputs its prior training class bias ($\approx 0.09 - 0.11$).

**Conclusion**: The classification of `NORMAL / STABLE` is **an artifact of ReLU activation extinction**, not a verified recognition of stable terrain.

---

## 5. Spatial Validity Audit

### 5.1 Geographic Projection & Grid Dimensions
- **Coordinate Reference System**: EPSG:4326 (WGS84 ellipsoidal latitude / longitude).
- **Bounding Box**: `[min_lon=76.95, min_lat=12.00, max_lon=77.05, max_lat=12.10]`.
- **Longitude Span**: $\Delta \lambda = 0.10^\circ \approx 10.89\text{ km}$ (at $\phi = 12.05^\circ$).
- **Latitude Span**: $\Delta \phi = 0.10^\circ \approx 11.06\text{ km}$.
- **Total Area Covered**: $\approx 120.4\text{ km}^2$.

### 5.2 Spatial Resolution Degradation
- **Raster Dimensions**: $128 \times 128$ pixels.
- **Pixel Spacing (East-West)**: $10,886\text{ m} / 128\text{ pixels} = 85.05\text{ m/pixel}$.
- **Pixel Spacing (North-South)**: $11,058\text{ m} / 128\text{ pixels} = 86.39\text{ m/pixel}$.
- **Effective Ground Sampling Distance (GSD)**: **$\approx 85.7\text{ meters per pixel}$**.
- **Sentinel-2 Native GSD (B02, B03, B04, B08)**: **$10.0\text{ meters per pixel}$**.

### 5.3 Consequences for Landslide Detection
1. **8.6x Spatial Downsampling**:
   The Processing API resampled the imagery down from 10m to 86m to fit the requested $128 \times 128$ grid over a $0.10^\circ$ AOI.
2. **Ground Area Footprint Mismatch**:
   - In Landslide4Sense, each $128 \times 128$ patch represents $1.28\text{ km} \times 1.28\text{ km}$ ($1.64\text{ km}^2$) at native 10m resolution.
   - The test raster covers $120.4\text{ km}^2$—an area **73 times larger** than the training patches.
3. **Loss of Geomorphological Detail**:
   Typical roadside landslides span widths of 20m to 100m. At 86m/pixel, a 30m landslide occupies only $\approx 0.12$ pixels. Spatial features (scarps, flow tracks, debris fans) are completely averaged out.

---

## 6. Temporal Validity Audit

- **Sensing Timestamp**: `2026-10-29T05:09:29.024000Z`
- **Retrieval / Inference Timestamp**: `2026-11-01T13:19:15Z`
- **Elapsed Ingestion Latency**: 3 days, 8 hours, 9 minutes.
- **Observation Frequency**: Sentinel-2 constellation provides 5-day revisit cycles at the equator (and 2–3 days at mid-latitudes). An acquisition 3 days old is within the expected observation window for macro-level geotechnical verification.
- **Pipeline Modality**: The current execution was a targeted, single-scene diagnostic pull. It does not yet run as an active autonomous monitoring loop over an established highway corridor.

---

## 7. Remediation Plan & Roadmap to Operational Readiness

To transition GeoGuard from a technical prototype to a scientifically certified monitoring service, the following three-phase remediation plan is required:

```mermaid
flowchart TD
    subgraph Step 1: Radiometric Alignment
        A[CDSE BOA Reflectance [0.0, 1.0]] --> B[Recalibrate Landslide4Sense to Physical BOA Reflectance]
        B --> C[Retrain GeoGuard_CNN_v2 on Physical BOA Reflectance]
        C --> D[Update Preprocessing Config to Natural BOA Statistics]
    end
    subgraph Step 2: Spatial Tiling Pipeline
        E[Confirmed Highway Corridor AOI] --> F[Request Imagery at Native 10m Resolution]
        F --> G[Extract 128x128 Patches with 20% Stride Overlap]
        G --> H[Run GeoGuard_CNN_v2 on Each 1.28km x 1.28km Patch]
    end
    subgraph Step 3: Production Validation
        D --> I[Live Highway Corridor Inference]
        H --> I
        I --> J[Validate Predictions Against Ground-Truth Geotechnical IoT Events]
    end
```

### Action 1: Radiometric Alignment (Retraining on Physical BOA Reflectance)
- **Problem**: `GeoGuard_CNN_v1` was trained on Landslide4Sense unit-centered data ($\mu \approx 1.0$).
- **Solution**:
  - Rescale Landslide4Sense optical bands back to physical Sentinel-2 BOA reflectance $[0.0, 1.0]$ using published sensor calibration factors.
  - Re-compute normalization parameters ($\mu, \sigma$) based on physical reflectance:
    - B02: $\mu \approx 0.055, \sigma \approx 0.035$
    - B03: $\mu \approx 0.080, \sigma \approx 0.040$
    - B04: $\mu \approx 0.065, \sigma \approx 0.045$
    - B08: $\mu \approx 0.285, \sigma \approx 0.090$
  - Retrain the CNN to produce `GeoGuard_CNN_v2.keras`.

### Action 2: 10m Native Resolution Tiling Engine
- **Problem**: Requesting fixed $128 \times 128$ dimensions over an arbitrary bounding box degrades resolution from 10m to 86m.
- **Solution**:
  - Configure Sentinel Hub Processing API to request native 10m resolution (specifying `resolution: [10, 10]` in meters instead of fixed pixel dimensions).
  - Implement a corridor patch-tiler that slices the resulting raster into $128 \times 128$ pixel patches ($1.28\text{ km} \times 1.28\text{ km}$), matching the spatial footprint of the training data.
  - Apply inference per patch and georeference patch predictions back to highway mileage markers.

### Action 3: Highway AOI Configuration
- **Problem**: `ml_service/.env` currently lacks the confirmed highway corridor coordinates.
- **Solution**:
  - Obtain confirmed highway corridor coordinates from project stakeholders.
  - Populate `HIGHWAY_CORRIDOR_NAME` and `HIGHWAY_AOI_BBOX` in `ml_service/.env`.
  - Re-enable the autonomous background polling scheduler once Actions 1 and 2 are deployed.
