# 🛡️ GeoGuard AI — Highway Intelligence & Spaceborne Landslide Surveillance

> **A mission-critical disaster management control room, IoT telemetry analytics platform, and spaceborne Earth Observation screening system for mountainous Himalayan corridors.**

[![Repository](https://img.shields.io/badge/GitHub-GeoGuard--ML--Integration-blue.svg)](https://github.com/aditisingh85110-del/GeoGuard-ML-Integration.git)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![ESA Copernicus](https://img.shields.io/badge/ESA-Sentinel--2%20L2A-003399.svg)](https://dataspace.copernicus.eu/)

---

## 📖 Executive Summary

**GeoGuard AI** transforms raw geotechnical sensor streams and European Space Agency (ESA) Copernicus Sentinel-2 satellite telemetry into actionable landslide early-warning intelligence. Developed specifically for high-risk mountain transit corridors across the Indian Himalayan Arc, GeoGuard bridges physical IoT ground stations (ESP32/Arduino inclinometers, soil moisture, precipitation gauges) with deep-learning spaceborne change screening for district disaster authorities, the Border Roads Organisation (BRO), and the National Disaster Response Force (NDRF).

---

## 🏛️ System Architecture

```
GeoGuard-ML-Integration/
│
├── app.py                      # Main Streamlit control room & navigation router
├── requirements.txt            # Frontend & GIS dependencies
├── README.md                   # System documentation & setup guide
├── .gitignore                  # Git exclusion rules (protects .env & datasets)
│
├── .streamlit/
│   └── config.toml             # Geotechnical dark slate theme & layout
│
├── database/
│   ├── __init__.py
│   └── db.py                   # SQLite engine, schema, indexing & CRUD
│
├── modules/
│   ├── ui.py                   # Geotechnical CSS, dark theme, banners & KPI cards
│   ├── satellite_client.py     # Resilient HTTP client for FastAPI ML backend
│   ├── data_store.py           # Central data coordinator & cache layer
│   ├── risk_engine.py          # Multi-factor geotechnical risk model & gauge
│   ├── sensor_simulator.py     # Physics-based telemetry & 24h historical seeder
│   ├── hardware_reader.py      # Non-blocking PySerial IoT gateway
│   ├── map_manager.py          # Folium GIS map with colored risk corridors
│   ├── weather_service.py      # Real-time mountain weather (Open-Meteo)
│   ├── ml_inference.py         # Scikit-learn Random Forest landslide classifier
│   └── report_generator.py     # CSV downloads & printable incident dossiers
│
├── pages/
│   ├── overview.py             # 1. Highway Control Center Overview
│   ├── live_sensor.py          # 2. Live Sensor Telemetry & waveforms
│   ├── risk_intelligence.py    # 3. Geotechnical Risk Engine & gauge
│   ├── geographic_map.py       # 4. GIS Map & live asset tracking
│   ├── satellite_intelligence.py # 5. Autonomous Himalayan Fleet & Sentinel-2 Screening
│   ├── historical_analytics.py # 6. Cross-sensor correlation & trends
│   ├── alert_center.py         # 7. Incident lifecycle & audit trail
│   ├── highway_safety.py       # 8. Road closures & bypass routing
│   ├── emergency_response.py   # 9. Inter-agency crisis coordination
│   ├── hardware_management.py  # 10. PySerial COM port & sensor health
│   ├── reports.py              # 11. Data downloads & incident dossiers
│   └── settings.py             # 12. Thresholds, cooldowns & database tools
│
└── ml_service/                 # Independent FastAPI Satellite & ML Microservice
    ├── app.py                  # FastAPI application & REST endpoints
    ├── config.py               # Pydantic environment configuration
    ├── requirements.txt        # ML & Geospatial dependencies (TensorFlow, Rasterio)
    ├── .env.example            # Template for Copernicus CDSE credentials
    │
    ├── models/
    │   ├── cnn_model.keras     # Trained CNN v1 model weights (1.8 MB)
    │   ├── cnn_model_v2.keras  # Trained CNN v2 model weights (1.8 MB)
    │   ├── model_metadata_v2.json
    │   └── preprocessing_config_v2.json
    │
    ├── src/
    │   ├── surveillance.py     # Autonomous Pan-Himalayan Fleet Surveillance Engine
    │   ├── judge_pipeline.py   # Multi-evidence change detection pipeline
    │   ├── bi_temporal_change.py # NDVI drop & spectral reflectance differencing
    │   ├── satellite.py        # Copernicus CDSE STAC & Sentinel Hub client
    │   ├── terrain.py          # Topographic DEM slope modeling
    │   └── india_boundary.py   # Coordinate verification & state bounds
    │
    ├── data/
    │   ├── surveillance_cache.json # Isolated cache of Himalayan fleet matrix
    │   ├── demo_chamoli_*.npy      # Verified 2021 disaster benchmark arrays
    │   └── ground_truth/           # Regional landslide inventories
    │
    └── reports/                # Benchmark evaluation reports & demo artifacts
```

---

## ⚡ Key Features

1. **Autonomous Pan-Himalayan Satellite Surveillance:** Continuously screens 6 strategic mountain corridors across Uttarakhand, Himachal Pradesh, Jammu & Kashmir, Ladakh, Sikkim, and Arunachal Pradesh using ESA Sentinel-2 L2A optical reflectance passes.
2. **Bi-Temporal Spectral Change Detection:** Computes paired vegetative canopy loss ($\Delta\text{NDVI} \le -0.150$), bare mineral exposure, and contiguous cluster filtering to distinguish slope disturbances from seasonal variance.
3. **Deep-Learning Model Diagnostics with OOD Guardrails:** Integrates trained CNN models with automatic out-of-distribution (OOD) saturation detection, preventing uncalibrated single-image false alarms from independently triggering alerts.
4. **Verified Chamoli Disaster Benchmark:** Features an authentic forensic benchmark of the 7 February 2021 Chamoli rock-ice avalanche with paired pre- and post-disaster Sentinel-2 imagery.
5. **Physical IoT Ground Telemetry:** Ingests soil moisture (% VWC), precipitation (mm/h), inclinometer tilt (degrees), and vibration from physical hardware (ESP32/Arduino) or real-time simulation.
6. **Highway Control Room Aesthetic:** Built with a modern dark slate palette, cyan accents, responsive Plotly charts, interactive GIS mapping, and clear human-readable status banners.
7. **End-to-End Alert Management:** Complete incident lifecycle (`NEW` → `ACKNOWLEDGED` → `UNDER INVESTIGATION` → `RESOLVED`) adhering to NDMA early-warning protocols.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10 or 3.11
- Git

### 1. Clone This Repository
```powershell
git clone https://github.com/aditisingh85110-del/GeoGuard-ML-Integration.git
cd GeoGuard-ML-Integration
```

### 2. Set Up Virtual Environment & Dependencies
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install frontend and ML backend dependencies
pip install -r requirements.txt
pip install -r ml_service/requirements.txt
```

### 3. Configure Credentials (Optional)
Copy `ml_service/.env.example` to `ml_service/.env` and supply your Copernicus Data Space Ecosystem (CDSE) client credentials:
```ini
CDSE_CLIENT_ID=your_cdse_client_id_here
CDSE_CLIENT_SECRET=your_cdse_client_secret_here
```
*(Note: If omitted, the service safely utilizes cached multi-temporal satellite observations and verified benchmarks without crashing).*

---

## 🚀 Running the Services

### Terminal 1 — Start the FastAPI ML Backend
```powershell
python -m uvicorn ml_service.app:app --host 127.0.0.1 --port 8000
```
- Interactive API Docs: `http://127.0.0.1:8000/docs`
- Surveillance Summary: `http://127.0.0.1:8000/api/v2/surveillance/summary`

### Terminal 2 — Start the Streamlit Dashboard
```powershell
streamlit run app.py --server.port 8501
```
- Access the web interface at `http://localhost:8501`

---

## ⚠️ Safety & Geotechnical Disclaimer
GeoGuard AI is an advanced engineering prototype for disaster surveillance. Observed optical satellite changes measure past surface alterations and do not forecast the exact timing of future slope failures. Satellite screening does not actuate highway boom barriers or close roads without concurrent physical geotechnical ground sensor confirmation.
