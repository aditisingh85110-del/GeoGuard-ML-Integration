# GeoGuard API Integration Guide for Frontend Teammates

This guide provides everything the frontend website and mobile app teammates need to display satellite-based landslide detection data in GeoGuard without modifying the ML service.

---

## 1. Quick Connection Summary

- **Base URL**: `http://localhost:8000` (or `http://<server-ip>:8000`)
- **Interactive Documentation (Swagger UI)**: `http://localhost:8000/docs`
- **Alternative Docs (ReDoc)**: `http://localhost:8000/redoc`
- **CORS Support**: Enabled on all endpoints (`Access-Control-Allow-Origin: *`).
- **Data Format**: Standard JSON for all responses.

---

## 2. API Endpoints Specification

### 2.1 Service Health Check
Verify the ML microservice, CNN model, and database are operational.

- **Method**: `GET`
- **Path**: `/health`
- **cURL**:
  ```bash
  curl -X GET http://localhost:8000/health
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "status": "healthy",
    "service": "GeoGuard Satellite Landslide Detection ML Service",
    "model_loaded": true,
    "model_version": "GeoGuard_CNN_v1",
    "database_connected": true,
    "timestamp": "2026-11-01T01:40:00.000000Z"
  }
  ```

---

### 2.2 Latest Corridor Macro Prediction
The primary endpoint for your main dashboard overview card. Returns the most recent macro landslide prediction for the monitored corridor.

- **Method**: `GET`
- **Path**: `/latest-prediction`
- **cURL**:
  ```bash
  curl -X GET http://localhost:8000/latest-prediction
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "id": 1,
    "prediction": "LANDSLIDE DETECTED",
    "probability": 0.895,
    "decision_threshold": 0.40,
    "acquisition_id": "S2B_MSIL2A_20261029T050929_N0513_R019_T43PGP_20261029T105708",
    "acquisition_timestamp": "2026-10-29T05:09:29.024000Z",
    "processing_timestamp": "2026-11-01T01:30:00.000000Z",
    "cloud_cover_percentage": 9.25,
    "aoi_name": "TEST_DEVELOPMENT_AOI",
    "aoi_bbox": [76.95, 12.00, 77.05, 12.10],
    "data_status": "VALID",
    "is_simulated": false,
    "model_version": "GeoGuard_CNN_v1",
    "total_patches": 2,
    "landslide_patches": 1,
    "high_risk_coordinates": [
      {
        "lat": 12.05,
        "lon": 77.00,
        "prob": 0.895
      }
    ],
    "message": "Macro landslide risk detected across 1 of 2 evaluated terrain patches."
  }
  ```

> [!TIP]
> **Dashboard Display Guidelines**:
> - If `prediction === "LANDSLIDE DETECTED"`, display a red hazard badge.
> - If `prediction === "NORMAL / STABLE"`, display a green stable badge.
> - Display `acquisition_timestamp` as the true date of the satellite flyover (e.g., "Captured by Sentinel-2 on Oct 29, 2026").
> - If `is_simulated === true`, render a small "DEV TEST SIMULATION" tag.

---

### 2.3 Historical Predictions Log
Used for populating historical tables, trend charts, and audit logs.

- **Method**: `GET`
- **Path**: `/predictions`
- **Query Parameters**:
  - `limit` (optional, default `50`): Max number of records to return.
  - `offset` (optional, default `0`): Pagination offset.
- **cURL**:
  ```bash
  curl -X GET "http://localhost:8000/predictions?limit=10&offset=0"
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "total_count": 14,
    "limit": 10,
    "offset": 0,
    "predictions": [
      {
        "id": 14,
        "prediction": "NORMAL / STABLE",
        "probability": 0.124,
        "decision_threshold": 0.40,
        "acquisition_id": "S2B_MSIL2A_20261029T050929_...",
        "acquisition_timestamp": "2026-10-29T05:09:29Z",
        "processing_timestamp": "2026-11-01T01:30:00Z",
        "cloud_cover_percentage": 9.25,
        "aoi_name": "TEST_DEVELOPMENT_AOI",
        "data_status": "VALID",
        "is_simulated": false,
        "total_patches": 2,
        "landslide_patches": 0,
        "high_risk_coordinates": [],
        "message": "Monitored corridor is normal and stable across all 2 evaluated terrain patches."
      }
    ]
  }
  ```

---

### 2.4 Copernicus Satellite Connection & Catalogue Status
Displays connection health to the European Space Agency CDSE STAC catalogue and indicates whether a newly acquired pass is available.

- **Method**: `GET`
- **Path**: `/satellite-status`
- **cURL**:
  ```bash
  curl -X GET http://localhost:8000/satellite-status
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "connection_status": "CONNECTED",
    "stac_catalogue_url": "https://stac.dataspace.copernicus.eu/v1/search",
    "credentials_configured": false,
    "last_checked_timestamp": "2026-11-01T01:44:44.253000Z",
    "latest_catalog_acquisition_id": "S2B_MSIL2A_20261029T050929_N0513_R019_T43PGP_20261029T105708",
    "latest_processed_acquisition_id": "S2B_MSIL2A_20261029T050929_N0513_R019_T43PGP_20261029T105708",
    "has_new_acquisition": false,
    "mode": "SIMULATED",
    "monitored_aoi": {
      "name": "TEST_DEVELOPMENT_AOI",
      "bbox": [76.95, 12.00, 77.05, 12.10],
      "is_test_aoi": true
    },
    "polling_interval_minutes": 15
  }
  ```

---

### 2.5 Trigger On-Demand Satellite Check
Allows dashboard users to click a "Check for Satellite Updates" button without waiting for the 15-minute scheduler cycle.

- **Method**: `POST`
- **Path**: `/trigger-satellite-check`
- **cURL**:
  ```bash
  curl -X POST http://localhost:8000/trigger-satellite-check
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "status": "completed",
    "action": "none",
    "message": "Acquisition S2B_MSIL2A_... is already processed. Corridor status is current."
  }
  ```

---

### 2.6 IoT Cross-Validation & Sensor Fusion
Cross-validates live geotechnical IoT telemetry (volumetric soil moisture, 24h rainfall, vibration, and inclinometer tilt) against the latest macro satellite observation.

- **Method**: `POST`
- **Path**: `/iot-fusion-cross-validate`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "sensor_id": "GEO-NODE-01",
    "soil_moisture_percent": 45.2,
    "rainfall_mm_24h": 68.0,
    "vibration_intensity": 0.18,
    "tilt_angle_deg": 2.8,
    "battery_percent": 94.0
  }
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "cross_validation_status": "AGREEMENT",
    "combined_risk_level": "CRITICAL",
    "satellite_macro_status": {
      "prediction": "LANDSLIDE DETECTED",
      "probability": 0.895,
      "acquisition_id": "S2B_MSIL2A_20261029T050929_...",
      "acquisition_timestamp": "2026-10-29T05:09:29Z",
      "data_status": "VALID"
    },
    "iot_micro_telemetry": {
      "sensor_id": "GEO-NODE-01",
      "timestamp": "2026-11-01T01:40:00Z",
      "soil_moisture_percent": 45.2,
      "rainfall_mm_24h": 68.0,
      "vibration_intensity": 0.18,
      "tilt_angle_deg": 2.8,
      "battery_percent": 94.0
    },
    "action_recommendation": "CRITICAL EMERGENCY: Visual satellite detection confirmed by active geotechnical sensor displacement. Immediate highway closure and barrier deployment required.",
    "fusion_rationale": "Macro satellite CNN detected terrain scarp/movement with 89.5% confidence, corroborated by high geotechnical telemetry (Tilt: 2.8°, Rain: 68.0mm, Moisture: 45.2%)."
  }
  ```

---

## 3. Sensor Fusion Decision Matrix

| Satellite CNN Prediction | Geotechnical IoT Telemetry | Combined Risk Level | Status Badge | Operational Action |
|:---|:---|:---|:---|:---|
| **LANDSLIDE DETECTED** | **CRITICAL** (Tilt $\ge 2^\circ$, Moisture $\ge 40\%$, Rain $\ge 50$mm) | **CRITICAL** | `CRITICAL_EMERGENCY` | Immediate highway closure, siren activation, and emergency personnel dispatch. |
| **LANDSLIDE DETECTED** | **NORMAL** (Sub-threshold tilt, moisture, and rain) | **HIGH** | `MACRO_ALERT_ONLY` | Visual satellite scarp observed outside sensor array. Deploy reconnaissance drone to coordinates. |
| **NORMAL / STABLE** | **CRITICAL** (High tilt displacement or pore-water saturation) | **WARNING** | `SUBSURFACE_PRECURSOR` | Subsurface shear plane actively moving prior to surface scarp failure. Impose cautionary speed limit. |
| **NORMAL / STABLE** | **NORMAL** (Stable terrain, low water table) | **LOW** | `ALL_CLEAR` | Harmonious stability across optical and geotechnical sensors. Maintain routine monitoring. |

---

## 4. Frontend Integration Examples (React / React Native)

### 4.1 Custom React Hook for Corridor Landslide Monitoring
```javascript
import { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:8000';

export function useSatelliteLandslideData(pollIntervalMs = 30000) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchLatest = async () => {
    try {
      const response = await fetch(`${API_BASE}/latest-prediction`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: Failed to fetch prediction`);
      }
      const json = await response.json();
      setData(json);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLatest();
    const interval = setInterval(fetchLatest, pollIntervalMs);
    return () => clearInterval(interval);
  }, [pollIntervalMs]);

  return { data, loading, error, refetch: fetchLatest };
}
```

### 4.2 React / React Native Component Example
```jsx
import React from 'react';
import { View, Text, StyleSheet, ActivityIndicator } from 'react-native';
import { useSatelliteLandslideData } from './useSatelliteLandslideData';

export function SatelliteLandslideCard() {
  const { data, loading, error } = useSatelliteLandslideData();

  if (loading) return <ActivityIndicator size="small" color="#0066cc" />;
  if (error) return <Text style={styles.errorText}>Satellite Stream Offline: {error}</Text>;
  if (!data) return null;

  const isLandslide = data.prediction === 'LANDSLIDE DETECTED';

  return (
    <View style={[styles.card, isLandslide ? styles.alertCard : styles.safeCard]}>
      <Text style={styles.title}>Sentinel-2 Satellite Verification</Text>
      
      <View style={styles.badgeRow}>
        <Text style={[styles.badge, isLandslide ? styles.alertBadge : styles.safeBadge]}>
          {data.prediction}
        </Text>
        <Text style={styles.probText}>Confidence: {(data.probability * 100).toFixed(1)}%</Text>
      </View>

      <Text style={styles.detailText}>Corridor: {data.aoi_name}</Text>
      <Text style={styles.detailText}>
        Acquisition Date: {new Date(data.acquisition_timestamp).toLocaleDateString()}
      </Text>
      
      {isLandslide && data.high_risk_coordinates.length > 0 && (
        <Text style={styles.coordText}>
          Primary Hotspot: {data.high_risk_coordinates[0].lat.toFixed(4)},{' '}
          {data.high_risk_coordinates[0].lon.toFixed(4)}
        </Text>
      )}

      {data.is_simulated && (
        <Text style={styles.simText}>Mode: Development Simulation</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  card: { padding: 16, borderRadius: 8, marginVertical: 8 },
  alertCard: { backgroundColor: '#ffebee', borderColor: '#e53935', borderWidth: 1 },
  safeCard: { backgroundColor: '#e8f5e9', borderColor: '#43a047', borderWidth: 1 },
  title: { fontSize: 16, fontWeight: 'bold', marginBottom: 8 },
  badgeRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 6 },
  badge: { paddingHorizontal: 8, paddingVertical: 4, borderRadius: 4, color: '#fff', fontWeight: 'bold' },
  alertBadge: { backgroundColor: '#d32f2f' },
  safeBadge: { backgroundColor: '#2e7d32' },
  probText: { marginLeft: 10, fontSize: 13, color: '#555' },
  detailText: { fontSize: 12, color: '#444', marginTop: 2 },
  coordText: { fontSize: 12, color: '#b71c1c', fontWeight: 'bold', marginTop: 4 },
  simText: { fontSize: 10, color: '#888', fontStyle: 'italic', marginTop: 4 },
  errorText: { color: '#c62828', fontSize: 12 },
});
```
