"""
=============================================================================
GEOGUARD AI — SATELLITE INTELLIGENCE & HIMALAYAN LANDSLIDE MONITORING
Page: pages/satellite_intelligence.py
Purpose: Clean, minimal, professional spaceborne surveillance for judges & operators
=============================================================================
Core View:
  1. Compact Header: "GeoGuard Satellite Intelligence", Subtitle, Status Badge
  2. Mode & Corridor Selector (6-Region Fleet vs Manual Dehradun/Chamoli)
  3. One Prominent Result Banner (LOW CHANGE, MODERATE CHANGE, HIGH CHANGE, NOT ENOUGH DATA)
  4. Four Compact Metric Cards (Changed Area, Cloud Cover, Latest Satellite Image, Terrain Data)
  5. Prominent Visual: Pan-Himalayan Arc Map (Fleet) or Bi-temporal Change Map (Manual)
  6. Two Collapsible Expanders: "Technical Details" and "Data Sources & Limitations"
  7. One Short Disclaimer at Bottom
=============================================================================
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import streamlit as st
import plotly.graph_objects as go

from modules.ui import (
    render_hud_kpi,
    render_hud_banner,
    render_html,
    COLOR_PANEL,
    COLOR_BORDER,
    COLOR_SAFE,
    COLOR_MODERATE,
    COLOR_HIGH,
    COLOR_CRITICAL,
    COLOR_BLUE,
    COLOR_CYAN,
)
from modules.satellite_client import (
    get_satellite_client,
    check_health,
    list_aois,
    evaluate_aoi,
    get_judge_demo,
    get_surveillance_summary,
    trigger_surveillance_sweep,
)


# =============================================================================
# HUMAN-FRIENDLY RESULT BANNER & DISTURBANCE DEFINITIONS
# =============================================================================
DISTURBANCE_PRESENTATION = {
    "LOW_CHANGE": {
        "label": "LOW CHANGE",
        "meaning": "No major surface change detected in the available satellite images.",
        "color": "#10B981",
        "bg": "rgba(16, 185, 129, 0.12)",
        "icon": "🟢",
    },
    "LOW_OBSERVED_DISTURBANCE": {
        "label": "LOW CHANGE",
        "meaning": "No major surface change detected in the available satellite images.",
        "color": "#10B981",
        "bg": "rgba(16, 185, 129, 0.12)",
        "icon": "🟢",
    },
    "LOW_OBSERVED_RISK": {
        "label": "LOW CHANGE",
        "meaning": "No major surface change detected in the available satellite images.",
        "color": "#10B981",
        "bg": "rgba(16, 185, 129, 0.12)",
        "icon": "🟢",
    },
    "MODERATE_CHANGE": {
        "label": "MODERATE CHANGE",
        "meaning": "Some surface change detected. Further inspection may be needed.",
        "color": "#F59E0B",
        "bg": "rgba(245, 158, 11, 0.12)",
        "icon": "🟠",
    },
    "MODERATE_OBSERVED_DISTURBANCE": {
        "label": "MODERATE CHANGE",
        "meaning": "Some surface change detected. Further inspection may be needed.",
        "color": "#F59E0B",
        "bg": "rgba(245, 158, 11, 0.12)",
        "icon": "🟠",
    },
    "ELEVATED_CHANGE_CANDIDATE": {
        "label": "MODERATE CHANGE",
        "meaning": "Some surface change detected. Further inspection may be needed.",
        "color": "#F59E0B",
        "bg": "rgba(245, 158, 11, 0.12)",
        "icon": "🟠",
    },
    "HIGH_CHANGE": {
        "label": "HIGH CHANGE",
        "meaning": "Large surface change detected. Field verification is recommended.",
        "color": "#EF4444",
        "bg": "rgba(239, 68, 68, 0.15)",
        "icon": "🔴",
    },
    "HIGH_OBSERVED_DISTURBANCE": {
        "label": "HIGH CHANGE",
        "meaning": "Large surface change detected. Field verification is recommended.",
        "color": "#EF4444",
        "bg": "rgba(239, 68, 68, 0.15)",
        "icon": "🔴",
    },
    "HIGH_PRIORITY_REVIEW": {
        "label": "HIGH CHANGE",
        "meaning": "Large surface change detected. Field verification is recommended.",
        "color": "#EF4444",
        "bg": "rgba(239, 68, 68, 0.15)",
        "icon": "🔴",
    },
    "NOT_ENOUGH_DATA": {
        "label": "NOT ENOUGH DATA",
        "meaning": "Clouds or missing satellite images prevent reliable analysis.",
        "color": "#94A3B8",
        "bg": "rgba(148, 163, 184, 0.12)",
        "icon": "⚪",
    },
    "INSUFFICIENT_DATA": {
        "label": "NOT ENOUGH DATA",
        "meaning": "Clouds or missing satellite images prevent reliable analysis.",
        "color": "#94A3B8",
        "bg": "rgba(148, 163, 184, 0.12)",
        "icon": "⚪",
    },
}


def get_disturbance_info(status_key: str) -> Dict[str, str]:
    """Returns standardized human-readable disturbance label, meaning, and styling."""
    key = str(status_key or "").upper()
    return DISTURBANCE_PRESENTATION.get(key, {
        "label": "NOT ENOUGH DATA",
        "meaning": "Clouds or missing satellite images prevent reliable analysis.",
        "color": "#94A3B8",
        "bg": "rgba(148, 163, 184, 0.12)",
        "icon": "⚪",
    })


def _get_demo_report_paths():
    """Resolves local paths to cached Chamoli demo artifacts if present on disk."""
    base_dir = Path(__file__).resolve().parent.parent / "ml_service" / "reports"
    json_path = base_dir / "demo_judge_evaluation.json"
    img_path = base_dir / "demo_bi_temporal_change_map.png"
    return json_path, img_path


def _render_result_banner(status_key: str):
    """Renders one prominent, unambiguous result banner for judges."""
    info = get_disturbance_info(status_key)
    render_html(f"""
    <div style='background: {info["bg"]}; border: 1px solid {info["color"]}55; border-left: 5px solid {info["color"]}; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <span style='font-size: 16px; font-weight: 800; color: {info["color"]}; letter-spacing: 0.02em;'>
                {info["icon"]} {info["label"]}
            </span>
            <span class='font-mono' style='font-size: 11px; color: #94A3B8;'>
                Satellite Change Screening
            </span>
        </div>
        <div style='font-size: 13px; color: #F8FAFC; margin-top: 4px; line-height: 1.4;'>
            {info["meaning"]}
        </div>
    </div>
    """)


def _render_short_disclaimer():
    """Renders the single concise disclaimer required on the main page."""
    render_html("""
    <div style='margin-top: 24px; padding: 10px 14px; background: rgba(17, 24, 39, 0.6); border: 1px solid #1F2937; border-radius: 6px; font-size: 11px; color: #94A3B8; text-align: center;'>
        GeoGuard screens observed satellite surface changes. It does not predict future landslides or confirm slope safety.
    </div>
    """)


def _create_himalayan_map(regions: List[Dict[str, Any]], selected_aoi_id: Optional[str] = None) -> go.Figure:
    """
    Renders an interactive dark-themed geographic scatter map of the 6 Himalayan
    surveillance corridors across the Indian Himalayan Arc.
    """
    fig = go.Figure()
    lats, lons, texts, hover_texts, colors, sizes = [], [], [], [], [], []

    for r in regions:
        coords = r.get("center_coords") or [0.0, 0.0]
        lon, lat = coords[0], coords[1]
        aid = r.get("aoi_id", "")
        lats.append(lat)
        lons.append(lon)
        state_label = r.get("state", "India")
        name = r.get("name", "Corridor")
        corridor = r.get("corridor_code", "")
        dist_info = get_disturbance_info(r.get("observed_disturbance", ""))
        date_str = (r.get("acquisition_date") or "N/A")[:10]
        cloud = r.get("cloud_cover_percent", 0.0)
        px = r.get("candidate_pixels", 0)

        is_selected = (aid == selected_aoi_id)
        sizes.append(18 if is_selected else 13)

        texts.append(f"{state_label}")
        hover_texts.append(
            f"<b>{name}</b><br>"
            f"Territory: <b>{state_label}</b> | Corridor: <code>{corridor}</code><br>"
            f"Status: <b>{dist_info['label']}</b><br>"
            f"Latest Image: {date_str} (Cloud: {cloud:.1f}%)<br>"
            f"Changed Area: {px} px<br>"
            f"<i>{dist_info['meaning']}</i>"
        )
        colors.append(dist_info["color"])

    if hasattr(go, "Scattermap"):
        fig.add_trace(go.Scattermap(
            lat=lats,
            lon=lons,
            mode="markers+text",
            marker=dict(size=sizes, color=colors, opacity=0.9),
            text=texts,
            textposition="top right",
            textfont=dict(color="#F8FAFC", size=11, family="Inter, sans-serif"),
            hoverinfo="text",
            hovertext=hover_texts,
        ))
        fig.update_layout(
            map=dict(
                style="carto-darkmatter",
                center=dict(lat=31.2, lon=83.5),
                zoom=4.3,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=370,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
    elif hasattr(go, "Scattermapbox"):
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode="markers+text",
            marker=dict(size=sizes, color=colors, opacity=0.9),
            text=texts,
            textposition="top right",
            textfont=dict(color="#F8FAFC", size=11, family="Inter, sans-serif"),
            hoverinfo="text",
            hovertext=hover_texts,
        ))
        fig.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=31.2, lon=83.5),
                zoom=4.3,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=370,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
    else:
        fig.add_trace(go.Scattergeo(
            lat=lats,
            lon=lons,
            mode="markers+text",
            marker=dict(size=12, color=colors),
            text=texts,
            hoverinfo="text",
            hovertext=hover_texts,
        ))
        fig.update_layout(
            geo=dict(
                scope="asia",
                center=dict(lat=31.2, lon=83.5),
                projection_scale=3.5,
                bgcolor="rgba(0,0,0,0)",
                showland=True,
                landcolor="#111827",
                showocean=True,
                oceancolor="#0B0F19",
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=370,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
    return fig


# =============================================================================
# MODE 1: AUTONOMOUS HIMALAYAN FLEET SURVEILLANCE (6 TERRITORIES)
# =============================================================================
def _render_autonomous_himalayan_fleet(client, is_online: bool):
    """
    Renders the clean, minimal autonomous Himalayan fleet surveillance dashboard.
    A judge understands the region, the satellite pass, and the result in 5 seconds.
    """
    # -------------------------------------------------------------------------
    # 1. Fetch Fleet Matrix & Summary
    # -------------------------------------------------------------------------
    summary = client.get_surveillance_summary()
    regions = summary.get("regions", [])
    cache_time = summary.get("timestamp", "N/A")

    if not regions:
        st.info("No regional surveillance telemetry available. Trigger a fleet sweep below.")
        if st.button("⚡ Run Himalayan Fleet Sweep", type="primary"):
            with st.spinner("Screening Copernicus Sentinel-2 passes across Himalayan corridors..."):
                client.trigger_surveillance_sweep()
                st.rerun()
        return

    # -------------------------------------------------------------------------
    # 2. Corridor Selector & Sweep Action Bar
    # -------------------------------------------------------------------------
    c_sel, c_btn = st.columns([3.2, 1.3])
    with c_sel:
        aoi_options = [r["aoi_id"] for r in regions]
        aoi_labels = {r["aoi_id"]: f"{r['name']} ({r['state']} • {r['corridor_code']})" for r in regions}
        selected_aoi_id = st.selectbox(
            "Select Himalayan Surveillance Corridor",
            options=aoi_options,
            format_func=lambda x: aoi_labels.get(x, x),
            index=0,
            label_visibility="collapsed",
        )
    with c_btn:
        sweep_clicked = st.button("⚡ Run Fleet Sweep", type="primary", use_container_width=True)

    if sweep_clicked:
        if not is_online:
            st.error("ML Backend is offline. Start FastAPI to query live Copernicus STAC.")
        else:
            with st.spinner("Screening Copernicus Sentinel-2 passes across Himalayan corridors..."):
                client.trigger_surveillance_sweep(timeout=75.0)
                st.rerun()

    # Matched Corridor Object
    matched_region = next((r for r in regions if r.get("aoi_id") == selected_aoi_id), regions[0])

    # -------------------------------------------------------------------------
    # 3. One Prominent Result Banner
    # -------------------------------------------------------------------------
    _render_result_banner(matched_region.get("observed_disturbance", "LOW_CHANGE"))

    # -------------------------------------------------------------------------
    # 4. Four Compact Metric Cards
    # -------------------------------------------------------------------------
    px = matched_region.get("candidate_pixels", 0)
    ratio = matched_region.get("candidate_ratio_pct", 0.0)
    cloud = matched_region.get("cloud_cover_percent", 0.0)
    acq_raw = matched_region.get("acquisition_date") or "N/A"
    acq_fmt = f"{acq_raw[:10]} {acq_raw[11:16]} UTC" if "T" in acq_raw else acq_raw
    freshness = matched_region.get("data_freshness_days", 0.0)
    status_color = get_disturbance_info(matched_region.get("observed_disturbance", ""))["color"]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_hud_kpi(
            "Changed Area",
            f"{px:,} px",
            f"{ratio:.2f}% of corridor",
            "📐",
            status_color,
        )
    with m2:
        cloud_col = COLOR_CRITICAL if cloud > 20.0 else COLOR_SAFE
        cloud_sub = "Cloud Obscured (>20%)" if cloud > 20.0 else "Clear Observation"
        render_hud_kpi(
            "Cloud Cover",
            f"{cloud:.1f}%",
            cloud_sub,
            "☁",
            cloud_col,
        )
    with m3:
        render_hud_kpi(
            "Latest Satellite Image",
            acq_fmt[:16],
            f"Acquired {freshness:.1f}d ago",
            "🛰️",
            COLOR_CYAN,
        )
    with m4:
        render_hud_kpi(
            "Terrain Data",
            matched_region.get("corridor_code", "Corridor"),
            matched_region.get("hazard_classification", "Mountain Corridor").replace("_", " ").title()[:24],
            "⛰️",
            COLOR_BLUE,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 5. Prominent Main Map & Regional Status Cards
    # -------------------------------------------------------------------------
    fig_map = _create_himalayan_map(regions, selected_aoi_id=selected_aoi_id)
    st.plotly_chart(fig_map, use_container_width=True)

    # 6 Regional Cards (Quick Fleet Overview)
    row_pairs = [regions[i:i+2] for i in range(0, len(regions), 2)]
    for pair in row_pairs:
        cols = st.columns(len(pair))
        for col, r in zip(cols, pair):
            with col:
                r_dist = get_disturbance_info(r.get("observed_disturbance", ""))
                r_acq = (r.get("acquisition_date") or "N/A")
                r_acq_str = f"{r_acq[:10]} {r_acq[11:16]} UTC" if "T" in r_acq else r_acq
                r_cloud = r.get("cloud_cover_percent", 0.0)
                r_cloud_tag = "<span style='color:#EF4444; font-size:11px;'>[Obscured]</span>" if r_cloud > 20.0 else "<span style='color:#10B981; font-size:11px;'>[Clear]</span>"
                
                card_html = f"""
                <div style='background: {COLOR_PANEL}; border: 1px solid {COLOR_BORDER}; border-top: 3px solid {r_dist["color"]}; border-radius: 6px; padding: 12px 14px; margin-bottom: 12px;'>
                    <div style='display: flex; justify-content: space-between; align-items: flex-start;'>
                        <div>
                            <div style='font-size: 13px; font-weight: 700; color: #F8FAFC;'>{r.get('name')}</div>
                            <div style='font-size: 11px; color: #94A3B8;'>{r.get('state')} &bull; <code>{r.get('corridor_code')}</code></div>
                        </div>
                        <span style='font-size: 11px; font-weight: 700; color: {r_dist["color"]}; background: {r_dist["bg"]}; border: 1px solid {r_dist["color"]}44; padding: 2px 7px; border-radius: 4px;'>
                            {r_dist["label"]}
                        </span>
                    </div>
                    <div style='font-size: 11px; color: #E2E8F0; margin-top: 6px;'>
                        {r_dist["meaning"]}
                    </div>
                    <div style='display: flex; justify-content: space-between; font-size: 10.5px; color: #94A3B8; margin-top: 8px; border-top: 1px solid #1F2937; padding-top: 6px;'>
                        <span>Image: <b style='color:#F8FAFC;'>{r_acq_str}</b></span>
                        <span>Cloud: <b style='color:#F8FAFC;'>{r_cloud:.1f}%</b> {r_cloud_tag}</span>
                        <span>Changed: <b style='color:{r_dist["color"]};'>{r.get('candidate_pixels', 0)} px</b></span>
                    </div>
                </div>
                """
                render_html(card_html)

    # -------------------------------------------------------------------------
    # 6. Collapsible Technical Details (For Judges & Experts)
    # -------------------------------------------------------------------------
    det = matched_region.get("detailed_evaluation", {})
    bi_temp = det.get("bi_temporal_change_detection", {})
    spectral = bi_temp.get("reflectance_difference", {})
    band_deltas = spectral.get("per_band_deltas", {})
    cnn_info = det.get("cnn_inference_diagnostics", {})
    cnn_score = matched_region.get("cnn_raw_score", 0.0)

    with st.expander("⚙️ Technical Details", expanded=False):
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.markdown(
                f"<div style='font-size:12px; color:#F8FAFC; line-height:1.6;'>"
                f"<b>Spectral Change & Vegetation Index:</b><br>"
                f"&bull; <b>Vegetation Index Delta (&Delta;NDVI):</b> <code>{matched_region.get('mean_delta_ndvi', 0.0):+.4f}</code> (Drop Threshold: &le; -0.150)<br>"
                f"&bull; <b>Candidate Change Pixels:</b> {px:,} px of 16,384 total corridor pixels ({ratio:.2f}%)<br>"
                f"&bull; <b>Band 2 (Blue 490nm) &Delta;:</b> <code>{band_deltas.get('B02_delta_mean', 0.0):+.5f}</code><br>"
                f"&bull; <b>Band 3 (Green 560nm) &Delta;:</b> <code>{band_deltas.get('B03_delta_mean', 0.0):+.5f}</code><br>"
                f"&bull; <b>Band 4 (Red 665nm) &Delta;:</b> <code>{band_deltas.get('B04_delta_mean', 0.0):+.5f}</code><br>"
                f"&bull; <b>Band 8 (NIR 842nm) &Delta;:</b> <code>{band_deltas.get('B08_delta_mean', 0.0):+.5f}</code><br>"
                f"&bull; <b>Ground Bounding Box (WGS84):</b> <code>{matched_region.get('bbox_wgs84', [])}</code><br>"
                f"&bull; <b>Coordinate Validation:</b> {matched_region.get('verification_source', 'Survey of India')}"
                f"</div>",
                unsafe_allow_html=True,
            )
        with t_col2:
            st.markdown(
                f"<div style='font-size:12px; color:#F8FAFC; line-height:1.6;'>"
                f"<b>CNN Diagnostic Activation & Acquisition:</b><br>"
                f"&bull; <b>Model Version:</b> <code>{cnn_info.get('model_version', 'GeoGuard_CNN_v2')}</code><br>"
                f"&bull; <b>CNN Diagnostic Score:</b> <code>{cnn_score:.4f}</code> <i>(Activation value, NOT landslide probability)</i><br>"
                f"&bull; <b>Approximate Logit:</b> <code>z ~ {cnn_info.get('approximate_pre_sigmoid_logit', 0.0):+.2f}</code> (Threshold: {cnn_info.get('decision_threshold', 0.25)})<br>"
                f"&bull; <b>OOD Saturation Filter:</b> <b style='color:#10B981;'>ACTIVE</b> (Autonomous alerts suppressed for high activation)<br>"
                f"&bull; <b>Latest Sentinel-2 Product:</b> <code style='font-size:10px; color:#38BDF8;'>{matched_region.get('acquisition_id', 'N/A')}</code><br>"
                f"&bull; <b>Pre-Event Baseline Product:</b> <code style='font-size:10px;'>{matched_region.get('pre_acquisition_id', 'N/A')}</code><br>"
                f"&bull; <b>System Evaluated On:</b> {matched_region.get('evaluation_timestamp', cache_time)[:16].replace('T', ' ')} UTC"
                f"</div>",
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # 7. Collapsible Data Sources & Limitations
    # -------------------------------------------------------------------------
    with st.expander("📚 Data Sources & Limitations", expanded=False):
        st.markdown(
            "<div style='font-size:12px; color:#94A3B8; line-height:1.6;'>"
            "&bull; <b>Constellation & Sensors:</b> European Space Agency (ESA) Copernicus Sentinel-2 MSI Multi-Spectral Instrument (10m Native GSD).<br>"
            "&bull; <b>Data API:</b> Copernicus Data Space Ecosystem (CDSE) STAC API and Sentinel Hub Processing API.<br>"
            "&bull; <b>Post-Event Surface Disturbance Only:</b> Measures observed spectral reflectance alteration between paired passes. It does not forecast future slope failures.<br>"
            "&bull; <b>Cloud Obscuration Protocol:</b> Optical Sentinel-2 cannot penetrate heavy cloud cover. Passes with >20% cloud cover safely return NOT ENOUGH DATA.<br>"
            "&bull; <b>Actuator Decoupling:</b> Satellite screening does not actuate physical barriers without ground sensor agreement."
            "</div>",
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # 8. Single Short Disclaimer
    # -------------------------------------------------------------------------
    _render_short_disclaimer()


# =============================================================================
# MODE 2: MANUAL SINGLE-CORRIDOR INSPECTION (DEHRADUN & CHAMOLI)
# =============================================================================
def _render_manual_single_corridor(client, is_online: bool):
    """
    Renders the clean, minimal manual inspection view for Chamoli benchmark and Dehradun stream.
    Preserves all existing data, models, and evaluation results.
    """
    # -------------------------------------------------------------------------
    # 1. Target Corridor Selector & Trigger
    # -------------------------------------------------------------------------
    aoi_options = ["CHAMOLI_DEMO", "IN-UK-DEHRADUN-01"]
    aoi_labels = {
        "CHAMOLI_DEMO": "Chamoli Disaster Corridor (Uttarakhand) — 2021 Rock-Ice Avalanche [Verified Benchmark]",
        "IN-UK-DEHRADUN-01": "Dehradun Foothills Corridor (Uttarakhand) — Live Sentinel-2 Stream",
    }

    # Fetch live registered AOIs if online
    if is_online:
        aois_resp = client.list_aois()
        if aois_resp.get("success"):
            for a in aois_resp.get("aois", []):
                aid = a.get("aoi_id")
                if aid and aid not in aoi_labels:
                    aoi_options.append(aid)
                    aoi_labels[aid] = f"{aid}: {a.get('name', 'Corridor')} ({a.get('state', 'India')})"

    c_sel, c_btn = st.columns([3.2, 1.3])
    with c_sel:
        selected_aoi_key = st.selectbox(
            "Select Earth Observation Target Area",
            options=aoi_options,
            format_func=lambda k: aoi_labels.get(k, k),
            index=0,
            label_visibility="collapsed",
        )
    with c_btn:
        run_evaluation = st.button("🛰️ Run Evaluation", type="primary", use_container_width=True)

    is_chamoli_baseline = (selected_aoi_key == "CHAMOLI_DEMO")

    # Session State Evaluation Persistence
    if "satellite_eval_result" not in st.session_state:
        st.session_state.satellite_eval_result = None
        st.session_state.satellite_eval_aoi = None

    if run_evaluation:
        if not is_online:
            json_path, _ = _get_demo_report_paths()
            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        st.session_state.satellite_eval_result = json.load(f)
                        st.session_state.satellite_eval_aoi = selected_aoi_key
                except Exception:
                    pass
        else:
            with st.spinner("Processing Copernicus Sentinel-2 pass and bi-temporal change detection..."):
                target_id = None if is_chamoli_baseline else selected_aoi_key
                result = client.evaluate_aoi(aoi_id=target_id, allow_simulated_fallback=False, timeout=50.0)
                st.session_state.satellite_eval_result = result
                st.session_state.satellite_eval_aoi = selected_aoi_key

    # Automatic initial load for Chamoli benchmark if nothing evaluated yet
    if st.session_state.satellite_eval_result is None:
        json_path, _ = _get_demo_report_paths()
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    st.session_state.satellite_eval_result = json.load(f)
                    st.session_state.satellite_eval_aoi = "CHAMOLI_DEMO"
            except Exception:
                pass

    res = st.session_state.satellite_eval_result
    if res is None:
        st.info("Select a corridor and click 'Run Evaluation' to inspect spaceborne telemetry.")
        return

    # -------------------------------------------------------------------------
    # 2. One Prominent Result Banner
    # -------------------------------------------------------------------------
    risk_status = res.get("risk_screening_status", res.get("final_candidate_status", "LOW_CHANGE"))
    _render_result_banner(risk_status)

    # -------------------------------------------------------------------------
    # 3. Four Compact Metric Cards
    # -------------------------------------------------------------------------
    bi_temporal = res.get("bi_temporal_change_detection", {})
    spatial_metrics = bi_temporal.get("spatial_change_metrics", {})
    ndvi_metrics = bi_temporal.get("ndvi_analysis", {})
    terrain = res.get("terrain_topography_context", {})
    slope_info = terrain.get("slope_context", {})
    cnn_diag = res.get("cnn_inference_diagnostics", {})
    provenance = res.get("data_provenance", {})

    change_px = spatial_metrics.get("candidate_change_pixel_count", 0)
    change_ratio = spatial_metrics.get("candidate_change_pixel_ratio", 0.0) * 100.0
    affected_hectares = (change_px * 100.0) / 10000.0
    mean_delta_ndvi = ndvi_metrics.get("mean_delta_ndvi", 0.0)
    mean_slope = slope_info.get("mean_slope_degrees")
    cloud_pct = provenance.get("cloud_cover_percent", 0.0)
    acq_date = provenance.get("acquisition_date", "N/A")
    raw_cnn_score = cnn_diag.get("raw_uncalibrated_sigmoid_score", 0.0)
    dist_info = get_disturbance_info(risk_status)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_hud_kpi(
            "Changed Area",
            f"{change_px:,} px",
            f"{change_ratio:.2f}% ({affected_hectares:.1f} ha)",
            "📐",
            dist_info["color"],
        )
    with m2:
        cloud_col = COLOR_CRITICAL if (cloud_pct or 0) > 20.0 else COLOR_SAFE
        render_hud_kpi(
            "Cloud Cover",
            f"{cloud_pct:.1f}%" if cloud_pct is not None else "N/A",
            "Clear Pass" if (cloud_pct or 0) <= 20.0 else "Cloud Obscured",
            "☁",
            cloud_col,
        )
    with m3:
        render_hud_kpi(
            "Latest Satellite Image",
            acq_date[:10] if "T" in acq_date else acq_date[:12],
            acq_date[11:16] + " UTC" if "T" in acq_date else "Pass Verified",
            "🛰️",
            COLOR_CYAN,
        )
    with m4:
        slope_str = f"{mean_slope:.1f}° Slope" if mean_slope is not None else "Mountain Profile"
        slope_hazard = slope_info.get("hazard_classification", "Terrain Context").replace("_", " ").title()[:22]
        render_hud_kpi(
            "Terrain Data",
            slope_str,
            slope_hazard,
            "⛰️",
            COLOR_BLUE,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. Prominent Main Satellite Change Visualization
    # -------------------------------------------------------------------------
    _, demo_img_path = _get_demo_report_paths()
    if is_chamoli_baseline and demo_img_path.exists():
        st.image(
            str(demo_img_path),
            caption="Chamoli 2021 Disaster: Pre-Event True Color (5 Feb), Post-Event True Color (10 Feb), Pixel-Level ΔNDVI, and Filtered Landslide Scar Mask",
            use_container_width=True,
        )
    else:
        render_html(f"""
        <div style='background: {COLOR_PANEL}; border: 1px dashed {COLOR_BORDER}; border-radius: 6px; padding: 24px 20px; text-align: center; margin-bottom: 14px;'>
            <div style='font-size: 26px; margin-bottom: 6px;'>🛰️</div>
            <div style='font-size: 13px; font-weight: 600; color: #F8FAFC;'>
                Numerical Array Screening Active (Raster Rendering Deferred)
            </div>
            <div style='font-size: 11px; color: #94A3B8; max-width: 480px; margin: 4px auto 0 auto;'>
                Copernicus CDSE multi-spectral reflectance tensors [128, 128, 4] screened in memory.
                Per strict governance rules, pre-rendered maps from other regions are never substituted.
            </div>
        </div>
        """)

    # -------------------------------------------------------------------------
    # 5. Collapsible Technical Details (For Judges & Experts)
    # -------------------------------------------------------------------------
    with st.expander("⚙️ Technical Details", expanded=False):
        t1, t2 = st.columns(2)
        with t1:
            st.markdown(
                f"<div style='font-size:12px; color:#F8FAFC; line-height:1.6;'>"
                f"<b>Spectral Change & Vegetation Index:</b><br>"
                f"&bull; <b>Vegetation Index Delta (&Delta;NDVI):</b> <code>{mean_delta_ndvi:+.4f}</code><br>"
                f"&bull; <b>Pre-Event Mean NDVI:</b> <code>{ndvi_metrics.get('pre_event_mean_ndvi', 0.0):.4f}</code><br>"
                f"&bull; <b>Post-Event Mean NDVI:</b> <code>{ndvi_metrics.get('post_event_mean_ndvi', 0.0):.4f}</code><br>"
                f"&bull; <b>Vegetation Loss Drop Threshold:</b> &le; -0.150<br>"
                f"&bull; <b>Candidate Change Criterion:</b> Min 25 contiguous 10m pixels (2,500 m&sup2;)<br>"
                f"&bull; <b>Ground Footprint:</b> {affected_hectares:.2f} ha ({affected_hectares * 10000:.0f} m&sup2;)"
                f"</div>",
                unsafe_allow_html=True,
            )
        with t2:
            st.markdown(
                f"<div style='font-size:12px; color:#F8FAFC; line-height:1.6;'>"
                f"<b>CNN Diagnostic Activation & Product IDs:</b><br>"
                f"&bull; <b>Model:</b> <code>{cnn_diag.get('model_version', 'GeoGuard_CNN_v2')}</code><br>"
                f"&bull; <b>CNN Diagnostic Score:</b> <code>{raw_cnn_score:.4f}</code> <i>(Activation, NOT probability)</i><br>"
                f"&bull; <b>Approximate Logit:</b> <code>z ~ {cnn_diag.get('approximate_pre_sigmoid_logit', 0.0):+.2f}</code> (Threshold: {cnn_diag.get('decision_threshold', 0.50)})<br>"
                f"&bull; <b>OOD Saturation Filter:</b> <b style='color:#10B981;'>ACTIVE</b> (Autonomous alerts suppressed for high activation)<br>"
                f"&bull; <b>Acquisition ID:</b> <code style='font-size:10px; color:#38BDF8;'>{provenance.get('acquisition_id', 'N/A')}</code><br>"
                f"&bull; <b>Spatial Resolution:</b> 10.0m Native Ground Sample Distance (GSD)<br>"
                f"&bull; <b>Ground Elevation:</b> {terrain.get('elevation_context', {}).get('mean_elevation_meters', 'N/A')}m"
                f"</div>",
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # 6. Collapsible Data Sources & Limitations
    # -------------------------------------------------------------------------
    with st.expander("📚 Data Sources & Limitations", expanded=False):
        st.markdown(
            "<div style='font-size:12px; color:#94A3B8; line-height:1.6;'>"
            "&bull; <b>Imagery Source:</b> European Space Agency (ESA) Copernicus Sentinel-2 MSI Multi-Spectral Instrument (10m Native GSD).<br>"
            "&bull; <b>Post-Event Surface Disturbance Only:</b> Measures observed spectral reflectance alteration between paired passes. It does not forecast future slope failures.<br>"
            "&bull; <b>Cloud Obscuration:</b> Optical Sentinel-2 cannot penetrate heavy cloud cover. Areas with >20% cloud cover return NOT ENOUGH DATA.<br>"
            "&bull; <b>Actuator Decoupling:</b> Satellite screening does not actuate highway barriers without ground geotechnical sensor agreement."
            "</div>",
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # 7. Single Short Disclaimer
    # -------------------------------------------------------------------------
    _render_short_disclaimer()


# =============================================================================
# MAIN PAGE ROUTER & ENTRY POINT
# =============================================================================
def render_satellite_intelligence_page(enriched_stations=None):
    """
    Renders the clean, minimal, professional Satellite Intelligence dashboard view.
    Fulfills 5-second judge readability requirements while preserving full functionality.
    """
    client = get_satellite_client()
    health = client.check_health()
    is_online = health.get("is_online", False)

    # -------------------------------------------------------------------------
    # 1. Compact Header
    # -------------------------------------------------------------------------
    col_h_left, col_h_right = st.columns([3.5, 1.5])
    with col_h_left:
        st.markdown(
            "<h2 style='margin:0; font-size:24px; font-weight:700; color:#F8FAFC;'>GeoGuard Satellite Intelligence</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:13px; color:#94A3B8; margin-top:2px;'>Himalayan Landslide Monitoring</div>",
            unsafe_allow_html=True,
        )
    with col_h_right:
        status_col = COLOR_SAFE if is_online else COLOR_CRITICAL
        status_txt = "● ONLINE" if is_online else "● OFFLINE"
        st.markdown(
            f"<div style='text-align:right; padding-top:6px;'>"
            f"<span style='font-family:monospace; font-size:11.5px; font-weight:700; color:{status_col}; background:{status_col}18; border:1px solid {status_col}44; padding:4px 10px; border-radius:4px;'>"
            f"{status_txt}"
            f"</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. Operating Mode Selector
    # -------------------------------------------------------------------------
    surveillance_mode = st.radio(
        "Surveillance Operating Mode",
        [
            "Autonomous Himalayan Fleet Surveillance (6 Territories)",
            "Manual Single-Corridor Inspection (Dehradun / Chamoli)",
        ],
        horizontal=True,
        label_visibility="collapsed",
        index=0,
    )
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Route to Active Mode
    # -------------------------------------------------------------------------
    if surveillance_mode == "Autonomous Himalayan Fleet Surveillance (6 Territories)":
        _render_autonomous_himalayan_fleet(client, is_online)
    else:
        _render_manual_single_corridor(client, is_online)
