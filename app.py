"""Streamlit Dashboard for the AI Smart Surveillance System."""

import datetime
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import pandas as pd
import streamlit as st

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Page Configuration
st.set_page_config(
    page_title="AI Smart Surveillance System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Verify PyTorch and Ultralytics dependencies
try:
    import torch
    from ultralytics import YOLO
    from src.database import DatabaseManager
    from src.pipeline import SurveillancePipeline
    from src.utils import load_config
    ENV_READY = True
except ImportError as err:
    ENV_READY = False
    MISSING_ERR = str(err)

# Custom CSS for dark-themed surveillance UI
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e2124;
        border: 1px solid #2e3440;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .stAlert {
        border-radius: 8px;
    }
    .css-1r6slb0 {
        padding: 1rem 1rem 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_default_config() -> Dict[str, Any]:
    """Loads base configuration."""
    return load_config("config.yaml")


def main() -> None:
    st.title("🛡️ AI Smart Surveillance System")
    st.markdown("Real-time AI Video Analytics • YOLOv8 Object Tracking • Suspicious Behavior Alerting")

    if not ENV_READY:
        st.error(f"⚠️ Missing Dependency in Python {sys.version.split()[0]}: `{MISSING_ERR}`")
        st.warning(
            f"""
            ### 💡 How to launch using the configured Python 3.10 environment:

            Your terminal launched Streamlit using Python `{sys.version.split()[0]}`, but **PyTorch, YOLOv8, and ByteTrack** are installed in **Python 3.10** on this machine.

            #### Option 1: Run with Python 3.10 directly in your terminal:
            ```bash
            py -3.10 -m streamlit run app.py
            ```

            #### Option 2: Double-click the one-click launcher file in the project folder:
            ```
            run_dashboard.bat
            ```
            """
        )
        return

    config = get_default_config()
    db_path = config.get("database", {}).get("db_path", "surveillance.db")
    db = DatabaseManager(db_path=db_path)

    # =========================================================================
    # SIDEBAR CONTROLS
    # =========================================================================
    with st.sidebar:
        st.header("⚙️ Surveillance Controls")

        # Video Source
        source_type = st.radio("Video Source", ["Sample Video / File", "Webcam", "RTSP / IP Camera"])
        source_val = 0

        if source_type == "Webcam":
            cam_idx = st.number_input("Webcam Device Index", min_value=0, max_value=5, value=0)
            source_val = int(cam_idx)
        elif source_type == "Sample Video / File":
            default_sample = "samples/sample_surveillance.mp4"
            if not os.path.exists(default_sample):
                default_sample = ""
            video_path = st.text_input("Video File Path", value=default_sample, help="Path to local mp4/avi video file")
            uploaded_file = st.file_uploader("Or Upload Video File", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_file is not None:
                temp_dir = Path("evidence/uploads")
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                source_val = str(temp_path)
            else:
                source_val = video_path if video_path else 0
        else:
            rtsp_url = st.text_input("RTSP Stream URL", value="rtsp://127.0.0.1:8554/live")
            source_val = rtsp_url

        st.divider()

        # Model & Inference Parameters
        st.subheader("Model & Detection")
        model_name = st.selectbox("YOLOv8 Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
        conf_thresh = st.slider("Confidence Threshold", 0.10, 0.95, float(config["detection"]["confidence_threshold"]), 0.05)
        iou_thresh = st.slider("NMS IoU Threshold", 0.10, 0.90, float(config["detection"]["iou_threshold"]), 0.05)
        tracker_type = st.selectbox("Tracking Algorithm", ["bytetrack", "centroid"], index=0)

        st.divider()

        # Suspicious Activity Rules
        st.subheader("Suspicious Activity Rules")
        rule_intrusion = st.checkbox("Restricted Zone Intrusion", value=config["rules"]["zone_intrusion"]["enabled"])
        rule_loitering = st.checkbox("Loitering Detection", value=config["rules"]["loitering"]["enabled"])
        rule_abandoned = st.checkbox("Abandoned Object Detection", value=config["rules"]["abandoned_object"]["enabled"])
        rule_crowding = st.checkbox("Crowd Surge / Overcrowding", value=config["rules"]["crowding"]["enabled"])
        rule_after_hours = st.checkbox("After-Hours Intrusion", value=config["rules"]["after_hours"]["enabled"])

        st.divider()

        # Pipeline Session State Controls
        if "pipeline_running" not in st.session_state:
            st.session_state.pipeline_running = False

        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("▶️ Start Stream", use_container_width=True, type="primary"):
                st.session_state.pipeline_running = True
        with col_stop:
            if st.button("⏹️ Stop Stream", use_container_width=True):
                st.session_state.pipeline_running = False

    # Apply sidebar overrides to configuration
    config["detection"]["model_path"] = model_name
    config["detection"]["confidence_threshold"] = conf_thresh
    config["detection"]["iou_threshold"] = iou_thresh
    config["tracking"]["tracker_type"] = tracker_type
    config["rules"]["zone_intrusion"]["enabled"] = rule_intrusion
    config["rules"]["loitering"]["enabled"] = rule_loitering
    config["rules"]["abandoned_object"]["enabled"] = rule_abandoned
    config["rules"]["crowding"]["enabled"] = rule_crowding
    config["rules"]["after_hours"]["enabled"] = rule_after_hours

    # =========================================================================
    # TABS INTERFACE
    # =========================================================================
    tab_live, tab_alerts, tab_config = st.tabs(["🎥 Live Surveillance", "🚨 Security Alerts Log", "⚙️ Configuration"])

    # -------------------------------------------------------------------------
    # TAB 1: LIVE MONITORING
    # -------------------------------------------------------------------------
    with tab_live:
        # Metrics Row
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        fps_metric = m_col1.empty()
        latency_metric = m_col2.empty()
        tracks_metric = m_col3.empty()
        alerts_metric = m_col4.empty()

        # Initial metric display
        fps_metric.metric("Pipeline FPS", "0.0")
        latency_metric.metric("Inference Latency", "0.0 ms")
        tracks_metric.metric("Active Tracked Entities", "0")
        alerts_metric.metric("Total Recorded Alerts", str(db.get_total_count()))

        video_placeholder = st.empty()
        banner_placeholder = st.empty()

        if st.session_state.pipeline_running:
            try:
                pipeline = SurveillancePipeline(
                    config=config,
                    override_source=source_val,
                )
                pipeline.start_capture()

                while st.session_state.pipeline_running:
                    success, frame, ts = pipeline.capture.read(timeout=1.0)
                    if not success or frame is None:
                        if pipeline.capture and pipeline.capture._is_file:
                            st.info("End of video stream reached.")
                            st.session_state.pipeline_running = False
                            break
                        continue

                    annotated, tracks, triggered_alerts, fps, inf_ms = pipeline.process_frame(frame, timestamp=ts)

                    # Update live video view (convert BGR to RGB for Streamlit)
                    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                    # Update metrics
                    fps_metric.metric("Pipeline FPS", f"{fps:.1f}")
                    latency_metric.metric("Inference Latency", f"{inf_ms:.1f} ms")
                    tracks_metric.metric("Active Tracked Entities", str(len(tracks)))
                    alerts_metric.metric("Total Recorded Alerts", str(pipeline._total_alerts_count))

                    # Display banner alert if recently triggered
                    if triggered_alerts:
                        for a in triggered_alerts:
                            banner_placeholder.error(f"🚨 **ALERT ({a.severity})**: {a.message}")

            except Exception as e:
                st.error(f"Error during surveillance execution: {e}")
                st.session_state.pipeline_running = False
            finally:
                if "pipeline" in locals():
                    pipeline.stop()
        else:
            video_placeholder.info("Stream is stopped. Configure settings in the sidebar and click **Start Stream**.")

    # -------------------------------------------------------------------------
    # TAB 2: ALERTS LOG & EVIDENCE
    # -------------------------------------------------------------------------
    with tab_alerts:
        st.subheader("Recorded Security Alerts & Evidence Vault")

        # Filters
        f_col1, f_col2, f_col3 = st.columns([2, 2, 1])
        with f_col1:
            all_rules = [
                "All Rules",
                "Restricted Zone Intrusion",
                "Loitering Detected",
                "Abandoned Object",
                "Crowd Surge / Overcrowding",
                "After-Hours Intrusion",
            ]
            selected_rule_filter = st.selectbox("Filter by Rule", all_rules)
        with f_col2:
            limit_count = st.slider("Max Records to Show", 10, 500, 50, 10)
        with f_col3:
            st.write("")
            st.write("")
            rule_arg = None if selected_rule_filter == "All Rules" else selected_rule_filter
            csv_path = Path("evidence/alerts_export.csv")
            if st.button("📥 Export CSV", use_container_width=True):
                db.export_to_csv(csv_path, rule_name=rule_arg)
                if csv_path.is_file():
                    with open(csv_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download CSV File",
                            data=f,
                            file_name="surveillance_alerts.csv",
                            mime="text/csv",
                        )

        # Query Database
        alerts_data = db.get_alerts(rule_name=rule_arg, limit=limit_count)

        if not alerts_data:
            st.info("No security alerts matching criteria have been recorded.")
        else:
            # Table View
            df = pd.DataFrame(alerts_data)
            display_cols = ["id", "datetime_str", "rule_name", "severity", "object_class", "track_id", "confidence", "message"]
            st.dataframe(df[display_cols], use_container_width=True)

            st.divider()
            st.subheader("Evidence Inspection")

            # Detail view for selected alert
            alert_ids = [a["id"] for a in alerts_data]
            selected_id = st.selectbox("Select Alert ID to Inspect Evidence", alert_ids)

            selected_record = next((a for a in alerts_data if a["id"] == selected_id), None)
            if selected_record:
                ev_col1, ev_col2 = st.columns(2)
                with ev_col1:
                    st.markdown("#### Snapshot Evidence")
                    snap_path = selected_record.get("snapshot_path")
                    if snap_path and os.path.exists(snap_path):
                        st.image(snap_path, caption=f"Snapshot: {os.path.basename(snap_path)}", use_container_width=True)
                    else:
                        st.warning("No snapshot available for this event.")

                with ev_col2:
                    st.markdown("#### Video Clip Evidence (~5s Pre & Post)")
                    clip_path = selected_record.get("clip_path")
                    if clip_path and os.path.exists(clip_path):
                        st.video(clip_path)
                    else:
                        st.info("Video clip recording in progress or unavailable.")

    # -------------------------------------------------------------------------
    # TAB 3: CONFIGURATION INSPECTOR
    # -------------------------------------------------------------------------
    with tab_config:
        st.subheader("System Configuration (config.yaml)")
        st.json(config)


if __name__ == "__main__":
    main()
