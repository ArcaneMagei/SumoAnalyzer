"""Robot Sumo Analyzer - tracking-focused MVP."""

from datetime import datetime
import tempfile

import cv2
import pandas as pd
import streamlit as st

from models.dohyo import DohyoDetector
from tracking.tracker import RobotTracker


st.set_page_config(page_title="Robot Sumo Analyzer", page_icon="🤖", layout="wide")

if "processed_video" not in st.session_state:
    st.session_state.processed_video = None
if "track_rows" not in st.session_state:
    st.session_state.track_rows = []

st.title("🤖 Robot Sumo Match Analyzer")
st.caption("Current scope: per-frame dohyo tracking + AI-assisted two-robot tracking + annotated output video.")

with st.sidebar:
    st.header("⚙️ Processing")
    min_confidence = st.slider("Minimum confidence to display", 0.0, 1.0, 0.25, 0.05)
    use_ai_detector = st.checkbox("Use AI robot detector (recommended)", value=True)
    ai_weights_path = st.text_input("AI weights path (.pt)", value="models/weights/robot_sumo.pt")
    ai_confidence = st.slider("AI confidence", 0.05, 0.95, 0.20, 0.05)

uploaded_file = st.file_uploader("Choose match video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name

    cap = cv2.VideoCapture(input_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    c1, c2 = st.columns(2)
    with c1:
        st.video(uploaded_file)
    with c2:
        st.metric("Resolution", f"{width}x{height}")
        st.metric("FPS", f"{fps:.2f}")
        st.metric("Frames", total_frames)

    if st.button("🚀 Process and export annotated video", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        cap = cv2.VideoCapture(input_path)
        ok, first = cap.read()
        if not ok:
            st.error("Could not read video.")
            cap.release()
        else:
            status.text("Detecting and tracking dohyo...")
            dohyo = DohyoDetector()
            center, radius = dohyo.detect(first)
            if center is None or radius is None:
                st.error("Could not detect dohyo edges.")
                cap.release()
            else:
                tracker = RobotTracker(
                    ai_weights_path=ai_weights_path if use_ai_detector else None,
                    ai_confidence=ai_confidence,
                )
                if use_ai_detector:
                    if tracker.ai_enabled:
                        st.info(f"AI detector loaded: {ai_weights_path}")
                    else:
                        st.warning("AI detector unavailable (missing weights/deps). Falling back to classical CV. See docs/AI_WEIGHTS_SETUP.md for full setup.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                out_path = tempfile.NamedTemporaryFile(delete=False, suffix="_annotated.mp4").name
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 30.0, (width, height))

                rows = []
                frame_idx = 0
                preview_slot = st.empty()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Update dohyo per-frame to handle camera movement and perspective changes.
                    d_center, d_radius = dohyo.track(frame)
                    if d_center is None or d_radius is None or dohyo.last_detection_info is None:
                        annotated = frame.copy()
                        cv2.putText(annotated, "DOHYO LOST", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        writer.write(annotated)
                        frame_idx += 1
                        continue

                    tracks = tracker.update(frame, dohyo.last_detection_info)

                    # Draw dohyo references first: outer edge, white border inner edge, center cross
                    annotated = dohyo.draw_overlay(frame)
                    # Draw robots
                    tracks_vis = tracker.draw_tracks(annotated)
                    annotated = tracks_vis

                    # De-emphasize low-confidence boxes in final export.
                    if min_confidence > 0:
                        for rid, state in tracks.items():
                            if state.confidence < min_confidence:
                                x, y, w, h = state.bbox_img
                                cv2.rectangle(annotated, (x, y), (x + w, y + h), (110, 110, 110), 1)

                    writer.write(annotated)

                    if frame_idx % 20 == 0:
                        preview_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_idx}")

                    for rid, state in tracks.items():
                        rows.append(
                            {
                                "frame": frame_idx,
                                "time_s": frame_idx / (fps if fps > 0 else 30.0),
                                "robot_id": rid,
                                "bbox_x": state.bbox_img[0],
                                "bbox_y": state.bbox_img[1],
                                "bbox_w": state.bbox_img[2],
                                "bbox_h": state.bbox_img[3],
                                "center_x": state.center_img[0],
                                "center_y": state.center_img[1],
                                "front_x": state.front_img[0],
                                "front_y": state.front_img[1],
                                "heading_deg": state.heading_deg,
                                "in_dohyo": state.in_dohyo,
                                "eliminated": state.eliminated,
                                "confidence": state.confidence,
                                "dohyo_center_x": dohyo.last_detection_info["center"][0],
                                "dohyo_center_y": dohyo.last_detection_info["center"][1],
                            }
                        )

                    frame_idx += 1
                    if total_frames > 0 and frame_idx % 5 == 0:
                        progress.progress(min(100, int(100 * frame_idx / total_frames)))

                cap.release()
                writer.release()
                progress.progress(100)
                status.text("✅ Done")

                st.session_state.processed_video = out_path
                st.session_state.track_rows = rows

                st.success("Annotated video generated.")

if st.session_state.processed_video:
    st.subheader("Annotated output")
    st.video(st.session_state.processed_video)

    with open(st.session_state.processed_video, "rb") as f:
        st.download_button(
            "⬇️ Download annotated video",
            f,
            file_name=f"sumo_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
            mime="video/mp4",
        )

if st.session_state.track_rows:
    st.subheader("Tracking table")
    df = pd.DataFrame(st.session_state.track_rows)
    st.dataframe(df.tail(200), use_container_width=True)
