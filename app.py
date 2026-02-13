"""
Robot Sumo Analyzer - Main Application
Streamlit interface for match video analysis
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime
import tempfile

# Page configuration
st.set_page_config(
    page_title="Robot Sumo Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None
if 'match_data' not in st.session_state:
    st.session_state.match_data = None
if 'robot_database' not in st.session_state:
    st.session_state.robot_database = {}

# Title
st.title("🤖 Robot Sumo Match Analyzer")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Detection settings
    st.subheader("Detection")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    use_auto_calibration = st.checkbox("Auto Dohyo Calibration", value=True)
    is_slowmo = st.checkbox("Slow Motion Video", value=False)
    
    # Strategy thresholds
    st.subheader("Strategy Classification")
    positioning_speed = st.number_input("Positioning Max Speed (cm/s)", value=5, step=1)
    search_speed_min = st.number_input("Search Min Speed (cm/s)", value=10, step=5)
    search_speed_max = st.number_input("Search Max Speed (cm/s)", value=40, step=5)
    attack_speed = st.number_input("Attack Min Speed (cm/s)", value=50, step=10)
    special_rotation = st.number_input("Special Rotation Rate (deg/s)", value=180, step=30)
    
    st.markdown("---")
    st.markdown("**GPU Status:**")
    try:
        import torch
        if torch.cuda.is_available():
            st.success(f"✅ {torch.cuda.get_device_name(0)}")
        else:
            st.warning("⚠️ CUDA not available")
    except:
        st.error("❌ PyTorch not installed")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📹 Video Processing", "📊 Match Analysis", "🗄️ Robot Database"])

# ============================================================================
# TAB 1: VIDEO PROCESSING
# ============================================================================
with tab1:
    st.header("Upload and Process Match Video")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a match video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file of a robot sumo match"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Display video info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📁 File: {uploaded_file.name}")
            st.info(f"💾 Size: {uploaded_file.size / 1024 / 1024:.2f} MB")
        
        # Video preview
        st.subheader("Video Preview")
        video_col1, video_col2 = st.columns(2)
        
        with video_col1:
            st.video(uploaded_file)
        
        with video_col2:
            # Get video properties
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            st.metric("Resolution", f"{width}x{height}")
            st.metric("Frame Rate", f"{fps} FPS")
            st.metric("Duration", f"{duration:.2f}s")
            st.metric("Total Frames", frame_count)
        
        # Manual calibration option
        if not use_auto_calibration:
            st.subheader("Manual Dohyo Calibration")
            st.info("Click 3 points on the white line in the first frame")
            # Placeholder for manual calibration interface
            st.warning("⚠️ Manual calibration UI - Coming in next update")
        
        # Process button
        st.markdown("---")
        if st.button("🚀 Analyze Match", type="primary", use_container_width=True):
            with st.spinner("Processing video... This may take 30-60 seconds"):
                try:
                    # Import processing modules
                    from models.dohyo import DohyoDetector
                    from models.homography import HomographyTransform
                    from tracking.tracker import RobotTracker
                    from strategy.classifier import StrategyClassifier
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Initialize detectors
                    status_text.text("Initializing models...")
                    dohyo_detector = DohyoDetector()
                    progress_bar.progress(10)
                    
                    # Step 2: Detect dohyo
                    status_text.text("Detecting dohyo...")
                    cap = cv2.VideoCapture(video_path)
                    ret, first_frame = cap.read()
                    if not ret:
                        st.error("Failed to read video")
                        cap.release()
                    else:
                        # Expect detect() -> (center, radius) or None/None
                        center, radius = dohyo_detector.detect(first_frame)

                        if center is None or radius is None:
                            st.error("Dohyo not detected. Try manual calibration.")
                            cap.release()
                        else:
                            progress_bar.progress(20)

                            # 77 cm radius, dohyo_detector should have a method or use known size
                            # If DohyoDetector has get_physical_scale(radius): use that
                            try:
                                scale = dohyo_detector.get_physical_scale(radius)  # px per cm
                            except AttributeError:
                                # Fallback: assume 77 cm radius
                                scale = radius / 77.0

                            dohyo_result = {
                                "center": center,
                                "radius": radius,
                                "scale": scale,      # ← REQUIRED by HomographyTransform
                                "shape": "circle",
                            }

                            # If your detector stores ellipse info, propagate it
                            if getattr(dohyo_detector, "last_detection_info", None):
                                info = dohyo_detector.last_detection_info
                                if info.get("shape") == "ellipse":
                                    dohyo_result["shape"] = "ellipse"
                                    # params typically (d1, d2, angle)
                                    if "params" in info:
                                        dohyo_result["params"] = info["params"]

                            # Step 3: Setup homography
                            status_text.text("Calibrating perspective transform...")
                            homography = HomographyTransform(dohyo_result)

                            progress_bar.progress(30)
                            
                            # Step 4: Process video
                            status_text.text("Detecting and tracking robots...")
                            tracker = RobotTracker(dohyo_center=center, dohyo_radius=radius)

                            strategy_classifier = StrategyClassifier(
                                positioning_speed=positioning_speed,
                                search_speed=(search_speed_min, search_speed_max),
                                attack_speed=attack_speed,
                                special_rotation=special_rotation
                            )
                            
                            all_detections = []
                            frame_idx = 0
                            
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                # Track robots
                                robots_dict = tracker.update(frame)  # Dict[int, RobotState]

                                # Filter: keep only highest confidence robots inside dohyo (max 2)
                                candidates = []
                                for robot_id, robot_state in robots_dict.items():
                                    if (robot_state.confidence > 0.3 and  # confidence threshold
                                        len(robot_state.position_history) > 3):  # tracked for a while
                                        candidates.append((robot_id, robot_state))

                                # Sort by confidence, take top 2
                                candidates.sort(key=lambda x: x[1].confidence, reverse=True)
                                top_robots = dict(candidates[:2])

                                # Convert to list of dicts for compatibility (max 2 active robots)
                                tracked_robots = []
                                for robot_id, robot_state in top_robots.items():
                                    robot_dict = {
                                        "id": robot_id,
                                        "center": robot_state.position,
                                        "bbox": robot_state.bbox,
                                        "velocity": robot_state.velocity,
                                        "confidence": robot_state.confidence,
                                        "front_point": robot_state.front_point,
                                        "heading_deg": robot_state.heading_deg,
                                        "occluded": robot_state.occluded,
                                        "top_down_pos": None,
                                    }
                                    tracked_robots.append(robot_dict)

                                # Transform to top-down
                                for robot in tracked_robots:
                                    robot["top_down_pos"] = homography.transform_point(robot["center"])

                                all_detections.append({
                                    'frame': frame_idx,
                                    'timestamp': frame_idx / fps,
                                    'robots': tracked_robots
                                })
                                # Live preview every 30 frames
                                show_preview = st.session_state.get("show_preview", True)
                                if frame_idx % 30 == 0 and show_preview:
                                    # Draw visualization
                                    vis_frame = frame.copy()
                                    
                                    # Draw dohyo
                                    cv2.circle(vis_frame, center, radius, (0, 255, 0), 3)
                                    
                                    # Draw robots
                                    for robot in tracked_robots:
                                        color = (0, 255, 255) if robot.get("occluded") else (0, 255, 0)
                                        cv2.rectangle(vis_frame, 
                                                    (robot["bbox"][0], robot["bbox"][1]), 
                                                    (robot["bbox"][0] + robot["bbox"][2], robot["bbox"][1] + robot["bbox"][3]), 
                                                    color, 2)
                                        cv2.circle(vis_frame, robot["center"], 8, (0, 0, 255), -1)
                                        if robot.get("front_point"):
                                            cv2.circle(vis_frame, robot["front_point"], 4, (255, 255, 0), -1)
                                            cv2.line(vis_frame, robot["center"], robot["front_point"], (255, 255, 0), 2)
                                        label = f"R{robot['id']}:{robot['confidence']:.1f}"
                                        if robot.get("occluded"):
                                            label += " OCC"
                                        cv2.putText(vis_frame, label, 
                                                (robot["center"][0]-30, robot["center"][1]-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                    
                                    # Show in Streamlit (resize for preview)
                                    preview_img = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                                    preview_img = cv2.resize(preview_img, (640, 480))
                                    
                                    # Store in session state for display
                                    if "preview_frames" not in st.session_state:
                                        st.session_state.preview_frames = []
                                    st.session_state.preview_frames.append(preview_img)
                                    
                                    if len(st.session_state.preview_frames) > 20:  # Keep last 20 frames
                                        st.session_state.preview_frames.pop(0)
                                
                                frame_idx += 1
                                if frame_idx % 10 == 0:
                                    progress = 30 + int((frame_idx / frame_count) * 40)
                                    progress_bar.progress(min(progress, 70))
                            
                            cap.release()
                            progress_bar.progress(75)
                            
                            # Step 5: Classify strategies
                            status_text.text("Classifying strategies...")
                            match_data = strategy_classifier.classify_match(all_detections)
                            progress_bar.progress(90)
                            
                            # Step 6: Generate visualizations
                            status_text.text("Generating visualizations...")
                            # Store results in session state
                            st.session_state.match_data = match_data
                            st.session_state.video_path = video_path
                            progress_bar.progress(100)
                            
                            status_text.text("✅ Processing complete!")
                            st.success("Match analyzed successfully!")
                            st.balloons()
                
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    st.exception(e)
    # Live Preview Section
    st.subheader("🔴 Live Processing Preview")
    if hasattr(st.session_state, "preview_frames") and st.session_state.preview_frames:
        latest_preview = st.session_state.preview_frames[-1]
        st.image(latest_preview, caption=f"Frame {frame_idx} Preview")
    else:
        st.info("Processing preview will appear here...")
# ============================================================================
# TAB 2: MATCH ANALYSIS
# ============================================================================
with tab2:
    st.header("Match Analysis & Visualization")
    
    if st.session_state.match_data is None:
        st.info("👈 Process a video in the 'Video Processing' tab first")
    else:
        match_data = st.session_state.match_data
        
        # Summary statistics
        st.subheader("📈 Match Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Duration", f"{match_data['duration']:.2f}s")
        with col2:
            st.metric("Total Frames", match_data['total_frames'])
        with col3:
            st.metric("Robots Detected", len(match_data['robots']))
        with col4:
            st.metric("Strategy Changes", match_data['strategy_changes'])
        
        # Trajectory plot
        st.subheader("🗺️ Top-Down Trajectory")
        
        fig = go.Figure()
        
        # Draw dohyo
        theta = np.linspace(0, 2*np.pi, 100)
        dohyo_r = 154 / 2  # cm
        fig.add_trace(go.Scatter(
            x=dohyo_r * np.cos(theta),
            y=dohyo_r * np.sin(theta),
            mode='lines',
            line=dict(color='black', width=2),
            name='Dohyo Outer',
            showlegend=False
        ))
        
        # Draw white line
        white_r = dohyo_r - 5
        fig.add_trace(go.Scatter(
            x=white_r * np.cos(theta),
            y=white_r * np.sin(theta),
            mode='lines',
            line=dict(color='gray', width=2, dash='dash'),
            name='White Line',
            showlegend=False
        ))
        
        # Color mapping for strategies
        strategy_colors = {
            'positioning': 'blue',
            'search': 'green',
            'attack': 'red',
            'special': 'orange'
        }
        
        # Plot robot trajectories
        for robot_id, robot_data in match_data['robots'].items():
            trajectory = robot_data['trajectory']
            strategies = robot_data['strategies']
            
            # Plot trajectory with color coding
            x_coords = [p[0] for p in trajectory]
            y_coords = [p[1] for p in trajectory]
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines+markers',
                line=dict(width=3),
                marker=dict(size=4),
                name=f"Robot {robot_id}",
                hovertemplate=f"Robot {robot_id}<br>X: %{{x:.1f}}cm<br>Y: %{{y:.1f}}cm<extra></extra>"
            ))
        
        fig.update_layout(
            width=800,
            height=800,
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
                range=[-100, 100],
                title="X (cm)"
            ),
            yaxis=dict(
                range=[-100, 100],
                title="Y (cm)"
            ),
            title="Robot Trajectories (Top-Down View)",
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics table
        st.subheader("📊 Robot Statistics")
        
        stats_data = []
        for robot_id, robot_data in match_data['robots'].items():
            stats_data.append({
                'Robot ID': robot_id,
                'Avg Speed (cm/s)': f"{robot_data['avg_speed']:.1f}",
                'Max Speed (cm/s)': f"{robot_data['max_speed']:.1f}",
                'Attack Count': robot_data['attack_count'],
                'Center Time (s)': f"{robot_data['center_time']:.2f}",
                'Distance (cm)': f"{robot_data['total_distance']:.1f}"
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # Strategy timeline
        st.subheader("⏱️ Strategy Timeline")
        
        timeline_fig = go.Figure()
        
        for robot_id, robot_data in match_data['robots'].items():
            strategies = robot_data['strategy_timeline']
            
            for strategy_segment in strategies:
                timeline_fig.add_trace(go.Scatter(
                    x=[strategy_segment['start'], strategy_segment['end']],
                    y=[robot_id, robot_id],
                    mode='lines',
                    line=dict(
                        color=strategy_colors.get(strategy_segment['strategy'], 'gray'),
                        width=20
                    ),
                    name=strategy_segment['strategy'],
                    showlegend=True,
                    hovertemplate=f"Robot {robot_id}<br>{strategy_segment['strategy']}<br>%{{x:.2f}}s<extra></extra>"
                ))
        
        timeline_fig.update_layout(
            title="Strategy Timeline by Robot",
            xaxis_title="Time (seconds)",
            yaxis_title="Robot ID",
            height=300,
            hovermode='closest'
        )
        
        st.plotly_chart(timeline_fig, use_container_width=True)
        
        # Export options
        st.subheader("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export CSV", use_container_width=True):
                # Generate CSV data
                csv_data = match_data['export_csv']()
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"match_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export JSON", use_container_width=True):
                # Generate JSON data
                import json
                json_data = json.dumps(match_data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"match_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            st.button("Export Video", use_container_width=True, disabled=True)
            st.caption("Coming soon")

# ============================================================================
# TAB 3: ROBOT DATABASE
# ============================================================================
with tab3:
    st.header("Robot Database & Identity Management")
    
    if len(st.session_state.robot_database) == 0:
        st.info("No robots in database yet. Process some matches to build the database.")
    else:
        st.subheader("🤖 Detected Robots")
        
        # Robot gallery
        cols = st.columns(4)
        for idx, (robot_id, robot_info) in enumerate(st.session_state.robot_database.items()):
            with cols[idx % 4]:
                st.image(robot_info['thumbnail'], use_column_width=True)
                st.caption(f"Robot {robot_id}")
                
                # Rename button
                new_name = st.text_input(
                    "Name",
                    value=robot_info.get('name', f"Robot {robot_id}"),
                    key=f"name_{robot_id}"
                )
                
                if st.button("Update", key=f"update_{robot_id}"):
                    robot_info['name'] = new_name
                    st.success(f"Updated to: {new_name}")
                
                # Statistics
                st.metric("Matches", robot_info['match_count'])
                st.metric("Win Rate", f"{robot_info['win_rate']*100:.1f}%")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Robot Sumo Analyzer v1.0 | Built for 3kg Mega Sumo Analysis</p>
    </div>
    """,
    unsafe_allow_html=True
)
