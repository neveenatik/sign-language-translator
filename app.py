"""
Real-time ASL (American Sign Language) to Text Translator
Live webcam translation using hand pose detection and gesture recognition.
"""

import streamlit as st
import cv2
import numpy as np
import time
from src.pose_detector import PoseDetector
from src.gesture_recognizer import GestureRecognizer
from src.utils import SmoothingFilter, TextBuffer, display_fps, display_gesture, resize_frame
import os


# Page configuration
st.set_page_config(
    page_title="ASL to Text Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤟 Real-time ASL to Text Translator")
st.markdown("Translate American Sign Language to text using your webcam in real-time.")


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    mode = st.radio(
        "Select Mode:",
        ["Live Translation", "Upload Video", "Demo"]
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Minimum confidence to display prediction"
    )
    
    smoothing_window = st.slider(
        "Smoothing Window:",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of frames to smooth predictions"
    )
    
    show_landmarks = st.checkbox(
        "Show Hand Landmarks",
        value=True,
        help="Visualize detected hand landmarks"
    )
    
    st.markdown("---")
    st.info(
        "💡 **Tip**: Position your hands clearly in front of the camera. "
        "The model works best with good lighting and clear hand visibility."
    )


# Initialize session state
if 'text_buffer' not in st.session_state:
    st.session_state.text_buffer = TextBuffer()
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0


def initialize_detector_and_recognizer():
    """Initialize pose detector and gesture recognizer."""
    detector = PoseDetector(static_image_mode=False, max_num_hands=2)
    
    # Try to load pre-trained model if it exists
    model_path = "models/asl_model.h5"
    recognizer = GestureRecognizer(model_path=model_path if os.path.exists(model_path) else None)
    
    return detector, recognizer


def process_frame(frame, detector, recognizer, smoothing_filter, confidence_threshold, show_landmarks):
    """
    Process a single frame.
    
    Args:
        frame: Input frame
        detector: PoseDetector instance
        recognizer: GestureRecognizer instance
        smoothing_filter: SmoothingFilter instance
        confidence_threshold: Confidence threshold for predictions
        show_landmarks: Whether to show landmarks
        
    Returns:
        Tuple of (processed_frame, prediction, text_output)
    """
    frame = resize_frame(frame, width=640, height=480)
    
    if show_landmarks:
        annotated_frame, hand_landmarks_list = detector.detect(frame)
    else:
        _, hand_landmarks_list = detector.detect(frame)
        annotated_frame = frame.copy()
    
    prediction = None
    text_output = ""
    
    if hand_landmarks_list and recognizer.is_trained:
        # Process first detected hand
        landmarks = hand_landmarks_list[0]
        normalized_landmarks = detector.normalize_landmarks(landmarks)
        
        # Get raw prediction
        raw_prediction = recognizer.predict(normalized_landmarks, confidence_threshold)
        
        # Apply smoothing
        smoothed_prediction = smoothing_filter.smooth_prediction(raw_prediction)
        
        if smoothed_prediction:
            prediction = smoothed_prediction
            text = st.session_state.text_buffer.add_character(prediction['label'])
            text_output = text
            
            # Display on frame
            if raw_prediction:
                annotated_frame = display_gesture(
                    annotated_frame,
                    prediction['label'],
                    raw_prediction['confidence']
                )
    
    return annotated_frame, prediction, text_output


def run_live_translation():
    """Run live webcam translation."""
    st.subheader("📹 Live Translation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        frame_placeholder = st.empty()
    
    with col2:
        st.markdown("### Translated Text:")
        text_placeholder = st.empty()
        stats_placeholder = st.empty()
    
    # Control buttons
    col1_btn, col2_btn, col3_btn = st.columns(3)
    with col1_btn:
        start_btn = st.button("▶️ Start", key="start")
    with col2_btn:
        stop_btn = st.button("⏹️ Stop", key="stop")
    with col3_btn:
        clear_btn = st.button("🗑️ Clear", key="clear")
    
    detector, recognizer = initialize_detector_and_recognizer()
    smoothing_filter = SmoothingFilter(window_size=smoothing_window)
    
    if not recognizer.is_trained:
        st.warning(
            "⚠️ **Pre-trained model not found.** "
            "The model will need to be trained with ASL gesture data first. "
            "See the Training section for guidance."
        )
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not access webcam. Please check permissions.")
        return
    
    st.session_state.running = start_btn
    
    fps = 0
    frame_time_queue = []
    
    while st.session_state.running and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break
        
        # Calculate FPS
        frame_start = time.time()
        
        frame = cv2.flip(frame, 1)
        processed_frame, prediction, text_output = process_frame(
            frame,
            detector,
            recognizer,
            smoothing_filter,
            confidence_threshold,
            show_landmarks
        )
        
        # Display FPS
        frame_time = time.time() - frame_start
        frame_time_queue.append(frame_time)
        if len(frame_time_queue) > 10:
            frame_time_queue.pop(0)
            fps = 10 / sum(frame_time_queue)
        
        processed_frame = display_fps(processed_frame, fps)
        
        # Update displays
        frame_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
        text_placeholder.code(st.session_state.text_buffer.get_text() or "Waiting for gestures...")
        stats_placeholder.metric("FPS", f"{fps:.1f}")
        
        if clear_btn:
            st.session_state.text_buffer.clear()
            smoothing_filter.reset()
    
    cap.release()
    detector.close()


def run_demo():
    """Run demo mode with sample image."""
    st.subheader("🎬 Demo Mode")
    st.info("Demo mode - shows sample processing. Upload video or use Live Translation for real-time translation.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sample Processing")
        # Create a sample frame
        sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(sample_frame, "Start Live Translation", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        st.image(sample_frame, channels="BGR")
    
    with col2:
        st.markdown("### Instructions")
        st.markdown("""
        1. **Select Live Translation** from the sidebar
        2. **Click Start** to begin
        3. **Make ASL gestures** in front of the camera
        4. **Watch** as gestures are recognized
        5. **Copy** your translated text
        """)


def run_training_guide():
    """Display training guide."""
    st.subheader("🎓 Model Training Guide")
    
    with st.expander("📚 How to Train the Model", expanded=False):
        st.markdown("""
        ### Training Your ASL Recognition Model
        
        #### Step 1: Collect Training Data
        1. Create labeled folders in `data/raw/` for each ASL letter (A-Z, space, delete)
        2. Use the `collect_data.py` script to gather hand landmark data
        3. Aim for 50-100 samples per gesture
        
        #### Step 2: Prepare the Data
        ```bash
        python scripts/prepare_data.py
        ```
        
        #### Step 3: Train the Model
        ```bash
        python scripts/train_model.py
        ```
        
        #### Step 4: Evaluate the Model
        - Check the training metrics in `notebooks/ASL_Model_Optimization_90percent.ipynb`
        - Fine-tune hyperparameters if needed
        
        #### Step 5: Save and Deploy
        - The model will be saved to `models/asl_model.h5`
        - It will automatically be used in the Live Translation mode
        """)


# Main app logic
if mode == "Live Translation":
    run_live_translation()
elif mode == "Upload Video":
    st.subheader("📹 Upload Video")
    st.info("Video upload functionality - coming soon!")
elif mode == "Demo":
    run_demo()

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 📖 Resources")
    st.markdown("[MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)")
    st.markdown("[TensorFlow](https://tensorflow.org)")

with col2:
    st.markdown("### 💾 Data Location")
    st.markdown("`models/` - Trained models")
    st.markdown("`data/` - Training data")

with col3:
    st.markdown("### 🔧 Quick Links")
    st.markdown("[Training Guide](#-model-training-guide)")
    st.markdown("[Documentation](./README.md)")
