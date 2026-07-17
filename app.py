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
from src.utils import SmoothingFilter, TextBuffer, display_fps, display_gesture, resize_frame, prepare_display_text, draw_text
from src.languages import available_languages, get_language, DEFAULT_LANGUAGE
import os
import tempfile

import av
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer


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
    
    _languages = available_languages()
    _lang_codes = [lang.code for lang in _languages]
    language_code = st.selectbox(
        "Sign Language:",
        _lang_codes,
        index=_lang_codes.index(DEFAULT_LANGUAGE),
        format_func=lambda code: get_language(code).name,
        help="Choose which sign language alphabet to recognize.",
    )
    language = get_language(language_code)
    
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
        "The model works best with good lighting and clear hand visibility. "
        "Perform gestures slowly and hold each sign steady so the model can "
        "predict accurately."
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


RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


class ASLVideoProcessor(VideoProcessorBase):
    """Runs hand detection + gesture recognition on each browser webcam frame."""

    STABLE_FRAMES = 6      # frames a letter must persist before it's committed
    COOLDOWN_FRAMES = 10   # frames to wait after committing a letter

    def __init__(self, language=None):
        self.language = language if language is not None else get_language(DEFAULT_LANGUAGE)
        self.detector = PoseDetector(static_image_mode=False, max_num_hands=1)
        model_path = self.language.model_path
        self.recognizer = GestureRecognizer(
            model_path=model_path if os.path.exists(model_path) else None,
            labels=self.language.labels,
        )
        self.smoother = SmoothingFilter(window_size=5)
        self.confidence_threshold = 0.6
        self.show_landmarks = True
        self.sentence = ""
        self._last_label = None
        self._stable = 0
        self._cooldown = 0

    def _commit(self, label):
        if label == "space":
            self.sentence += " "
        elif label == "delete":
            self.sentence = self.sentence[:-1]
        else:
            self.sentence += label
        self.sentence = self.sentence[-40:]

    def _update_sentence(self, label):
        if label is None:
            self._last_label = None
            self._stable = 0
            return
        if label == self._last_label:
            self._stable += 1
        else:
            self._last_label = label
            self._stable = 1
        if self._cooldown > 0:
            self._cooldown -= 1
            return
        if self._stable == self.STABLE_FRAMES:
            self._commit(label)
            self._cooldown = self.COOLDOWN_FRAMES

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        if self.show_landmarks:
            annotated, hands = self.detector.detect(img)
        else:
            _, hands = self.detector.detect(img)
            annotated = img

        label = None
        if not self.recognizer.is_trained:
            cv2.putText(annotated, "Model not loaded", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        elif hands:
            norm = self.detector.normalize_landmarks(hands[0])
            raw = self.recognizer.predict(norm, self.confidence_threshold)
            smooth = self.smoother.smooth_prediction(raw)
            if smooth:
                label = smooth["label"]
                annotated = display_gesture(
                    annotated, label, raw["confidence"] if raw else 0.0
                )
        self._update_sentence(label)

        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, h - 50), (w, h), (0, 0, 0), -1)
        display_sentence = prepare_display_text(self.sentence, self.language.rtl) or "..."
        annotated = draw_text(
            annotated, display_sentence, (15, h - 15),
            rtl=self.language.rtl, color=(0, 255, 0),
        )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def run_live_translation():
    """Run live browser-webcam translation via WebRTC."""
    st.subheader("📹 Live Translation")
    st.caption(
        f"Allow camera access, then sign {language.name} letters. Hold each sign "
        "steady for a moment — recognized letters build up at the bottom of the "
        "video."
    )

    if not os.path.exists(language.model_path):
        st.warning(
            f"⚠️ **Pre-trained model not found** (`{language.model_path}`). "
            "The camera will still stream and track your hand, but letters won't "
            "be recognized until a model is trained for this language. See the "
            "training guide below."
        )

    ctx = webrtc_streamer(
        key=f"live-{language.code}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: ASLVideoProcessor(language=language),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.confidence_threshold = confidence_threshold
        ctx.video_processor.show_landmarks = show_landmarks

    if st.button("🗑️ Clear text"):
        if ctx.video_processor:
            ctx.video_processor.sentence = ""
    st.caption("Tip: the 'space' and 'delete' signs edit the running text.")


def run_upload_video():
    """Translate an uploaded ASL fingerspelling video into text."""
    st.subheader("📹 Upload Video")
    st.caption(
        "Upload a short clip of sign-language fingerspelling. The app extracts "
        "hand landmarks frame-by-frame and builds the recognized text."
    )

    if not os.path.exists(language.model_path):
        st.warning(
            f"⚠️ **Pre-trained model not found** (`{language.model_path}`), "
            "so video can't be translated yet for this language. See the "
            "training guide below."
        )
        return

    uploaded = st.file_uploader(
        "Choose a video file", type=["mp4", "mov", "avi", "mkv", "webm"]
    )
    if uploaded is None:
        return

    st.video(uploaded)

    if not st.button("🔤 Translate video"):
        return

    # Persist the upload to a temp file so OpenCV can read it.
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not read the uploaded video. Try a different format (mp4).")
        _safe_remove(video_path)
        return

    detector = PoseDetector(static_image_mode=False, max_num_hands=1)
    recognizer = GestureRecognizer(model_path=language.model_path, labels=language.labels)
    smoother = SmoothingFilter(window_size=smoothing_window)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / 8)))  # analyze ~8 frames per second

    STABLE_FRAMES = 3      # sampled frames a letter must persist to be committed
    COOLDOWN_FRAMES = 4    # sampled frames to wait after committing
    sentence = ""
    last_label = None
    stable = 0
    cooldown = 0

    progress = st.progress(0.0, text="Analyzing video…")
    preview = st.empty()
    idx = 0
    processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step != 0:
            idx += 1
            continue

        annotated, hands = detector.detect(frame)
        label = None
        if hands:
            norm = detector.normalize_landmarks(hands[0])
            raw = recognizer.predict(norm, confidence_threshold)
            smooth = smoother.smooth_prediction(raw)
            if smooth:
                label = smooth["label"]
                annotated = display_gesture(
                    annotated, label, raw["confidence"] if raw else 0.0
                )

        # Debounce: commit a letter only after it stays stable, then cool down.
        if label is None:
            last_label, stable = None, 0
        else:
            if label == last_label:
                stable += 1
            else:
                last_label, stable = label, 1
            if cooldown > 0:
                cooldown -= 1
            elif stable == STABLE_FRAMES:
                if label == "space":
                    sentence += " "
                elif label == "delete":
                    sentence = sentence[:-1]
                else:
                    sentence += label
                cooldown = COOLDOWN_FRAMES

        processed += 1
        if processed % 5 == 0:
            _preview_text = prepare_display_text(sentence, language.rtl) or "…"
            preview.image(
                annotated, channels="BGR", caption=f"Reading: {_preview_text}"
            )
        if total:
            progress.progress(min(idx / total, 1.0), text="Analyzing video…")
        idx += 1

    cap.release()
    _safe_remove(video_path)
    progress.progress(1.0, text="Done")

    st.success("✅ Translation complete")
    st.markdown("### Recognized text")
    st.text_area(
        "Output",
        value=prepare_display_text(sentence.strip(), language.rtl)
        or "(no letters recognized — try clearer, slower signing)",
        height=100,
    )


def _safe_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


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

        #### Option A — Train from a public image dataset (recommended)
        1. Download an ASL alphabet image dataset (e.g. Kaggle *ASL Alphabet*)
           with one subfolder per letter.
        2. Extract landmarks into a training set:
        ```bash
        python scripts/build_dataset_from_images.py \\
            --input path/to/asl_alphabet_train --per-class 400
        ```
        3. Train the model (features match the live app exactly):
        ```bash
        python scripts/train_from_landmarks.py --epochs 60
        ```

        #### Option B — Collect your own samples
        1. Capture landmarks per letter with your webcam:
        ```bash
        python scripts/collect_data.py --gesture A --samples 100
        ```
        2. Train on the collected data:
        ```bash
        python scripts/train_model.py
        ```

        #### Result
        - The model is saved to `models/asl_model.h5` and is picked up
          automatically by Live Translation.
        """)


# Main app logic
if mode == "Live Translation":
    run_live_translation()
elif mode == "Upload Video":
    run_upload_video()
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
