# Integration Guide: Connecting Notebooks to Streamlit App

## Overview
This guide shows how to integrate the optimizations from both notebooks into the Streamlit app for production-ready live translation.

---

## Quick Reference

| Component | Source | Integration Point |
|-----------|--------|-------------------|
| Trained Model | `ASL_Model_Optimization_90percent.ipynb` | `models/asl_model.h5` |
| Pipeline Classes | `Live_Translation_Pipeline.ipynb` | `RealTimePipeline` |
| Temporal Smoothing | `Live_Translation_Pipeline.ipynb` | `AdvancedTemporalSmoother` |
| Phrase Recognition | `Live_Translation_Pipeline.ipynb` | `PhraseRecognizer` |
| Performance Metrics | `Live_Translation_Pipeline.ipynb` | FPS/latency tracking |

---

## Step 1: Extract Optimized Classes

The `Live_Translation_Pipeline.ipynb` notebook contains these key classes:

### 1. AdvancedTemporalSmoother
**Purpose**: Reduces noise in frame-to-frame predictions
**Copy to**: `src/utils.py` (update existing SmoothingFilter)

```python
class AdvancedTemporalSmoother:
    """
    Advanced smoothing for real-time prediction stability.
    Handles noisy predictions and provides confidence-weighted output.
    """
    def __init__(self, window_size=5, confidence_threshold=0.6):
        # ...
    
    def smooth(self, prediction, confidence):
        # Weighted voting across window
        # ...
```

### 2. PhraseRecognizer
**Purpose**: Converts letter sequence to phrases
**Copy to**: `src/utils.py`

```python
class PhraseRecognizer:
    """
    Converts sequence of letter predictions into phrases.
    """
    def __init__(self, min_frames_between_letters=10):
        # ...
    
    def add_letter(self, letter):
        # Handles timing, space, delete
        # ...
```

### 3. RealTimePipeline
**Purpose**: Complete end-to-end processing
**Copy to**: `src/pipeline.py` (new file)

```python
class RealTimePipeline:
    """
    Complete pipeline for real-time ASL translation.
    """
    def __init__(self, model, confidence_threshold=0.6):
        # ...
    
    def process_frame(self, frame, show_landmarks=True):
        # Returns FPS, prediction, phrase, etc.
        # ...
```

---

## Step 2: Update Streamlit App

### Current Implementation (app.py)
```python
# Basic components
detector = PoseDetector()
recognizer = GestureRecognizer()
smoothing_filter = SmoothingFilter()
text_buffer = TextBuffer()
```

### Improved Implementation
```python
# Use optimized pipeline
from src.pipeline import RealTimePipeline
from src.utils import AdvancedTemporalSmoother, PhraseRecognizer

pipeline = RealTimePipeline(
    model=recognizer.model,
    confidence_threshold=confidence_threshold,
    smoothing_window=smoothing_window
)

# Process frame
result = pipeline.process_frame(frame, show_landmarks=show_landmarks)

# Use results
st.write(f"Phrase: {result['phrase']}")
st.metric("FPS", f"{result['fps']:.1f}")
st.metric("Inference", f"{result['inference_ms']:.1f}ms")
```

---

## Step 3: Updated Streamlit Live Translation

Here's the recommended refactored `run_live_translation()` function:

```python
def run_live_translation():
    """Run live webcam translation with optimized pipeline."""
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
    
    # Initialize optimized pipeline
    from src.gesture_recognizer import GestureRecognizer
    from src.pipeline import RealTimePipeline
    
    recognizer = GestureRecognizer(model_path="models/asl_model.h5")
    
    if not recognizer.is_trained:
        st.warning("⚠️ Pre-trained model not found. Train first using scripts/train_model.py")
        return
    
    # Create pipeline with settings from sidebar
    pipeline = RealTimePipeline(
        model=recognizer.model,
        confidence_threshold=confidence_threshold,
        smoothing_window=smoothing_window
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not access webcam")
        return
    
    st.session_state.running = start_btn
    
    while st.session_state.running and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break
        
        frame = cv2.flip(frame, 1)
        
        # Process with optimized pipeline
        result = pipeline.process_frame(
            frame,
            show_landmarks=show_landmarks
        )
        
        # Update displays
        frame_placeholder.image(result['frame'], channels="BGR", use_column_width=True)
        text_placeholder.code(result['phrase'] or "Waiting for gestures...")
        
        # Show performance metrics
        col_fps, col_latency = st.columns(2)
        with col_fps:
            st.metric("FPS", f"{result['fps']:.1f}")
        with col_latency:
            st.metric("Latency", f"{result['inference_ms']:.1f}ms")
        
        if clear_btn:
            pipeline.reset()
    
    cap.release()
    pipeline.detector.close()
```

---

## Step 4: File Organization

```
src/
├── __init__.py
├── pose_detector.py              # Existing: hand detection
├── gesture_recognizer.py         # Existing: gesture classification
├── utils.py                      # UPDATE: add new smoothing/phrase classes
├── pipeline.py                   # NEW: RealTimePipeline class
└── performance.py                # NEW: Performance tracking utilities

models/
├── asl_model.h5                  # From training notebook
└── asl_model.tflite              # Quantized (optional for mobile)

notebooks/
├── ASL_Model_Optimization_90percent.ipynb
├── Live_Translation_Pipeline.ipynb
└── IMPROVEMENTS_ANALYSIS.md

app.py                            # UPDATE: use RealTimePipeline
```

---

## Step 5: Performance Optimization Tips

### For Faster Inference
1. **Use TFLite Model** (optional)
   ```python
   # In gesture_recognizer.py
   if use_tflite:
       interpreter = tf.lite.Interpreter("models/asl_model.tflite")
       interpreter.allocate_tensors()
   ```

2. **Enable GPU** (if available)
   ```python
   import tensorflow as tf
   gpus = tf.config.list_physical_devices('GPU')
   if gpus:
       print(f"Using GPU: {gpus}")
   ```

3. **Batch Processing** (for videos)
   ```python
   # Process multiple frames at once
   batch = np.stack([frame1, frame2, frame3])
   predictions = model.predict(batch, verbose=0)
   ```

### For Better Accuracy
1. Increase smoothing window (5-10 frames)
2. Raise confidence threshold (0.7+)
3. Collect more diverse training data
4. Fine-tune model with new data

---

## Step 6: Testing Checklist

- [ ] Trained model loads correctly
- [ ] RealTimePipeline processes frames
- [ ] FPS measured and displayed
- [ ] Smoothing reduces prediction noise
- [ ] Phrases build correctly
- [ ] Webcam input works
- [ ] Performance acceptable (15-30 FPS)
- [ ] Text output clear and usable

---

## Step 7: Deployment

### Local Testing
```bash
streamlit run app.py
```

### Docker Deployment
```dockerfile
FROM python:3.8
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

### Mobile (Optional)
Use TFLite model for mobile deployment:
```python
# models/asl_model.tflite (75% smaller)
# Can run on phones, tablets, edge devices
```

---

## Performance Expectations After Integration

### Metric | Expected Value
- **Model Accuracy** | 85-90%
- **Inference Latency** | 20-50ms
- **Real-time FPS** | 15-30 FPS
- **User Experience** | Smooth, responsive
- **Model Size** | 15MB (full) or 4MB (quantized)

---

## Common Issues & Solutions

### Issue: Low FPS (<10)
**Solutions**:
1. Reduce frame resolution
2. Skip every other frame for inference
3. Use TFLite model instead
4. Enable GPU acceleration
5. Close other applications

### Issue: Noisy Predictions
**Solutions**:
1. Increase smoothing window (5→10)
2. Raise confidence threshold
3. Collect more training data
4. Improve lighting conditions

### Issue: Phrases Not Building
**Solutions**:
1. Check `min_frames_between_letters` setting
2. Verify gestures are confident enough
3. Test with known gesture sequence
4. Debug `PhraseRecognizer.add_letter()`

---

## Next Steps

1. ✅ Train model using `ASL_Model_Optimization_90percent.ipynb`
2. ✅ Generate optimized artifacts using `Live_Translation_Pipeline.ipynb`
3. ✅ Copy classes from notebooks to `src/pipeline.py`
4. ✅ Update `app.py` to use `RealTimePipeline`
5. ✅ Test with `streamlit run app.py`
6. ✅ Measure FPS and accuracy
7. ✅ Optimize further based on metrics

---

## Support Resources

- **Training Details**: `notebooks/ASL_Model_Optimization_90percent.ipynb`
- **Deployment Details**: `notebooks/Live_Translation_Pipeline.ipynb`
- **Implementation Guide**: This file
- **Performance Analysis**: `notebooks/IMPROVEMENTS_ANALYSIS.md`
- **Main App**: `app.py`
- **Quick Start**: `QUICKSTART.md`
