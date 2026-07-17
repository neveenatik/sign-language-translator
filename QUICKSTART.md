# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Choose Your Path

#### Option A: Demo Mode (No Training Data Needed)
```bash
streamlit run app.py
```
Select "Demo" from the sidebar to explore the interface.

#### Option B: Live Translation (Recommended)
You'll need to collect and train the model first.

### 3. Collect Training Data (ASL Alphabet)

Collect samples for each letter (A-Z) plus special gestures:

```bash
# Collect 50 samples for letter A
python scripts/collect_data.py --gesture A --samples 50

# Repeat for other letters (B, C, D, etc.)
python scripts/collect_data.py --gesture B --samples 50
# ... continue for all letters

# Special gestures
python scripts/collect_data.py --gesture space --samples 50
python scripts/collect_data.py --gesture delete --samples 50
```

**Tips while collecting:**
- Good lighting is important
- Keep your hand fully in frame
- Use consistent distance (~30-60cm from camera)
- Capture different angles for each gesture

### 4. Prepare Data
```bash
python scripts/prepare_data.py
```

### 5. Train Model
```bash
python scripts/train_model.py --epochs 50 --batch-size 32
```

This will create `models/asl_model.h5`

### 6. Run Live Translation
```bash
streamlit run app.py
```

Select "Live Translation" and start signing!

## 📁 Expected Folder Structure After Setup

```
sign-language-translator/
├── data/
│   ├── raw/
│   │   ├── A/
│   │   │   ├── A_000.json
│   │   │   ├── A_001.json
│   │   │   └── ...
│   │   ├── B/, C/, ... Z/
│   │   ├── space/
│   │   └── delete/
│   └── processed/
│       ├── A_000.npz
│       ├── A_001.npz
│       └── ...
└── models/
    └── asl_model.h5
```

## 🎯 Minimum Requirements for Good Accuracy

- **At least 30 samples per gesture** (more is better)
- **26 letters** (A-Z)
- **2 special gestures** (space, delete)
- **Total: ~900+ samples**

For better accuracy:
- 50-100 samples per gesture
- Varied lighting conditions
- Different hand positions/distances
- Total: 1,300-2,600 samples

## ⚡ Performance Tips

- **Faster Training**: Use `--batch-size 64` or higher
- **Better Accuracy**: Increase `--epochs 100` or more
- **Live Translation**: Adjust "Smoothing Window" (higher = smoother but slower response)
- **Confidence**: Set to 0.5-0.7 for balanced detection

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No data found" | Collect gesture data using `collect_data.py` |
| Low accuracy | Collect more samples (100+ per gesture) |
| Webcam not working | Check camera permissions in System Preferences |
| Slow performance | Disable "Show Landmarks" or reduce epochs |
| Model file not found | Train the model first: `python scripts/train_model.py` |

## 📚 Next Steps

1. Improve model accuracy by collecting more diverse data
2. Fine-tune hyperparameters (learning rate, layer sizes)
3. Experiment with different architectures
4. Add continuous gesture recognition (not just letters)
5. Deploy as a web service

## 💡 Pro Tips

- **Capture Multiple Angles**: Train on different hand positions
- **Use Multiple Hands**: Collect data with both left and right hands
- **Test After Training**: Verify accuracy before deploying
- **Regular Retraining**: Collect more data and retrain periodically
- **Version Your Models**: Save models with timestamps

---

Need help? Check `README.md` for comprehensive documentation!
