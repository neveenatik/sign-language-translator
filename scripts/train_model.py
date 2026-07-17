"""
Training script for ASL gesture recognition model.
Trains a neural network on collected gesture data.
"""

import numpy as np
import json
import os
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gesture_recognizer import GestureRecognizer
from src.pose_detector import PoseDetector


def load_gesture_data():
    """
    Load all collected gesture data.
    
    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    data_dir = Path("data/raw")
    
    X = []
    y = []
    
    recognizer = GestureRecognizer()
    labels_to_idx = {label: idx for idx, label in enumerate(recognizer.get_labels())}
    
    detector = PoseDetector()
    
    print("📂 Loading gesture data...")
    
    for gesture_folder in data_dir.iterdir():
        if not gesture_folder.is_dir():
            continue
        
        gesture = gesture_folder.name
        if gesture not in labels_to_idx:
            print(f"⚠️  Skipping unknown gesture: {gesture}")
            continue
        
        label_idx = labels_to_idx[gesture]
        
        print(f"  Loading {gesture}... ", end="")
        gesture_samples = 0
        
        for json_file in gesture_folder.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    landmarks = np.array(json.load(f))
                
                # Normalize landmarks
                normalized = detector.normalize_landmarks(landmarks)
                X.append(normalized)
                y.append(label_idx)
                gesture_samples += 1
            except Exception as e:
                print(f"⚠️  Error loading {json_file}: {e}")
        
        print(f"✓ ({gesture_samples} samples)")
    
    detector.close()
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n📊 Total samples loaded: {len(X)}")
    print(f"   Features shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    
    return X, y


def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test and normalize.
    
    Args:
        X: Features
        y: Labels
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print("\n🔀 Splitting data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert labels to one-hot
    import tensorflow as tf
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=28)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=28)
    
    return X_train, X_test, y_train_onehot, y_test_onehot


def train_model(epochs=50, batch_size=32):
    """
    Load data and train the model.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("\n🤖 ASL Gesture Recognition Model Training\n")
    
    # Load data
    X, y = load_gesture_data()
    
    if len(X) == 0:
        print("❌ No training data found. Please collect gesture data first.")
        print("   Use: python scripts/collect_data.py --gesture A --samples 50")
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    
    # Create and train model
    print("\n🏋️  Training model...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}\n")
    
    recognizer = GestureRecognizer()
    
    history = recognizer.train(
        X_train, y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/asl_model.h5"
    recognizer.save_model(model_path)
    
    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {model_path}")
    
    # Print summary
    final_train_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    
    print(f"\n📊 Final Metrics:")
    print(f"   Training Accuracy: {final_train_acc:.4f}")
    print(f"   Validation Accuracy: {final_val_acc:.4f}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train ASL gesture recognition model")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
