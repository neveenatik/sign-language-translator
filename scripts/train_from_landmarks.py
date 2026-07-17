#!/usr/bin/env python3
"""Train the ASL gesture model from a prebuilt landmark ``.npz`` and save it.

Use this after ``scripts/build_dataset_from_images.py``. Unlike
``scripts/train_model.py`` this deliberately does **not** apply a
``StandardScaler``, so the features match exactly what the live app feeds at
inference time (the raw ``PoseDetector.normalize_landmarks`` output). Keeping
train-time and inference-time preprocessing identical is what makes the live
predictions trustworthy.

Usage:
    python scripts/train_from_landmarks.py \
        --input data/processed/landmarks.npz \
        --epochs 60 --output models/asl_model.h5
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gesture_recognizer import GestureRecognizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/processed/landmarks.npz")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output", default="models/asl_model.h5")
    args = ap.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        print(f"❌ Landmark dataset not found: {data_path}")
        print("   Build it first with scripts/build_dataset_from_images.py")
        sys.exit(1)

    data = np.load(data_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    print(f"📊 Loaded {len(X)} samples with {X.shape[1]} features")

    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    recognizer = GestureRecognizer()
    n_classes = len(recognizer.get_labels())

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_train_oh = tf.keras.utils.to_categorical(y_train, n_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, n_classes)

    print(f"🏋️  Training: {len(X_train)} train / {len(X_val)} val, "
          f"{args.epochs} epochs (early stopping on val_accuracy)\n")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        ),
    ]
    recognizer.model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    recognizer.save_model(args.output)

    _, val_acc = recognizer.model.evaluate(X_val, y_val_oh, verbose=0)
    print(f"\n✅ Saved {args.output}")
    print(f"   Validation accuracy: {val_acc:.3f}")


if __name__ == "__main__":
    main()
