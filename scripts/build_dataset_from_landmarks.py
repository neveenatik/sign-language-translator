#!/usr/bin/env python3
"""Build a landmark dataset (X, y) from webcam-collected JSON samples.

Consumes the raw landmark JSON files produced by ``scripts/collect_data.py``
(one folder per label under ``data/raw/<language>/<label>/*.json``), applies the
same ``PoseDetector.normalize_landmarks`` transform used at inference time, and
writes a compressed ``.npz`` that ``scripts/train_from_landmarks.py`` can train
on.

This is the counterpart to ``build_dataset_from_images.py`` for the
"collect your own data" workflow — the realistic path for ArSL, where public
single-hand landmark datasets are scarce.

Usage:
    python scripts/build_dataset_from_landmarks.py --language arsl \
        --output data/processed/arsl_landmarks.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pose_detector import PoseDetector  # noqa: E402
from src.languages import get_language  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", default="asl", help="Sign language code (asl, arsl)")
    ap.add_argument("--input", default=None,
                    help="Raw data dir (default: data/raw/<language>)")
    ap.add_argument("--output", default=None,
                    help="Output .npz (default: data/processed/<language>_landmarks.npz)")
    args = ap.parse_args()

    language = get_language(args.language)
    root = Path(args.input) if args.input else Path("data/raw") / language.code
    out = Path(args.output) if args.output else Path("data/processed") / f"{language.code}_landmarks.npz"

    if not root.exists():
        print(f"❌ Raw data dir not found: {root}")
        print("   Collect samples first with scripts/collect_data.py")
        sys.exit(1)

    labels = language.labels
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    detector = PoseDetector()

    X, y = [], []
    print(f"🖐️  Building {language.name} dataset from {root}\n")
    for label in labels:
        label_dir = root / label
        if not label_dir.is_dir():
            continue
        count = 0
        for jf in sorted(label_dir.glob("*.json")):
            try:
                landmarks = np.array(json.load(open(jf)))
            except Exception:
                continue
            if landmarks.shape != (21, 3):
                continue
            X.append(detector.normalize_landmarks(landmarks))
            y.append(label_to_idx[label])
            count += 1
        if count:
            print(f"  {label:>7}: {count}")

    detector.close()

    if not X:
        print("❌ No valid samples found — check the raw data folders.")
        sys.exit(1)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, labels=np.array(labels), language=language.code)

    print(f"\n✅ Saved {len(X)} samples to {out}")


if __name__ == "__main__":
    main()
