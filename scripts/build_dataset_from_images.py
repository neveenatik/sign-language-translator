#!/usr/bin/env python3
"""Build a landmark dataset (X, y) from a folder of ASL alphabet images.

Runs MediaPipe Hands over each image, extracts the 63-dim normalized landmark
vector used by the live app (``PoseDetector.normalize_landmarks``), and saves
them to a compressed ``.npz``.

Expected input layout (e.g. the Kaggle "ASL Alphabet" dataset by grassknoted):

    <input>/A/*.jpg
    <input>/B/*.jpg
    ...
    <input>/del/*.jpg      -> label "delete"
    <input>/space/*.jpg    -> label "space"
    <input>/nothing/*.jpg  -> skipped

Usage:
    python scripts/build_dataset_from_images.py \
        --input path/to/asl_alphabet_train \
        --output data/processed/landmarks.npz \
        --per-class 400
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gesture_recognizer import GestureRecognizer  # noqa: E402
from src.pose_detector import PoseDetector  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
# Folder-name aliases -> canonical label (None means "skip this folder").
FOLDER_ALIASES = {"del": "delete", "space": "space", "nothing": None}


def resolve_label(folder_name: str, valid: set) -> str | None:
    """Map a dataset folder name to a canonical ASL label, or None to skip."""
    low = folder_name.strip().lower()
    if low in FOLDER_ALIASES:
        return FOLDER_ALIASES[low]
    up = folder_name.strip().upper()
    if up in valid:
        return up
    return None


def find_class_dirs(root: Path, valid: set):
    """Return [(dir, label)] for class folders, descending one level if needed."""
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    resolved = [(d, resolve_label(d.name, valid)) for d in dirs]
    if any(lbl for _, lbl in resolved):
        return resolved
    # Dataset may be wrapped in an extra folder (e.g. asl_alphabet_train/asl_alphabet_train/).
    for d in dirs:
        sub = find_class_dirs(d, valid)
        if any(lbl for _, lbl in sub):
            return sub
    return resolved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Folder containing per-letter image subfolders")
    ap.add_argument("--output", default="data/processed/landmarks.npz")
    ap.add_argument("--per-class", type=int, default=0, help="Max images per class (0 = all)")
    args = ap.parse_args()

    root = Path(args.input).expanduser()
    if not root.exists():
        print(f"❌ Input path not found: {root}")
        sys.exit(1)

    labels = GestureRecognizer().get_labels()
    valid = set(labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    class_dirs = [(d, lbl) for d, lbl in find_class_dirs(root, valid) if lbl]
    if not class_dirs:
        print(f"❌ No class folders matching ASL labels were found under {root}")
        sys.exit(1)

    print(f"🖐️  Extracting hand landmarks from {len(class_dirs)} classes...\n")
    detector = PoseDetector(static_image_mode=True, max_num_hands=1)

    X, y = [], []
    total_imgs = total_hits = 0
    t0 = time.time()

    for d, lbl in class_dirs:
        imgs = [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXT]
        if args.per_class > 0:
            imgs = imgs[: args.per_class]
        hits = 0
        for p in imgs:
            total_imgs += 1
            img = cv2.imread(str(p))
            if img is None:
                continue
            _, hands = detector.detect(img)
            if not hands:
                continue
            X.append(detector.normalize_landmarks(hands[0]))
            y.append(label_to_idx[lbl])
            hits += 1
        total_hits += hits
        print(f"  {lbl:>7}: {hits:>5}/{len(imgs)} hands detected")

    detector.close()

    if not X:
        print("❌ No hands detected in any image — check the dataset path/contents.")
        sys.exit(1)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, labels=np.array(labels))

    dt = time.time() - t0
    print(f"\n✅ Saved {len(X)} samples ({total_hits}/{total_imgs} images had a "
          f"detectable hand) to {out} in {dt:.0f}s")
    uniq, cnts = np.unique(y, return_counts=True)
    print("   Class distribution:")
    for i, c in zip(uniq, cnts):
        print(f"     {labels[i]:>7}: {c}")


if __name__ == "__main__":
    main()
