"""
Prepare collected gesture data for training.
Normalizes and validates the collected data.
"""

import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pose_detector import PoseDetector


def prepare_data():
    """Prepare and validate collected gesture data."""
    
    data_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 Preparing gesture data...\n")
    
    detector = PoseDetector()
    stats = {}
    
    for gesture_folder in sorted(data_dir.iterdir()):
        if not gesture_folder.is_dir():
            continue
        
        gesture = gesture_folder.name
        json_files = list(gesture_folder.glob("*.json"))
        
        if not json_files:
            continue
        
        print(f"Processing {gesture}: ", end="")
        
        valid_count = 0
        invalid_count = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    landmarks = np.array(json.load(f))
                
                # Validate landmarks shape
                if landmarks.shape != (21, 3):
                    invalid_count += 1
                    continue
                
                # Normalize and save
                normalized = detector.normalize_landmarks(landmarks)
                
                output_file = processed_dir / f"{gesture}_{valid_count:03d}.npz"
                np.savez(output_file, landmarks=normalized)
                
                valid_count += 1
            except Exception as e:
                invalid_count += 1
        
        stats[gesture] = {"valid": valid_count, "invalid": invalid_count}
        print(f"✓ ({valid_count} valid, {invalid_count} invalid)")
    
    detector.close()
    
    print("\n📈 Summary:")
    total_valid = sum(s["valid"] for s in stats.values())
    total_invalid = sum(s["invalid"] for s in stats.values())
    print(f"   Total valid: {total_valid}")
    print(f"   Total invalid: {total_invalid}")
    print(f"\n✅ Data saved to: {processed_dir}")


if __name__ == "__main__":
    prepare_data()
