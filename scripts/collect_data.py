"""
Data collection script for training ASL model.
Records hand landmarks for specified gestures.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
import json
from datetime import datetime
from pathlib import Path


def collect_gesture_data(gesture: str, samples: int = 50):
    """
    Collect hand landmark data for a specific gesture.
    
    Args:
        gesture: Single letter gesture (A-Z) or special gesture (space, delete)
        samples: Number of samples to collect
    """
    # Setup paths
    data_dir = Path("data/raw")
    gesture_dir = data_dir / gesture
    gesture_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access webcam")
        return
    
    collected = 0
    skipped = 0
    
    print(f"\n🎬 Collecting data for gesture: {gesture}")
    print(f"📊 Target samples: {samples}")
    print(f"💾 Saving to: {gesture_dir}")
    print("\nPress SPACE to capture, SKIP for next, Q to quit")
    
    while collected < samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw instructions
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Collected: {collected}/{samples}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing = mp.solutions.drawing_utils
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            cv2.putText(frame, "READY! Press SPACE", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Show your hand...", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("ASL Data Collection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' ') and results.multi_hand_landmarks:
            # Save landmarks
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            
            filename = f"{gesture_dir}/{gesture}_{collected:03d}.json"
            with open(filename, 'w') as f:
                json.dump(landmarks, f)
            
            collected += 1
            print(f"✓ Saved sample {collected}")
        
        elif key == ord('s'):
            skipped += 1
            print(f"⊘ Skipped sample {skipped}")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print(f"\n✅ Collection complete!")
    print(f"📊 Collected: {collected} samples")
    print(f"⊘ Skipped: {skipped} samples")


def main():
    parser = argparse.ArgumentParser(description="Collect ASL gesture data")
    parser.add_argument("--gesture", type=str, required=True,
                       help="Gesture to collect (A-Z, space, delete)")
    parser.add_argument("--samples", type=int, default=50,
                       help="Number of samples to collect")
    
    args = parser.parse_args()
    
    collect_gesture_data(args.gesture, args.samples)


if __name__ == "__main__":
    main()
