"""
Hand pose detection module using MediaPipe.
Detects and extracts hand landmarks from video frames.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List


class PoseDetector:
    """
    Detects hand landmarks using MediaPipe Hands solution.
    Extracts normalized coordinates and hand presence information.
    """

    def __init__(self, static_image_mode: bool = False, max_num_hands: int = 2):
        """
        Initialize the hand detector.
        
        Args:
            static_image_mode: Whether to treat frames as static images
            max_num_hands: Maximum number of hands to detect (1 or 2)
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.landmarks_history = []

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Detect hand landmarks in a frame.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Tuple of (annotated_frame, hand_landmarks)
            - annotated_frame: Frame with drawn landmarks
            - hand_landmarks: List of landmark arrays or None
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        annotated_frame = frame.copy()
        hand_landmarks_list = None

        if results.multi_hand_landmarks:
            hand_landmarks_list = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on frame
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Extract normalized coordinates (0-1 range)
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                ])
                hand_landmarks_list.append(landmarks)

            self.landmarks_history.append(hand_landmarks_list)

        return annotated_frame, hand_landmarks_list

    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize hand landmarks by centering and scaling.
        
        Args:
            landmarks: Hand landmarks (21, 3) array
            
        Returns:
            Normalized landmarks (63,) flattened array
        """
        # Center at wrist (landmark 0)
        centered = landmarks - landmarks[0]
        
        # Scale to 0-1 range
        max_dist = np.max(np.linalg.norm(centered, axis=1))
        if max_dist > 0:
            normalized = centered / max_dist
        else:
            normalized = centered
            
        return normalized.flatten()

    def get_hand_bounding_box(self, landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get bounding box coordinates for hand landmarks.
        
        Args:
            landmarks: Hand landmarks (21, 3) array
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        return (
            int(np.min(x_coords) * 1000),
            int(np.min(y_coords) * 1000),
            int(np.max(x_coords) * 1000),
            int(np.max(y_coords) * 1000)
        )

    def close(self):
        """Clean up resources."""
        self.hands.close()
