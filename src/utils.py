"""
Utility functions for the ASL translator.
"""

import numpy as np
import cv2
from collections import deque
from typing import Deque, List, Dict


class SmoothingFilter:
    """
    Applies temporal smoothing to predictions to reduce noise.
    Uses a sliding window approach.
    """

    def __init__(self, window_size: int = 5):
        """
        Initialize the smoothing filter.
        
        Args:
            window_size: Size of the sliding window for smoothing
        """
        self.window_size = window_size
        self.predictions_history: Deque = deque(maxlen=window_size)

    def smooth_prediction(self, prediction: Dict) -> Dict:
        """
        Apply temporal smoothing to a prediction.
        
        Args:
            prediction: Current prediction dict
            
        Returns:
            Smoothed prediction or None if not enough history
        """
        if prediction:
            self.predictions_history.append(prediction['label'])
        
        if len(self.predictions_history) < self.window_size:
            return None
        
        # Get most common label in the window
        labels = list(self.predictions_history)
        from collections import Counter
        most_common = Counter(labels).most_common(1)[0][0]
        
        return {'label': most_common}

    def reset(self):
        """Reset the smoothing history."""
        self.predictions_history.clear()


class TextBuffer:
    """
    Manages translated text output with support for corrections.
    """

    def __init__(self, max_length: int = 500):
        """
        Initialize the text buffer.
        
        Args:
            max_length: Maximum text length before trimming
        """
        self.text = ""
        self.max_length = max_length

    def add_character(self, char: str) -> str:
        """
        Add a character to the buffer.
        
        Args:
            char: Character to add
            
        Returns:
            Updated text
        """
        if char == 'delete':
            self.text = self.text[:-1]
        elif char == 'space':
            self.text += ' '
        else:
            self.text += char
        
        # Trim if too long
        if len(self.text) > self.max_length:
            self.text = self.text[-self.max_length:]
        
        return self.text

    def get_text(self) -> str:
        """Get current text."""
        return self.text

    def clear(self):
        """Clear the buffer."""
        self.text = ""

    def delete_last_word(self):
        """Delete the last word from the buffer."""
        words = self.text.rsplit(' ', 1)
        self.text = words[0] if len(words) > 1 else ""


def display_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """
    Display FPS counter on frame.
    
    Args:
        frame: Input frame
        fps: Current FPS value
        
    Returns:
        Frame with FPS displayed
    """
    cv2.putText(
        frame,
        f'FPS: {fps:.1f}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    return frame


def display_gesture(frame: np.ndarray, gesture: str, confidence: float) -> np.ndarray:
    """
    Display recognized gesture on frame.
    
    Args:
        frame: Input frame
        gesture: Recognized gesture label
        confidence: Confidence score (0-1)
        
    Returns:
        Frame with gesture info displayed
    """
    text = f'{gesture} ({confidence:.2f})'
    cv2.putText(
        frame,
        text,
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    return frame


def resize_frame(frame: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    """
    Resize frame to target dimensions.
    
    Args:
        frame: Input frame
        width: Target width
        height: Target height
        
    Returns:
        Resized frame
    """
    return cv2.resize(frame, (width, height))


def get_frame_info(frame: np.ndarray) -> Dict:
    """
    Get information about a frame.
    
    Args:
        frame: Input frame
        
    Returns:
        Dict with frame info
    """
    h, w = frame.shape[:2]
    return {
        'width': w,
        'height': h,
        'shape': frame.shape
    }
