"""
Utility functions for the ASL translator.
"""

import os
import numpy as np
import cv2
from collections import deque
from typing import Deque, List, Dict, Optional, Tuple

# Bundled font that covers Arabic (and Latin) glyphs, used by the PIL text
# renderer so right-to-left scripts show up on the video overlay. OpenCV's
# built-in Hershey fonts cannot render Arabic.
_ARABIC_FONT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "fonts", "NotoNaskhArabic.ttf",
)
_FONT_CACHE: Dict[int, "object"] = {}


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


def prepare_display_text(text: str, rtl: bool = False) -> str:
    """
    Prepare recognized text for display.

    For right-to-left scripts (e.g. Arabic) this reshapes the letters into
    their connected forms and applies the bidi algorithm so the string renders
    correctly. If the optional `arabic_reshaper` / `python-bidi` packages are
    not installed it falls back to returning the text unchanged.

    Args:
        text: Raw recognized text (sequence of predicted labels).
        rtl: Whether the active language is right-to-left.

    Returns:
        Display-ready text.
    """
    if not rtl or not text:
        return text

    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        # Optional deps missing (or reshaping failed) — show the raw text.
        return text


def _load_font(size: int):
    """Load (and cache) the bundled TrueType font at the given size."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    try:
        from PIL import ImageFont

        if os.path.exists(_ARABIC_FONT_PATH):
            font = ImageFont.truetype(_ARABIC_FONT_PATH, size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = None
    _FONT_CACHE[size] = font
    return font


def draw_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    rtl: bool = False,
    font_size: int = 32,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Draw text onto a BGR frame using PIL so non-Latin scripts (Arabic) render.

    Falls back to `cv2.putText` for Latin text or if PIL/the bundled font is
    unavailable. `position` is the bottom-left baseline (to match cv2.putText).

    Args:
        frame: Input BGR frame (modified via a returned copy).
        text: Text to draw (already display-ordered for RTL if applicable).
        position: (x, y) bottom-left anchor, matching cv2.putText semantics.
        rtl: Whether the text is a right-to-left script.
        font_size: Font pixel size.
        color: BGR color tuple.

    Returns:
        Frame with the text drawn.
    """
    # Latin text renders fine (and faster) with OpenCV's built-in font.
    if not rtl:
        cv2.putText(
            frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2
        )
        return frame

    font = _load_font(font_size)
    if font is None:
        # No PIL/font available — best-effort fallback.
        cv2.putText(
            frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2
        )
        return frame

    from PIL import Image, ImageDraw

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    # cv2 anchors at the baseline/bottom; PIL anchors at the top — shift up.
    x, y = position
    top_y = max(0, y - font_size)
    draw.text((x, top_y), text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


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
