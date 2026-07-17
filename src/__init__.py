"""
__init__.py for the sign language translator package.

Imports are lazy (PEP 562) so that pulling in a single component doesn't drag
in unrelated heavy dependencies. In particular, training only needs
``GestureRecognizer`` (TensorFlow) and must not require MediaPipe/OpenCV, which
``PoseDetector`` imports.
"""

__version__ = "0.1.0"
__author__ = "Neveen Atik"

__all__ = [
    "PoseDetector",
    "GestureRecognizer",
    "SmoothingFilter",
    "TextBuffer",
]

# Map exported name -> submodule that defines it, resolved on first access.
_LAZY = {
    "PoseDetector": "pose_detector",
    "GestureRecognizer": "gesture_recognizer",
    "SmoothingFilter": "utils",
    "TextBuffer": "utils",
}


def __getattr__(name):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)
