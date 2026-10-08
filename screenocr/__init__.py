"""screenocr - screen region OCR to clipboard.

Public API
----------
Geometry
    :class:`~screenocr.geometry.Rect`,
    :func:`~screenocr.geometry.normalize_rect`
Capture
    :func:`~screenocr.capture.enable_dpi_awareness`,
    :func:`~screenocr.capture.capture_screen`,
    :func:`~screenocr.capture.capture_region`,
    :func:`~screenocr.capture.virtual_screen_rect`
OCR
    :class:`~screenocr.ocr.OCREngine`, :func:`~screenocr.ocr.items_to_text`
Clipboard
    :func:`~screenocr.clipboard.write_text`, :func:`~screenocr.clipboard.read_text`
Pipeline
    :func:`~screenocr.pipeline.recognize_and_copy`

Importing this package is cheap and platform neutral: ``mss``, Pillow,
``rapidocr-onnxruntime``/``onnxruntime`` and the Windows DLLs are all loaded
lazily, on first use.  The first ``OCREngine.recognize`` call loads the ONNX
models, so warm the engine up at application start.
"""

from __future__ import annotations

from .capture import (
    CaptureError,
    capture_region,
    capture_screen,
    enable_dpi_awareness,
    virtual_screen_rect,
)
from .clipboard import ClipboardError, is_available, read_text, write_text
from .geometry import Rect, normalize_rect
from .ocr import OCREngine, OCRError, group_lines, items_to_text, render_lines
from .pipeline import NoTextError, recognize_and_copy

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # geometry
    "Rect",
    "normalize_rect",
    # capture
    "CaptureError",
    "capture_region",
    "capture_screen",
    "enable_dpi_awareness",
    "virtual_screen_rect",
    # ocr
    "OCREngine",
    "OCRError",
    "group_lines",
    "items_to_text",
    "render_lines",
    # clipboard
    "ClipboardError",
    "is_available",
    "read_text",
    "write_text",
    # pipeline
    "NoTextError",
    "recognize_and_copy",
]
