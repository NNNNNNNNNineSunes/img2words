"""End-to-end pipeline: recognise an image and copy the text.

The clipboard is only written when recognition produced something: a blank or
unreadable capture raises :class:`NoTextError` and leaves the user's existing
clipboard contents untouched.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Optional

from . import clipboard as _clipboard
from .ocr import OCREngine, OCRError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image as PILImage

__all__ = ["NoTextError", "recognize_and_copy"]

_default_engine: Optional[OCREngine] = None
_default_engine_lock = threading.Lock()


class NoTextError(RuntimeError):
    """Raised when recognition returned no text (clipboard left unchanged)."""


def _get_default_engine() -> OCREngine:
    """Return the process wide engine, creating it on first use."""
    global _default_engine
    engine = _default_engine
    if engine is None:
        with _default_engine_lock:
            if _default_engine is None:
                _default_engine = OCREngine()
            engine = _default_engine
    return engine


def recognize_and_copy(image: "PILImage", engine: Optional[Any] = None) -> str:
    """Recognise *image* and copy the text to the clipboard.

    *engine* defaults to a lazily created, cached :class:`~screenocr.ocr.OCREngine`
    so repeated hotkey invocations do not reload the ONNX models.  Any object
    exposing ``recognize(image) -> str`` is accepted, which keeps the pipeline
    testable without the real models.

    Returns the text that was copied.  Raises :class:`NoTextError` when the
    result is empty (nothing is written) and propagates
    :class:`~screenocr.ocr.OCRError` / :class:`~screenocr.clipboard.ClipboardError`
    from the underlying stages.
    """
    if engine is None:
        engine = _get_default_engine()

    text = engine.recognize(image)
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise OCRError(
            f"the OCR engine returned {type(text).__name__}, expected str"
        )

    text = text.strip()
    if not text:
        raise NoTextError(
            "no text was recognised in the image; the clipboard was left unchanged"
        )

    _clipboard.write_text(text)
    return text
