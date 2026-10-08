"""OCR engine wrapper around ``rapidocr-onnxruntime`` (fixed on 1.4.4).

* The engine and its ONNX models are created lazily on the first
  :meth:`OCREngine.recognize` call, so importing this module never touches
  ``onnxruntime``.
* ``intra_op_num_threads`` is capped so recognising a capture does not stall
  the GUI.
* Recognised boxes are grouped back into readable lines: row breaks survive,
  Chinese fragments stay unspaced and Latin words keep their spaces.

Contract with the pinned ``rapidocr-onnxruntime==1.4.4`` (verified against the
installed package, not guessed):

* ``RapidOCR(**kwargs)`` merges bare keyword names into the ``Global`` config
  and copies ``intra_op_num_threads``/``inter_op_num_threads`` into the
  Det/Cls/Rec sessions.  Values outside ``1..cpu_count()`` are silently
  ignored, hence the clamping below.
* ``engine(image) -> (result, elapse)`` where ``result`` is
  ``[[box, text, score], ...]`` or ``(None, None)`` when no text was found.
* ``LoadImage`` accepts a ``PIL.Image.Image`` directly (converting RGB to BGR
  internally), so a captured image is handed over without a copy.
"""

from __future__ import annotations

import os
import re
import threading
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple

from .geometry import Rect

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image as PILImage

__all__ = ["OCRError", "OCREngine", "group_lines", "items_to_text", "render_lines"]

#: Minimum confidence accepted by RapidOCR (library default).
DEFAULT_TEXT_SCORE = 0.5
#: Per-session ONNX CPU thread cap used when ``num_threads`` is not given.
DEFAULT_MAX_THREADS = 4

Line = List[Tuple[str, Rect]]

_WHITESPACE = re.compile(r"\s+")
#: Zero-width characters that would otherwise corrupt spacing decisions.
_ZERO_WIDTH = {0x00AD: None, 0x200B: None, 0x200C: None, 0x200D: None, 0xFEFF: None}


class OCRError(RuntimeError):
    """Raised when the OCR engine cannot be loaded or fails to recognise."""


# ---------------------------------------------------------------------------
# layout: boxes -> readable text
# ---------------------------------------------------------------------------
def _is_cjk(char: str) -> bool:
    """``True`` for CJK ideographs, kana, hangul and CJK punctuation."""
    code = ord(char)
    return (
        0x2E80 <= code <= 0x9FFF
        or 0xAC00 <= code <= 0xD7AF
        or 0xF900 <= code <= 0xFAFF
        or 0xFE30 <= code <= 0xFE4F
        or 0xFF00 <= code <= 0xFFEF
        or 0x20000 <= code <= 0x3FFFF
    )


def _clean(text: str) -> str:
    """Collapse whitespace inside one box and drop zero-width characters."""
    return _WHITESPACE.sub(" ", text.translate(_ZERO_WIDTH)).strip()


def _same_line(a: Rect, b: Rect) -> bool:
    """``True`` when two boxes sit on the same visual row."""
    shorter = max(1, min(a.height, b.height))
    overlap = min(a.bottom, b.bottom) - max(a.top, b.top)
    if overlap * 2 >= shorter:  # at least half of the shorter box overlaps
        return True
    return abs(a.center_y - b.center_y) * 2 <= shorter  # centres half a box apart


def _needs_space(previous: str, upcoming: str, gap: float, height: float) -> bool:
    """Decide whether two fragments on one line must be space separated."""
    previous_cjk = _is_cjk(previous)
    upcoming_cjk = _is_cjk(upcoming)
    if previous_cjk and upcoming_cjk:
        # Chinese continues without a space unless there is a real visual gap
        # (a table column, a label followed by its value).
        return gap > 0.75 * height
    if previous_cjk != upcoming_cjk:
        return True  # "中文English" reads better as "中文 English"
    return gap > 0.18 * height


def group_lines(items: Sequence[Tuple[str, Rect]]) -> List[Line]:
    """Cluster ``(text, rect)`` pairs into lines in reading order.

    Lines are ordered top to bottom and each line left to right, so row breaks
    in the source image are preserved.
    """
    ordered = sorted(
        items, key=lambda item: (item[1].top, item[1].left, item[1].bottom, item[0])
    )
    lines: List[Line] = []
    for text, rect in ordered:
        for line in lines:
            if any(_same_line(rect, member) for _, member in line):
                line.append((text, rect))
                break
        else:
            lines.append([(text, rect)])

    lines.sort(key=lambda line: (min(r.top for _, r in line), min(r.left for _, r in line)))
    for line in lines:
        line.sort(key=lambda item: (item[1].left, item[1].top))
    return lines


def render_lines(lines: Sequence[Line]) -> str:
    """Render grouped lines, one row per line (``\\n`` separated)."""
    rendered: List[str] = []
    for line in lines:
        buffer = ""
        previous: Optional[Rect] = None
        for text, rect in sorted(line, key=lambda item: (item[1].left, item[1].top)):
            segment = _clean(text)
            if not segment:
                continue
            if buffer and previous is not None:
                gap = float(rect.left - previous.right)
                height = max(1.0, 0.5 * (rect.height + previous.height))
                if _needs_space(buffer[-1], segment[0], gap, height):
                    buffer += " "
            buffer += segment
            previous = rect
        if buffer.strip():
            rendered.append(buffer.strip())
    return "\n".join(rendered)


def items_to_text(items: Sequence[Tuple[str, Rect]]) -> str:
    """Group and render ``(text, rect)`` pairs into readable text."""
    return render_lines(group_lines(items))


# ---------------------------------------------------------------------------
# engine payload -> (text, rect) records
# ---------------------------------------------------------------------------
def _rect_of_box(box: Any) -> Optional[Rect]:
    """Axis aligned :class:`Rect` around a 4 point polygon, or ``None``."""
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
    except Exception:
        return None
    if not xs:
        return None
    return Rect(
        int(round(min(xs))), int(round(min(ys))),
        int(round(max(xs))), int(round(max(ys))),
    )


def _records(payload: Any) -> List[Tuple[str, Rect]]:
    """Normalise ``(result, elapse)`` / ``(None, None)`` into records.

    Items whose box cannot be parsed or whose text is not a string are
    skipped, so a surprising payload degrades to less text, never a crash.
    """
    if isinstance(payload, tuple) and len(payload) == 2:
        payload = payload[0]  # drop the elapsed-time list
    records: List[Tuple[str, Rect]] = []
    for item in payload or ():
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        text, rect = item[1], _rect_of_box(item[0])
        if isinstance(text, str) and rect is not None:
            records.append((text, rect))
    return records


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
def _resolve_threads(num_threads: Optional[int]) -> int:
    """Clamp the requested ONNX thread count into ``1..cpu_count()``."""
    cpu_count = os.cpu_count() or 1
    if num_threads is None:
        return max(1, min(DEFAULT_MAX_THREADS, cpu_count))
    if isinstance(num_threads, bool) or not isinstance(num_threads, int):
        try:
            num_threads = int(num_threads)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"num_threads must be an integer, got {type(num_threads).__name__}"
            ) from exc
    if num_threads < 1:
        raise ValueError(f"num_threads must be >= 1, got {num_threads}")
    return max(1, min(num_threads, cpu_count))


class OCREngine:
    """Lazy wrapper around ``rapidocr_onnxruntime.RapidOCR``.

    Engine creation is guarded by a lock and happens at most once.  Recognition
    itself is *not* serialised: call :meth:`recognize` from one thread at a
    time (the GUI does).

    ``engine_factory`` exists for tests: it is called instead of constructing
    the real RapidOCR engine, so none of the ONNX models are loaded.
    """

    def __init__(
        self,
        text_score: float = DEFAULT_TEXT_SCORE,
        *,
        num_threads: Optional[int] = None,
        engine_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        try:
            text_score = float(text_score)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"text_score must be a number, got {type(text_score).__name__}"
            ) from exc
        if not 0.0 <= text_score <= 1.0:
            raise ValueError(f"text_score must be within 0..1, got {text_score}")
        if engine_factory is not None and not callable(engine_factory):
            raise TypeError("engine_factory must be callable")

        self.text_score = text_score
        self.num_threads = _resolve_threads(num_threads)
        self._engine_factory = engine_factory
        self._engine: Optional[Any] = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """``True`` once the underlying RapidOCR engine exists."""
        return self._engine is not None

    def load(self) -> Any:
        """Create (once) and return the underlying engine."""
        engine = self._engine
        if engine is None:
            with self._lock:
                if self._engine is None:
                    self._engine = self._create_engine()
                engine = self._engine
        return engine

    def warmup(self) -> None:
        """Load the models now instead of during the first recognition."""
        self.load()

    def _create_engine(self) -> Any:
        if self._engine_factory is not None:
            try:
                engine = self._engine_factory()
            except Exception as exc:
                raise OCRError(f"failed to create the OCR engine: {exc}") from exc
            if engine is None:
                raise OCRError("the OCR engine factory returned None")
            return engine

        # Cap OpenMP too: it is consulted while the ONNX sessions start up.
        os.environ.setdefault("OMP_NUM_THREADS", str(self.num_threads))
        try:
            from rapidocr_onnxruntime import RapidOCR
        except Exception as exc:  # pragma: no cover - depends on the env
            raise OCRError(
                "rapidocr-onnxruntime is not importable; install the pinned "
                "runtime with: pip install -r requirements.txt"
            ) from exc
        try:
            return RapidOCR(
                text_score=self.text_score,
                intra_op_num_threads=self.num_threads,
                inter_op_num_threads=1,
            )
        except Exception as exc:
            raise OCRError(f"failed to initialise the RapidOCR engine: {exc}") from exc

    def recognize(self, image: "PILImage") -> str:
        """Recognise *image* and return readable text (``""`` when none).

        Raises :class:`OCRError` when the models cannot be loaded or
        recognition fails.
        """
        engine = self.load()
        try:
            records = _records(engine(image))
        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(f"OCR recognition failed: {exc}") from exc
        return items_to_text(records)
