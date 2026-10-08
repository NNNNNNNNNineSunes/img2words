"""Integer pixel geometry helpers.

The only public primitives are :class:`Rect` and :func:`normalize_rect`.
Everything else in :mod:`screenocr` builds on them.

Conventions
-----------
* ``right`` and ``bottom`` are **exclusive** edges, matching PIL crop boxes,
  ``mss`` monitor dictionaries (``left + width``) and screen coordinates in
  physical pixels.  ``width == right - left``.
* All coordinates are physical screen pixels.  Call
  :func:`screenocr.capture.enable_dpi_awareness` before interpreting them.

This module has no third-party dependencies and is safe to import anywhere.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

__all__ = ["Rect", "normalize_rect"]


def _coord(value: Any) -> int:
    """Coerce *value* to ``int``, accepting any integral type.

    Accepts ``int``, anything implementing ``__index__`` (including NumPy
    integers) and finite floats (rounded).  Booleans, NaN/inf and non-numeric
    values raise ``TypeError``/``ValueError`` instead of silently producing a
    bogus coordinate.
    """
    if isinstance(value, bool):
        raise TypeError("coordinates must be numbers, not bool")
    if isinstance(value, int):
        return value
    try:
        return operator.index(value)
    except TypeError:
        pass
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"coordinate must be finite, got {value!r}")
        return int(round(value))
    raise TypeError(
        f"coordinate must be an integer, got {type(value).__name__}"
    )


def _point(value: Any, label: str) -> Tuple[int, int]:
    """Coerce a 2-item sequence into an ``(x, y)`` integer pair."""
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a pair of coordinates, got {type(value).__name__}")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(
            f"{label} must be a pair of coordinates, got {type(value).__name__}"
        ) from exc
    if len(items) != 2:
        raise TypeError(f"{label} must have exactly 2 items, got {len(items)}")
    return _coord(items[0]), _coord(items[1])


@dataclass(frozen=True)
class Rect:
    """An immutable pixel rectangle with exclusive ``right``/``bottom`` edges."""

    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        for name in ("left", "top", "right", "bottom"):
            object.__setattr__(self, name, _coord(getattr(self, name)))

    # -- derived values ---------------------------------------------------
    @property
    def width(self) -> int:
        """Number of pixels covered horizontally (``right - left``)."""
        return self.right - self.left

    @property
    def height(self) -> int:
        """Number of pixels covered vertically (``bottom - top``)."""
        return self.bottom - self.top

    @property
    def is_empty(self) -> bool:
        """``True`` when the rectangle covers no pixel at all."""
        return self.width <= 0 or self.height <= 0

    @property
    def area(self) -> int:
        """Number of pixels covered (``0`` for empty rectangles)."""
        if self.is_empty:
            return 0
        return self.width * self.height

    @property
    def center_x(self) -> float:
        """Horizontal centre."""
        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        """Vertical centre."""
        return (self.top + self.bottom) / 2.0

    # -- conversions ------------------------------------------------------
    def as_tuple(self) -> Tuple[int, int, int, int]:
        """``(left, top, right, bottom)``."""
        return (self.left, self.top, self.right, self.bottom)

    def as_box(self) -> Tuple[int, int, int, int]:
        """PIL ``crop``/``Image.getbbox`` compatible box (identical layout)."""
        return (self.left, self.top, self.right, self.bottom)

    def as_region(self) -> dict:
        """``mss`` region dictionary (``left``/``top``/``width``/``height``)."""
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }

    # -- operations -------------------------------------------------------
    def clip(self, bounds: "Rect") -> Optional["Rect"]:
        """Intersect with *bounds*.

        Returns ``None`` when the intersection covers no pixel.  *bounds* may
        itself be empty, in which case ``None`` is returned.
        """
        if not isinstance(bounds, Rect):
            raise TypeError(f"bounds must be a Rect, got {type(bounds).__name__}")
        left = max(self.left, bounds.left)
        top = max(self.top, bounds.top)
        right = min(self.right, bounds.right)
        bottom = min(self.bottom, bounds.bottom)
        if right <= left or bottom <= top:
            return None
        return Rect(left, top, right, bottom)


def normalize_rect(
    start: Tuple[int, int],
    end: Tuple[int, int],
    bounds: Optional[Rect] = None,
    min_size: Optional[int] = 5,
) -> Optional[Rect]:
    """Turn a drag gesture into a rectangle.

    ``start``/``end`` are the mouse-down and mouse-up points in physical
    pixels; any drag direction (up-left, down-right, ...) produces the same
    normalised rectangle.  The result is clipped to ``bounds`` when given.

    Returns ``None`` when the gesture is unusable:

    * it covers no pixel (a click without movement),
    * it is smaller than ``min_size`` pixels in either dimension,
    * it lies entirely outside ``bounds`` (or edge-clips down to nothing).

    ``min_size=None`` (or any value below ``1``) still rejects empty
    rectangles; ``min_size=1`` accepts a single pixel.
    """
    start_x, start_y = _point(start, "start")
    end_x, end_y = _point(end, "end")

    left, right = (start_x, end_x) if start_x <= end_x else (end_x, start_x)
    top, bottom = (start_y, end_y) if start_y <= end_y else (end_y, start_y)
    rect = Rect(left, top, right, bottom)

    if bounds is not None:
        clipped = rect.clip(bounds)
        if clipped is None:
            return None
        rect = clipped

    needed = 1 if min_size is None else max(1, _coord(min_size))
    if rect.width < needed or rect.height < needed:
        return None
    return rect
