"""Screen capture in physical pixels (Windows DPI aware, powered by ``mss``).

``enable_dpi_awareness`` makes the process per-monitor-DPI aware (v2 when
available) so that a DPI unaware process does not receive a virtualised,
blurry 1707x960 view of a 2560x1440 desktop.  It is best effort: every failure
path returns ``False`` instead of raising, because awareness can only be set
once per process and may legitimately be denied.

``mss`` and Pillow are imported inside the functions, so importing this module
is free and safe on any platform.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from typing import TYPE_CHECKING, Any, Optional, Tuple

from .geometry import Rect

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image as PILImage

__all__ = [
    "CaptureError",
    "capture_region",
    "capture_screen",
    "enable_dpi_awareness",
    "virtual_screen_rect",
]

_WIN32 = sys.platform == "win32"

#: ``DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2`` (Windows 10 1703+).
_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)
#: ``PROCESS_PER_MONITOR_DPI_AWARE`` for the Windows 8.1 ``shcore`` API.
_PROCESS_PER_MONITOR_DPI_AWARE = 2
_ERROR_ACCESS_DENIED = 5
_E_ACCESSDENIED = -2147024891  # 0x80070005 as a signed HRESULT

_awareness_lock = threading.Lock()
_awareness_done = False
_awareness_result = False


class CaptureError(RuntimeError):
    """Raised when the screen (or the requested region) cannot be captured."""


def _load_library(name: str) -> Optional[Any]:
    if not _WIN32:
        return None
    try:
        return ctypes.WinDLL(name, use_last_error=True)
    except Exception:  # pragma: no cover - depends on the host system
        return None


def _apply_dpi_awareness() -> bool:
    """Try every documented DPI awareness API, newest first."""
    if not _WIN32:
        return False

    # 1. Windows 10 1703+: per monitor v2 (SetProcessDpiAwarenessContext).
    user32 = _load_library("user32")
    if user32 is not None:
        try:
            func = getattr(user32, "SetProcessDpiAwarenessContext", None)
            if func is not None:
                func.argtypes = [ctypes.c_void_p]
                func.restype = ctypes.c_int
                if func(_PER_MONITOR_AWARE_V2):
                    return True
                if ctypes.get_last_error() == _ERROR_ACCESS_DENIED:
                    # Awareness was already set for this process.
                    return True
        except Exception:
            pass

    # 2. Windows 8.1+: shcore.SetProcessDpiAwareness (per monitor v1).
    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        func = shcore.SetProcessDpiAwareness
        func.argtypes = [ctypes.c_int]
        func.restype = ctypes.c_long  # HRESULT
        result = func(_PROCESS_PER_MONITOR_DPI_AWARE)
        if result == 0 or result == _E_ACCESSDENIED:
            return True
    except Exception:
        pass

    # 3. Vista+: user32.SetProcessDPIAware (system DPI aware, last resort).
    if user32 is not None:
        try:
            func = user32.SetProcessDPIAware
            func.argtypes = []
            func.restype = ctypes.c_int
            if func():
                return True
        except Exception:
            pass
    return False


def enable_dpi_awareness(*, force: bool = False) -> bool:
    """Make this process DPI aware so captures use physical pixels.

    Returns ``True`` when the process is (or already was) per-monitor aware.
    Safe to call from anywhere, repeatedly, on any platform: the first call
    does the work, later calls return the cached result.
    """
    global _awareness_done, _awareness_result
    if not _WIN32:
        return False
    with _awareness_lock:
        if _awareness_done and not force:
            return _awareness_result
        try:
            _awareness_result = bool(_apply_dpi_awareness())
        except Exception:  # pragma: no cover - defensive
            _awareness_result = False
        _awareness_done = True
        return _awareness_result


def _load_mss() -> Any:
    try:
        import mss
    except Exception as exc:  # pragma: no cover - depends on the env
        raise CaptureError(
            "the 'mss' package is required for screen capture; "
            "install the pinned runtime with: pip install -r requirements.txt"
        ) from exc
    return mss


def _monitor_to_rect(monitor: Any) -> Rect:
    left = int(monitor["left"])
    top = int(monitor["top"])
    return Rect(left, top, left + int(monitor["width"]), top + int(monitor["height"]))


def _new_session(mss_module: Any) -> Any:
    """Create an mss session using the modern factory when available.

    ``mss.mss()`` is deprecated since mss 10.2 in favour of ``mss.MSS``; older
    mss releases only expose the function, so both are supported.
    """
    factory = getattr(mss_module, "MSS", None)
    if factory is not None:
        try:
            return factory()
        except TypeError:  # pragma: no cover - abstract base in older mss
            pass
    return mss_module.mss()


def _to_image(shot: Any) -> "PILImage":
    """PIL RGB image from an mss screen shot.

    ``shot.rgb`` is the tightly packed 3-byte-per-pixel buffer mss provides, so
    it maps directly onto PIL's ``RGB`` mode (no BGRX rawmode juggling).
    """
    from PIL import Image

    return Image.frombytes("RGB", (int(shot.width), int(shot.height)), shot.rgb)


def _coerce_rect(rect: Any) -> Rect:
    """Accept a :class:`Rect` or a 4 item sequence and normalise the ordering."""
    if isinstance(rect, Rect):
        candidate = rect
    elif isinstance(rect, (tuple, list)) and len(rect) == 4:
        candidate = Rect(rect[0], rect[1], rect[2], rect[3])
    else:
        raise TypeError(
            "rect must be a screenocr.geometry.Rect or a 4 item sequence, "
            f"got {type(rect).__name__}"
        )
    left, right = sorted((candidate.left, candidate.right))
    top, bottom = sorted((candidate.top, candidate.bottom))
    return Rect(left, top, right, bottom)


def virtual_screen_rect() -> Rect:
    """Bounding box of every monitor, in physical pixels."""
    mss = _load_mss()
    try:
        with _new_session(mss) as session:
            if not session.monitors:
                raise CaptureError("mss reported no monitors")
            return _monitor_to_rect(session.monitors[0])
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"failed to read the virtual desktop bounds: {exc}") from exc


def capture_screen() -> Tuple["PILImage", Rect]:
    """Capture the whole virtual desktop.

    Returns ``(image, rect)`` where *image* is an RGB ``PIL.Image.Image`` and
    *rect* is its position in physical desktop pixels.
    """
    enable_dpi_awareness()
    mss = _load_mss()
    try:
        with _new_session(mss) as session:
            if not session.monitors:
                raise CaptureError("mss reported no monitors")
            monitor = session.monitors[0]
            rect = _monitor_to_rect(monitor)
            shot = session.grab(
                {
                    "left": rect.left,
                    "top": rect.top,
                    "width": rect.width,
                    "height": rect.height,
                }
            )
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"failed to capture the screen: {exc}") from exc
    return _to_image(shot), rect


def capture_region(rect: Any) -> "PILImage":
    """Capture *rect* (physical desktop pixels) as an RGB image.

    The rectangle is normalised (reversed edges are fixed) and clamped to the
    virtual desktop.  Raises :class:`CaptureError` for an empty rectangle or
    one that does not overlap the desktop at all.
    """
    target = _coerce_rect(rect)
    if target.is_empty:
        raise CaptureError(f"cannot capture an empty region: {target.as_tuple()}")

    enable_dpi_awareness()
    mss = _load_mss()
    try:
        with _new_session(mss) as session:
            if not session.monitors:
                raise CaptureError("mss reported no monitors")
            desktop = _monitor_to_rect(session.monitors[0])
            clipped = target.clip(desktop)
            if clipped is None:
                raise CaptureError(
                    f"region {target.as_tuple()} does not overlap the virtual "
                    f"desktop {desktop.as_tuple()}"
                )
            shot = session.grab(clipped.as_region())
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(
            f"failed to capture region {target.as_tuple()}: {exc}"
        ) from exc
    return _to_image(shot)
