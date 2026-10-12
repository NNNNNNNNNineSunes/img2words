"""Windows clipboard access (``CF_UNICODETEXT``) via ``ctypes``.

The module is import-safe on every platform: the Windows DLLs are only loaded
when a clipboard function is actually called.  All handles are declared with
pointer sized ``restype``/``argtypes`` so 64-bit processes never truncate a
``HGLOBAL``/``HANDLE``.

Only the Unicode text format is read and written; the clipboard is never
touched by :mod:`screenocr` unit tests.
"""

from __future__ import annotations

import ctypes
import sys
import time
from typing import Any, Callable, Optional, Tuple

__all__ = [
    "ClipboardCommitRejected",
    "ClipboardError",
    "is_available",
    "read_text",
    "write_text",
]

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002

#: How many times a busy clipboard is retried before giving up.
DEFAULT_RETRIES = 10
#: Base delay in seconds between clipboard retries.
DEFAULT_RETRY_DELAY = 0.03

_WIN32 = sys.platform == "win32"


class ClipboardError(RuntimeError):
    """Raised when the Windows clipboard cannot be read or written."""


class ClipboardCommitRejected(ClipboardError):
    """Raised when a trusted ``commit_guard`` vetoes the clipboard commit."""


class _Busy(Exception):
    """Internal: the clipboard was momentarily locked by another process."""


_user32: Optional[Any] = None
_kernel32: Optional[Any] = None


def _load_api() -> Tuple[Any, Any]:
    """Load and configure ``user32``/``kernel32`` entry points (cached)."""
    global _user32, _kernel32
    if _user32 is not None and _kernel32 is not None:
        return _user32, _kernel32
    if not _WIN32:
        raise ClipboardError(
            "the clipboard is only supported on Windows "
            f"(current platform: {sys.platform})"
        )

    import ctypes.wintypes as wintypes

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except OSError as exc:  # pragma: no cover - depends on the host system
        raise ClipboardError(f"failed to load the Windows clipboard API: {exc}") from exc

    # Pointer sized signatures: a default `c_int` restype would truncate the
    # 64-bit handle returned by the system.
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
    user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE

    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.restype = wintypes.HGLOBAL

    _user32, _kernel32 = user32, kernel32
    return user32, kernel32


def is_available() -> bool:
    """``True`` when the Windows clipboard API can be loaded."""
    try:
        _load_api()
    except ClipboardError:
        return False
    return True


def _busy_message(attempts: int) -> str:
    return (
        f"could not open the clipboard after {attempts} attempt(s); "
        "another application is holding it"
    )


def _session(
    user32: Any,
    retries: int,
    retry_delay: float,
    action: Callable[[], Any],
) -> Any:
    """Run *action* with the clipboard open, retrying while it stays busy."""
    attempts = max(1, int(retries))
    delay = max(0.0, float(retry_delay))
    reason = "busy"
    for attempt in range(attempts):
        if not user32.OpenClipboard(None):
            if attempt + 1 < attempts:
                time.sleep(delay)
                continue
            raise ClipboardError(_busy_message(attempts))
        try:
            return action()
        except _Busy as exc:
            reason = str(exc)
        finally:
            user32.CloseClipboard()
        if attempt + 1 < attempts:
            time.sleep(delay)
    raise ClipboardError(
        f"the clipboard stayed busy for {attempts} attempt(s) ({reason})"
    )


def read_text(
    *,
    retries: int = DEFAULT_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> str:
    """Return the clipboard's Unicode text.

    Returns ``""`` when the clipboard holds no text (for example an image), so
    callers can treat "no text" and "empty text" alike.  Raises
    :class:`ClipboardError` when the clipboard cannot be opened or locked.
    """
    user32, kernel32 = _load_api()

    def action() -> str:
        if not user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
            return ""
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return ""
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            raise _Busy("GlobalLock failed while reading the clipboard")
        try:
            try:
                return ctypes.wstring_at(pointer)
            except Exception as exc:  # pragma: no cover - defensive
                raise ClipboardError(f"failed to read the clipboard text: {exc}") from exc
        finally:
            kernel32.GlobalUnlock(handle)

    return _session(user32, retries, retry_delay, action)


def write_text(
    text: str,
    *,
    retries: int = DEFAULT_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    commit_guard: Optional[Callable[[], bool]] = None,
) -> None:
    """Replace the clipboard contents with *text* (``CF_UNICODETEXT``).

    The global memory block is allocated and filled *before* the clipboard is
    opened, so an allocation or lock failure leaves the existing clipboard
    contents untouched.  Once the clipboard has been emptied there is no Win32
    way to roll back: if ``SetClipboardData`` then fails, the clipboard stays
    empty and :class:`ClipboardError` is raised.  Ownership of the block moves
    to the system on success, and every failure path frees it.

    A falsey *commit_guard* result raises :class:`ClipboardCommitRejected` before
    the clipboard is emptied; the prepared block is still freed. Exceptions from
    the guard propagate unchanged and are never classified as clipboard-busy.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    if commit_guard is not None and not callable(commit_guard):
        raise TypeError(f"commit_guard must be callable, got {type(commit_guard).__name__}")

    user32, kernel32 = _load_api()

    payload = text.encode("utf-16-le") + b"\x00\x00"
    size = len(payload)
    handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
    if not handle:
        raise ClipboardError("GlobalAlloc failed while preparing the clipboard text")
    guard_error = None
    try:
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            raise ClipboardError("GlobalLock failed while preparing the clipboard text")
        try:
            ctypes.memmove(pointer, payload, size)
        finally:
            kernel32.GlobalUnlock(handle)

        def action() -> None:
            nonlocal guard_error
            if commit_guard is not None:
                try:
                    authorized = bool(commit_guard())
                except BaseException as exc:
                    guard_error = exc
                    raise ClipboardCommitRejected(
                        "clipboard commit guard raised an exception"
                    ) from None
                if not authorized:
                    raise ClipboardCommitRejected("clipboard commit rejected by guard")
            if not user32.EmptyClipboard():
                raise ClipboardError("EmptyClipboard failed")
            if not user32.SetClipboardData(CF_UNICODETEXT, handle):
                raise ClipboardError(
                    "SetClipboardData failed "
                    f"(GetLastError={ctypes.get_last_error()})"
                )

        _session(user32, retries, retry_delay, action)
    except BaseException:
        kernel32.GlobalFree(handle)
        if guard_error is not None:
            raise guard_error
        raise
    # Ownership moved to the clipboard owner (the system); do not free it.
