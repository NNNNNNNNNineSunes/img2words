"""Pure recognition-mode vocabulary and modifier selection."""

from enum import Enum


class RecognitionMode(str, Enum):
    """Supported output modes for recognition."""

    TEXT = "text"
    INLINE = "formula_inline"
    DISPLAY = "formula_display"


def choose_mode(
    *, ctrl: bool = False, left_alt: bool = False, right_alt: bool = False, shift: bool = False
) -> RecognitionMode | None:
    """Select a mode from already-observed modifier states.

    Right Alt (including AltGr) and Ctrl+Left Alt are intentionally passed
    through as no selection. This function does not read operating-system input.
    """
    if right_alt or (ctrl and left_alt):
        return None
    if ctrl and not left_alt:
        return RecognitionMode.TEXT
    if left_alt:
        return RecognitionMode.DISPLAY if shift else RecognitionMode.INLINE
    return None
