"""Conservative, lossless LaTeX presentation formatting."""

from .recognition_mode import RecognitionMode


def _dollar_tokens(text: str) -> list[tuple[int, int]]:
    """Return active dollar runs, checking escape parity one character at a time."""
    tokens: list[tuple[int, int]] = []
    slash_run = 0
    for index, char in enumerate(text):
        if char == "\\":
            slash_run += 1
            continue
        if char == "$":
            if slash_run % 2 == 0:
                if tokens and tokens[-1][0] + tokens[-1][1] == index:
                    start, width = tokens[-1]
                    tokens[-1] = (start, width + 1)
                else:
                    tokens.append((index, 1))
        slash_run = 0
    return tokens


def _backslash_tokens(text: str) -> list[tuple[int, str]]:
    """Return active math delimiters using one linear scan over slash runs."""
    tokens: list[tuple[int, str]] = []
    index = 0
    while index < len(text):
        if text[index] != "\\":
            index += 1
            continue
        start = index
        while index < len(text) and text[index] == "\\":
            index += 1
        if index < len(text) and (index - start) % 2 and text[index] in "()[]":
            tokens.append((index - 1, "\\" + text[index]))
        index += 1
    return tokens


def _unwrap_once(body: str) -> str:
    dollars = _dollar_tokens(body)
    delimiters = _backslash_tokens(body)
    if dollars and delimiters:
        raise ValueError("mixed outer math delimiters are ambiguous")

    if dollars:
        if len(dollars) != 2:
            raise ValueError("unmatched or multiple dollar delimiters")
        (start, width), (end, end_width) = dollars
        if start != 0 or end + end_width != len(body) or width != end_width:
            raise ValueError("dollar delimiters do not form one outer block")
        if width not in (1, 2):
            raise ValueError("unsupported dollar delimiter width")
        unwrapped = body[width:-width]
        if _dollar_tokens(unwrapped) or _backslash_tokens(unwrapped):
            raise ValueError("nested or mixed math blocks are ambiguous")
        return unwrapped

    if delimiters:
        if len(delimiters) != 2:
            raise ValueError("unmatched or multiple backslash math delimiters")
        (start, opening), (end, closing) = delimiters
        pairs = {r"\(": r"\)", r"\[": r"\]"}
        if start != 0 or end + 2 != len(body) or pairs.get(opening) != closing:
            raise ValueError("backslash delimiters do not form one outer block")
        unwrapped = body[2:-2]
        if _dollar_tokens(unwrapped) or _backslash_tokens(unwrapped):
            raise ValueError("nested or mixed math blocks are ambiguous")
        return unwrapped

    return body


def format_latex(raw_latex: str, mode: RecognitionMode) -> str:
    """Wrap raw LaTeX for inline or display presentation without rewriting it."""
    if not isinstance(raw_latex, str):
        raise TypeError("raw_latex must be a string")
    if not isinstance(mode, RecognitionMode):
        raise TypeError("mode must be a RecognitionMode")
    if mode not in (RecognitionMode.INLINE, RecognitionMode.DISPLAY):
        raise ValueError("mode must be INLINE or DISPLAY")
    if "\x00" in raw_latex:
        raise ValueError("LaTeX source must not contain NUL")

    body = raw_latex.strip()
    if not body:
        raise ValueError("LaTeX source must not be blank")
    body = _unwrap_once(body)
    if not body.strip():
        raise ValueError("LaTeX body must not be blank")

    if mode is RecognitionMode.INLINE:
        return f"${body}$"
    return f"$$\n{body}\n$$"
