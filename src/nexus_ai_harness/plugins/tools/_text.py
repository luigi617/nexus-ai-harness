from __future__ import annotations

import re
from collections.abc import Iterator
from typing import TextIO

LINE_BREAK = re.compile(r"\r\n|\r|\n")
"""The line boundaries every file tool counts by: ``\\n``, ``\\r\\n``, lone ``\\r``.

These match Python's universal newlines (how ``read_file`` iterates a file), and
unlike :meth:`str.splitlines` they ignore form feeds and other separators, so
``grep``, ``read_file``, and ``edit_file`` agree on every line number.
"""

_CHUNK = 65_536


def split_lines(text: str) -> list[str]:
    """Split ``text`` into lines at :data:`LINE_BREAK`, dropping the terminators.

    Like :meth:`str.splitlines`, a trailing line break does not start an extra
    empty line.

    Args:
        text: The text to split.

    Returns:
        The lines, without their terminators.
    """
    lines = LINE_BREAK.split(text)
    if lines and not lines[-1]:
        lines.pop()
    return lines


def count_breaks(text: str) -> int:
    """Return how many :data:`LINE_BREAK` terminators ``text`` contains."""
    return sum(1 for _ in LINE_BREAK.finditer(text))


def iter_lines(handle: TextIO, max_chars: int) -> Iterator[tuple[str, bool]]:
    """Yield each line of ``handle`` clipped to ``max_chars``, in bounded memory.

    Reads fixed-size chunks rather than whole lines, so a multi-gigabyte line
    (a minified bundle, a log without newlines) never loads into memory: only
    its first ``max_chars`` characters are kept.

    Args:
        handle: A text file opened with ``newline=""`` so terminators arrive
            untranslated.
        max_chars: The most characters of one line to keep.

    Yields:
        ``(line, clipped)``: the line without its terminator, and whether it was
        longer than ``max_chars`` and cut.
    """
    kept = ""
    clipped = False
    held = ""
    while True:
        data = handle.read(_CHUNK)
        buf = held + data
        held = ""
        pos = 0
        for match in LINE_BREAK.finditer(buf):
            if match.group() == "\r" and match.end() == len(buf) and data:
                break  # the "\n" of a "\r\n" may start the next chunk
            kept, clipped = _extend(kept, clipped, buf[pos : match.start()], max_chars)
            yield kept, clipped
            kept, clipped = "", False
            pos = match.end()
        rest = buf[pos:]
        if not data:
            if rest or kept or clipped:
                yield _extend(kept, clipped, rest, max_chars)
            return
        if rest.endswith("\r"):
            held, rest = "\r", rest[:-1]
        kept, clipped = _extend(kept, clipped, rest, max_chars)


def _extend(kept: str, clipped: bool, piece: str, limit: int) -> tuple[str, bool]:
    """Append ``piece`` to a partial line, stopping at ``limit`` characters."""
    if clipped or not piece:
        return kept, clipped
    room = limit - len(kept)
    if len(piece) > room:
        return kept + piece[:room], True
    return kept + piece, False
