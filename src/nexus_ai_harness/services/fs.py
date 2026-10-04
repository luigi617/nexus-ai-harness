from __future__ import annotations

import os
import time

_REPLACE_ATTEMPTS = 8
_REPLACE_DELAY = 0.02


def replace_retrying(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    """Atomically replace ``dst`` with ``src``, retrying past Windows sharing locks.

    POSIX lets a rename proceed even while another handle has ``dst`` open;
    Windows can transiently refuse with a :class:`PermissionError` instead. A
    short retry clears that without weakening the atomicity guarantee, since
    each attempt is still a single ``os.replace`` call.

    Args:
        src: The file to move into place.
        dst: The path it replaces.

    Raises:
        PermissionError: If every attempt is refused.
    """
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_DELAY)
