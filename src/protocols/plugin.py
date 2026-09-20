from __future__ import annotations

from abc import ABC
from typing import ClassVar


class Plugin(ABC):  # noqa: B024
    """Common abstract base of every plugin."""

    requires: ClassVar[tuple[type[Plugin], ...]] = ()
