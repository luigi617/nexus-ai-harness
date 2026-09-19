from __future__ import annotations

from dataclasses import dataclass

from core.phase import Phase
from protocols.interceptor import Interceptor
from protocols.plugin import Plugin


@dataclass(frozen=True)
class InterceptorBinding:
    """An interceptor plus the target type and phase it fires around.

    Attributes:
        target: The plugin type whose invocation the interceptor wraps; an
            invoked plugin matches when it is an instance of ``target``.
        phase: ``Phase.BEFORE`` to run ahead of the invocation, ``Phase.AFTER``
            after it.
        interceptor: The interceptor to run.
    """

    target: type[Plugin]
    phase: Phase
    interceptor: Interceptor
