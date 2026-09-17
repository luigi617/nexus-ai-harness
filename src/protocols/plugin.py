from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable


@runtime_checkable
class Plugin(Protocol):
    """Common base of every plugin protocol: each declares a ``kind``. Used to
    bound the type-keyed registry lookups (``get``/``all``) so they resolve a
    plugin by its protocol type and return that exact type.

    A plugin may *optionally* declare a ``requires`` class var — a tuple of the
    protocol types it depends on — so the harness can validate a composition
    before it runs (see ``NexusAIHarness.validate``). Each entry is a protocol
    type such as ``Model`` or ``ContextManager`` and is satisfied when some
    registered plugin has the matching ``kind``. It is read defensively (a
    plugin that omits it is treated as having no dependencies), so it is left
    off this protocol's required members to keep ``kind`` the only structural
    obligation of a plugin.
    """

    kind: ClassVar[str]
