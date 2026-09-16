from __future__ import annotations

from collections.abc import Iterable

from core.permission import PermissionDecision
from protocols.context import Context
from protocols.permission import Permission


class AllowList(Permission):
    """Permit only tools whose name is on the list; deny the rest."""

    def __init__(self, allowed: Iterable[str]) -> None:
        self._allowed = set(allowed)

    def check(self, call: dict, ctx: Context) -> PermissionDecision:
        name = call.get("name", "")
        if name in self._allowed:
            return PermissionDecision.allow()
        return PermissionDecision.deny(f"tool {name!r} is not allowed")


class DenyList(Permission):
    """Deny tools whose name is on the list; allow the rest."""

    def __init__(self, denied: Iterable[str]) -> None:
        self._denied = set(denied)

    def check(self, call: dict, ctx: Context) -> PermissionDecision:
        name = call.get("name", "")
        if name in self._denied:
            return PermissionDecision.deny(f"tool {name!r} is blocked")
        return PermissionDecision.allow()


class AskUnless(Permission):
    """Allow trusted tools outright; ask the user about everything else."""

    def __init__(self, trusted: Iterable[str]) -> None:
        self._trusted = set(trusted)

    def check(self, call: dict, ctx: Context) -> PermissionDecision:
        name = call.get("name", "")
        if name in self._trusted:
            return PermissionDecision.allow()
        return PermissionDecision.ask(f"Allow tool {name!r}?")
