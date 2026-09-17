from __future__ import annotations

from collections.abc import Sequence

from harness.registry import Registry
from protocols.plugin import Plugin

_CHECK = "✓"
_CROSS = "✗"


class MissingDependencyError(Exception):
    """Raised by :meth:`NexusAIHarness.validate` when a registered plugin
    declares a dependency (via ``requires``) that no other registered plugin
    satisfies.

    ``missing`` is the list of ``(plugin_name, protocol_name)`` pairs that were
    unsatisfied; ``report`` is the rendered dependency tree shown in the message.
    """

    def __init__(self, missing: list[tuple[str, str]], report: str) -> None:
        self.missing = missing
        self.report = report
        super().__init__(report)


def _requires(plugin: object) -> Sequence[type[Plugin]]:
    return getattr(plugin, "requires", ()) or ()


def _render_tree(plugin_name: str, rows: list[tuple[str, bool]]) -> str:
    lines = [plugin_name]
    for i, (protocol_name, satisfied) in enumerate(rows):
        branch = "└──" if i == len(rows) - 1 else "├──"
        mark = _CHECK if satisfied else _CROSS
        lines.append(f"{branch} {protocol_name} {mark}")
    return "\n".join(lines)


def _analyze(
    registry: Registry,
) -> tuple[list[tuple[str, list[tuple[str, bool]]]], list[tuple[str, str]]]:
    """Return ``(trees, missing)`` where ``trees`` is one
    ``(plugin_name, rows)`` entry per plugin that declares dependencies and
    ``missing`` lists every unsatisfied ``(plugin_name, protocol_name)``."""
    plugins = registry.plugins()
    trees: list[tuple[str, list[tuple[str, bool]]]] = []
    missing: list[tuple[str, str]] = []
    for plugin in plugins:
        reqs = _requires(plugin)
        if not reqs:
            continue
        plugin_name = type(plugin).__name__
        # A dependency must be provided by *another* plugin.
        others = [p for p in plugins if p is not plugin]
        rows: list[tuple[str, bool]] = []
        for dep in reqs:
            satisfied = _is_satisfied(dep, others)
            rows.append((dep.__name__, satisfied))
            if not satisfied:
                missing.append((plugin_name, dep.__name__))
        trees.append((plugin_name, rows))
    return trees, missing


def _is_satisfied(dep: type, others: list) -> bool:
    """Is dependency ``dep`` provided by one of the ``others`` plugins?"""
    if getattr(dep, "_is_protocol", False):
        dep_kind = getattr(dep, "kind", None)
        return any(getattr(p, "kind", None) == dep_kind for p in others)
    return any(isinstance(p, dep) for p in others)


def describe_registry(registry: Registry) -> str:
    """Render the dependency tree for every plugin that declares ``requires``.

    Purely descriptive — never raises. Returns an empty string when no plugin
    declares any dependencies.
    """
    trees, _ = _analyze(registry)
    return "\n\n".join(_render_tree(name, rows) for name, rows in trees)


def validate_registry(registry: Registry) -> None:
    """Validate that every declared dependency in ``registry`` is satisfied.

    Raises :class:`MissingDependencyError` with a rendered tree when any is
    missing; returns ``None`` otherwise.
    """
    trees, missing = _analyze(registry)
    if not missing:
        return
    unsatisfied_trees = [
        _render_tree(name, rows)
        for name, rows in trees
        if any(not satisfied for _, satisfied in rows)
    ]
    lines = ["\n\n".join(unsatisfied_trees), ""]
    lines += [
        f"Missing dependency: {plugin_name} requires {protocol_name}"
        for plugin_name, protocol_name in missing
    ]
    raise MissingDependencyError(missing, "\n".join(lines))
