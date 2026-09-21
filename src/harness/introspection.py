from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from harness.registry import PluginStatus, Registry
from protocols.approver import Approver
from protocols.context_manager import ContextManager
from protocols.guard import Guard
from protocols.hook import Hook
from protocols.interceptor import Interceptor
from protocols.loop import Loop
from protocols.memory import MemoryStore
from protocols.model import Model
from protocols.permission import Permission
from protocols.plugin import Plugin
from protocols.router import Router
from protocols.spawner import Spawner
from protocols.tool import Tool


class Selection(Enum):
    """How the harness resolves the providers of a capability."""

    SINGLE = auto()
    ALL = auto()


@dataclass(frozen=True)
class _CapabilitySpec:
    protocol: type[Plugin]
    label: str
    selection: Selection


_CAPABILITIES: tuple[_CapabilitySpec, ...] = (
    _CapabilitySpec(Loop, "Loop", Selection.SINGLE),
    _CapabilitySpec(Model, "Model", Selection.SINGLE),
    _CapabilitySpec(Router, "Router", Selection.SINGLE),
    _CapabilitySpec(ContextManager, "ContextManagers", Selection.ALL),
    _CapabilitySpec(MemoryStore, "Memory", Selection.SINGLE),
    _CapabilitySpec(Tool, "Tools", Selection.ALL),
    _CapabilitySpec(Spawner, "Spawner", Selection.SINGLE),
    _CapabilitySpec(Approver, "Approver", Selection.SINGLE),
    _CapabilitySpec(Permission, "Permissions", Selection.ALL),
    _CapabilitySpec(Guard, "Guards", Selection.ALL),
    _CapabilitySpec(Hook, "Hooks", Selection.ALL),
    _CapabilitySpec(Interceptor, "Interceptors", Selection.ALL),
)


@dataclass(frozen=True)
class PluginInfo:
    """Introspected facts about one registered plugin."""

    name: str
    qualified_name: str
    status: PluginStatus
    provides: tuple[str, ...]
    requires: tuple[str, ...]
    target: str | None
    instance: Plugin


@dataclass(frozen=True)
class Capability:
    """A capability protocol together with the plugins that provide it."""

    protocol: str
    label: str
    providers: tuple[PluginInfo, ...]
    selected: PluginInfo | None
    selects_one: bool


@dataclass(frozen=True)
class HarnessInspection:
    """A structured snapshot of a harness's plugin composition."""

    name: str
    plugins: tuple[PluginInfo, ...]
    capabilities: tuple[Capability, ...]

    def plugin(self, name: str) -> PluginInfo | None:
        """Return the first plugin whose class name is ``name``, or ``None``."""
        return next((p for p in self.plugins if p.name == name), None)

    def capability(self, protocol: type[Plugin] | str) -> Capability | None:
        """Return the capability for ``protocol`` (a protocol type or its name)."""
        key = protocol if isinstance(protocol, str) else protocol.__name__
        return next((c for c in self.capabilities if c.protocol == key), None)

    def provider_of(self, protocol: type[Plugin] | str) -> PluginInfo | None:
        """Return the provider the harness resolves for a single-select capability."""
        cap = self.capability(protocol)
        return cap.selected if cap is not None else None

    def providers_of(self, protocol: type[Plugin] | str) -> tuple[PluginInfo, ...]:
        """Return every plugin providing ``protocol``, in registration order."""
        cap = self.capability(protocol)
        return cap.providers if cap is not None else ()

    def dependents_of(self, protocol: type[Plugin] | str) -> tuple[PluginInfo, ...]:
        """Return the plugins that declare ``protocol`` among their ``requires``."""
        key = protocol if isinstance(protocol, str) else protocol.__name__
        return tuple(p for p in self.plugins if key in p.requires)

    def render(self) -> str:
        """Render the composition as an indented tree rooted at the harness name."""
        return _render_tree(self)

    def __str__(self) -> str:
        return self.render()


def _plugin_info(plugin: Plugin, registry: Registry) -> PluginInfo:
    cls = type(plugin)
    provides = tuple(
        spec.protocol.__name__
        for spec in _CAPABILITIES
        if isinstance(plugin, spec.protocol)
    )
    requires = tuple(dep.__name__ for dep in plugin.requires)
    target = plugin.target.__name__ if isinstance(plugin, Interceptor) else None
    return PluginInfo(
        name=cls.__name__,
        qualified_name=f"{cls.__module__}.{cls.__qualname__}",
        status=registry.status_of(plugin) or PluginStatus.REGISTERED,
        provides=provides,
        requires=requires,
        target=target,
        instance=plugin,
    )


def inspect_registry(
    registry: Registry, *, name: str = "NexusAIHarness"
) -> HarnessInspection:
    """Build a :class:`HarnessInspection` describing ``registry``'s composition.

    Args:
        registry: The registry whose plugins and capabilities to inspect.
        name: The name to show as the root of the rendered tree.

    Returns:
        A structured snapshot with one entry per registered plugin and one
        capability per provided protocol.
    """
    plugins = registry.plugins()
    infos = {id(p): _plugin_info(p, registry) for p in plugins}
    # A Router picks the model per turn from ctx.all(Model), so no single Model
    # provider can be named statically.
    router_present = any(isinstance(p, Router) for p in plugins)
    capabilities: list[Capability] = []
    for spec in _CAPABILITIES:
        providers = tuple(infos[id(p)] for p in plugins if isinstance(p, spec.protocol))
        if not providers:
            continue
        single = spec.selection is Selection.SINGLE
        dynamic = spec.protocol is Model and router_present
        capabilities.append(
            Capability(
                protocol=spec.protocol.__name__,
                label=spec.label,
                providers=providers,
                # ctx.get resolves the last registration; None when router-driven.
                selected=providers[-1] if single and not dynamic else None,
                selects_one=single,
            )
        )
    return HarnessInspection(
        name=name,
        plugins=tuple(infos[id(p)] for p in plugins),
        capabilities=tuple(capabilities),
    )


def _render_tree(inspection: HarnessInspection) -> str:
    lines = [inspection.name]
    caps = inspection.capabilities
    for i, cap in enumerate(caps):
        last_cap = i == len(caps) - 1
        lines.append(f"{'└──' if last_cap else '├──'} {cap.label}")
        indent = "    " if last_cap else "│   "
        shadowed = cap.selects_one and len(cap.providers) > 1
        for j, provider in enumerate(cap.providers):
            leaf = "└──" if j == len(cap.providers) - 1 else "├──"
            # Name the winner only when providers compete for a single slot.
            mark = "  (selected)" if shadowed and provider is cap.selected else ""
            lines.append(f"{indent}{leaf} {provider.name}{mark}")
    return "\n".join(lines)
