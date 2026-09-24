from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# Sentinel model value: use the in-memory ScriptedModel and inline responses.
SCRIPTED = "scripted"

# Default live model when a case omits ``model``: fast, cheap Bedrock Haiku 4.5.
DEFAULT_MODEL = "bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0"


@dataclass
class Expect:
    """The checks run against one finished case.

    Every field defaults to "don't care"; set one to enable its check, and all
    active checks must pass. Behavioral checks are deterministic even against a
    live model; text checks anchor on facts to survive paraphrasing and match
    the whole transcript (all turns) for multi-turn cases.

    Attributes:
        no_error: The run must complete without raising.
        stop_reason: Exact match on how the loop ended, e.g. ``"completed"``.
        stop_reason_contains: Substring match on stop_reason, e.g. ``"guard:"``.
        tool_calls: Each ``{"name", "arguments_contains"?}`` must match a call.
        tool_calls_ordered: Require ``tool_calls`` as an in-order subsequence.
        tool_names: Exact multiset of tool names called (``[]`` means no tools).
        tool_denied: Each ``{"name"}`` must have been blocked by permissions.
        max_cost_usd: Upper bound on total USD cost.
        max_iterations: Upper bound on model calls.
        output_contains: Every substring must appear in the transcript.
        output_contains_any: At least one substring must appear.
        output_not_contains: None of these may appear.
        output_regex: The transcript must match this pattern.
    """

    no_error: bool = True
    stop_reason: str | None = None
    stop_reason_contains: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_ordered: bool = False
    tool_names: list[str] | None = None
    tool_denied: list[dict[str, Any]] = field(default_factory=list)
    max_cost_usd: float | None = None
    max_iterations: int | None = None
    output_contains: list[str] = field(default_factory=list)
    output_contains_any: list[str] = field(default_factory=list)
    output_not_contains: list[str] = field(default_factory=list)
    output_regex: str | None = None


@dataclass
class Spec:
    """One JSON-defined end-to-end case.

    The harness composition lives in Python (see ``harnesses.py``); this spec
    only carries the data: which builder to use, which model, the conversation,
    and what to assert.

    Attributes:
        name: Unique case id; shown as the pytest id.
        input: The opening user message.
        description: Human-readable note.
        tags: Free-form labels.
        harness: Name of a builder registered via ``@e2e_harness``.
        model: ``"scripted"`` or a live ``"provider:model-id"``; defaults to
            Bedrock Haiku 4.5.
        scripted_responses: Scripted mode only; one entry per model turn, each
            ``{"text"?, "tool_calls"?: [{"name", "arguments", "id"?}], "cost"?}``.
        user_turns: Follow-up user messages for multi-turn cases (one session).
        env: Environment variables set for the run, then restored.
        expect: The assertions to run.
        timeout_s: Per-case timeout in seconds.
        skip: Skip this case.
        source: Path the spec was loaded from; set by ``load_spec``, not JSON.
    """

    name: str
    input: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    harness: str = "default"
    model: str = DEFAULT_MODEL
    scripted_responses: list[dict[str, Any]] = field(default_factory=list)
    user_turns: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    expect: Expect = field(default_factory=Expect)
    timeout_s: float = 120.0
    skip: bool = False
    source: Path | None = None

    @property
    def is_live(self) -> bool:
        """Whether this case calls a real provider (vs the scripted model)."""
        return self.model != SCRIPTED


def _reject_unknown(data: dict[str, Any], cls: type, where: str) -> None:
    """Raise if ``data`` has keys that aren't fields of ``cls`` (except source)."""
    known = {f.name for f in fields(cls)} - {"source"}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"{where}: unknown field(s) {sorted(unknown)}; allowed: {sorted(known)}"
        )


def load_spec(path: str | Path) -> Spec:
    """Parse one JSON file into a :class:`Spec`, failing loudly on typos."""
    path = Path(path)
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")

    expect_data = data.pop("expect", {})
    if not isinstance(expect_data, dict):
        raise ValueError(f"{path}: 'expect' must be an object")
    _reject_unknown(expect_data, Expect, f"{path} expect")
    _reject_unknown(data, Spec, str(path))

    for required in ("name", "input"):
        if required not in data:
            raise ValueError(f"{path}: missing required field {required!r}")

    return Spec(**data, expect=Expect(**expect_data), source=path)


def discover_specs(root: str | Path) -> list[Spec]:
    """Load every ``*.json`` under ``root`` (recursively), sorted by path."""
    root = Path(root)
    return [load_spec(p) for p in sorted(root.rglob("*.json"))]
