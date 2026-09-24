from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from benchmarks.core.benchmark import Benchmark, Episode
from benchmarks.core.registry import register
from benchmarks.core.task import Score, Task
from core.events import Event, ResponseReceived
from harness import NexusAIHarness
from harness.result import RunResult
from harness.session import Session
from plugins.loops import ChatLoop
from protocols.context import Context
from protocols.hook import Hook
from protocols.model import Model
from protocols.tool import Tool

DEFAULT_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple")
# Category -> upstream file stem. ``simple`` maps to the Python split; the Java
# and JavaScript splits are out of scope for this AST checker.
_CATEGORY_FILE = {
    "simple": "simple_python",
    "multiple": "multiple",
    "parallel": "parallel",
    "parallel_multiple": "parallel_multiple",
    "irrelevance": "irrelevance",  # no ground truth: passing means calling nothing
}
_VERSION = os.getenv("BFCL_VERSION", "BFCL_v4")
_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "bfcl"
_GORILLA = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main"
# Upstream has moved the data dir across versions; try known layouts in order.
_RAW_BASES = tuple(
    base
    for base in (
        os.getenv("BFCL_RAW_BASE"),
        f"{_GORILLA}/berkeley-function-call-leaderboard/bfcl_eval/data",
        f"{_GORILLA}/berkeley-function-call-leaderboard/data",
    )
    if base
)
# BFCL uses dotted function names (e.g. ``math.factorial``), but every model
# backend (Bedrock/OpenAI/Anthropic) requires tool names to match this pattern,
# so sanitize before registering and match the sanitized form when grading.
_ILLEGAL_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_name(name: str) -> str:
    return _ILLEGAL_NAME_CHARS.sub("_", name)


# BFCL JSON-schema type spellings -> JSON Schema the model backends accept.
_TYPE_MAP = {
    "dict": "object",
    "float": "number",
    "tuple": "array",
    "integer": "integer",
}


@dataclass
class CapturedCalls:
    """Tool calls the model emitted this run, stashed for the grader."""

    calls: list[dict] = field(default_factory=list)


class _CallCaptor(Hook):
    """Record every emitted tool call (``ChatLoop`` drops them off the message)."""

    def on(self, event: Event, ctx: Context) -> None:
        if isinstance(event, ResponseReceived):
            ctx.state(CapturedCalls).calls.extend(event.response.tool_calls)


class _FunctionSpecTool(Tool):
    """A no-op tool exposing a BFCL function schema so the model can 'call' it."""

    # Declared abstract on Tool; set per-instance below.
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    parameters: ClassVar[dict] = {}

    def __init__(self, schema: dict) -> None:
        self.name = _sanitize_name(schema["name"])
        self.description = schema.get("description", "")
        self.parameters = _normalize_schema(schema.get("parameters", {}))

    def run(self, arguments: dict, ctx: Context) -> str:
        return ""  # never executed: ChatLoop grades the emitted call, not its result


@register
class BFCL(Benchmark):
    name = "bfcl"
    description = "Berkeley Function-Calling Leaderboard (single-turn AST categories)"

    def __init__(self, categories: tuple[str, ...] = DEFAULT_CATEGORIES) -> None:
        self.categories = categories

    def load_tasks(self, *, limit: int | None = None) -> list[Task]:
        tasks: list[Task] = []
        n_cats = len(self.categories)
        for idx, category in enumerate(self.categories):
            # Distribute ``limit`` across categories, handing the remainder to
            # the earliest categories, so the total fills to exactly ``limit``
            # (data permitting) instead of floor-dividing and under-filling.
            per_category = (
                None
                if limit is None
                else limit // n_cats + (1 if idx < limit % n_cats else 0)
            )
            stem = _CATEGORY_FILE.get(category, category)
            irrelevance = category == "irrelevance"
            questions = _load_jsonl(f"{_VERSION}_{stem}.json")
            # Irrelevance has no ground truth — passing means emitting no call.
            answers = (
                {}
                if irrelevance
                else {
                    a["id"]: a["ground_truth"]
                    for a in _load_jsonl(f"possible_answer/{_VERSION}_{stem}.json")
                }
            )
            for row in questions[:per_category]:
                tasks.append(
                    Task(
                        task_id=row["id"],
                        prompt=_render_prompt(row["question"]),
                        expected=answers.get(row["id"], []),
                        metadata={
                            "category": category,
                            "functions": row.get("function", []),
                            "irrelevance": category == "irrelevance",
                        },
                    )
                )
        return tasks[:limit] if limit is not None else tasks

    def build_harness(self, task: Task, model: Model, session: Session) -> Episode:
        harness = NexusAIHarness().use(ChatLoop()).use(model).use(_CallCaptor())
        for schema in task.metadata.get("functions", []):
            harness.use(_FunctionSpecTool(schema))
        return Episode(harness=harness, initial_input=task.prompt)  # single-turn

    def grade(self, task: Task, result: RunResult) -> Score:
        emitted = result.session.state(CapturedCalls).calls
        if task.metadata.get("irrelevance"):
            passed = len(emitted) == 0
            return Score(passed=passed, detail={"emitted": len(emitted)})
        passed = _ast_match(task.expected, emitted)
        return Score(
            passed=passed,
            detail={
                "expected": task.expected,
                "emitted": [
                    {"name": c.get("name"), "arguments": c.get("arguments", {})}
                    for c in emitted
                ],
            },
        )


# --- AST matching ---------------------------------------------------------


def _ast_match(ground_truth: list[dict], emitted: list[dict]) -> bool:
    """Every gold call must be matched exactly once by an emitted call."""
    if len(emitted) != len(ground_truth):
        return False
    remaining = list(emitted)
    for gold in ground_truth:  # each gold is {func_name: {param: [accepted, ...]}}
        ((func_name, params),) = gold.items()
        # Emitted names are sanitized (dots -> _) to satisfy backend constraints;
        # sanitize the gold name too so the comparison stays symmetric.
        gold_name = _sanitize_name(func_name)
        match = next(
            (
                c
                for c in remaining
                if c.get("name") == gold_name
                and _params_match(params, c.get("arguments", {}))
            ),
            None,
        )
        if match is None:
            return False
        remaining.remove(match)
    return True


def _params_match(gold_params: dict, args: dict) -> bool:
    """Each gold param must have an accepted value (or be legitimately absent)."""
    for key, accepted in gold_params.items():
        accepted_list = accepted if isinstance(accepted, list) else [accepted]
        if key not in args:
            if "" in accepted_list or None in accepted_list:
                continue  # BFCL marks optional/defaulted params with ""/None
            return False
        if not any(_value_eq(args[key], ok) for ok in accepted_list):
            return False
    # Reject calls that invent params BFCL didn't sanction for this function.
    return not set(args) - set(gold_params)


def _value_eq(produced: Any, accepted: Any) -> bool:
    # Nested object: BFCL encodes a dict-valued argument as {key: [accepted,...]},
    # so grade it recursively rather than comparing the dict as an opaque value.
    if isinstance(accepted, dict):
        return isinstance(produced, dict) and _params_match(accepted, produced)
    # bool is an int subclass in Python (True == 1, False == 0). Require both
    # sides to be bool before treating them as equal, so a boolean argument is
    # never accepted where a numeric gold value was expected (and vice versa).
    if isinstance(produced, bool) or isinstance(accepted, bool):
        return (
            isinstance(produced, bool)
            and isinstance(accepted, bool)
            and produced == accepted
        )
    if produced == accepted:
        return True
    # Number tolerance: 3 == 3.0, "3" == 3.
    try:
        if float(produced) == float(accepted):
            return True
    except (TypeError, ValueError):
        pass
    return str(produced).strip() == str(accepted).strip()


# --- data loading ---------------------------------------------------------


def _normalize_schema(params: dict) -> dict:
    """Rewrite BFCL type spellings into JSON Schema the backends accept.

    Recurses through nested object ``properties`` and array ``items`` so that a
    deeply-nested ``float``/``dict``/``tuple`` (e.g. an object property that is
    itself an object) is mapped too — leaving one unmapped keeps the schema
    from validating as JSON Schema draft 2020-12.
    """
    if not params:
        return {"type": "object", "properties": {}}
    out = dict(params)
    if "type" in out:
        out["type"] = _TYPE_MAP.get(out["type"], out["type"])
    if isinstance(out.get("properties"), dict):
        out["properties"] = {
            name: _normalize_schema(spec) for name, spec in out["properties"].items()
        }
    if isinstance(out.get("items"), dict):
        out["items"] = _normalize_schema(out["items"])
    return out


def _render_prompt(question: list) -> str:
    """Flatten BFCL's single-turn question into one user string (system folded in)."""
    turn = question[0] if question else []
    system = "\n".join(m["content"] for m in turn if m.get("role") == "system")
    user = "\n".join(m["content"] for m in turn if m.get("role") == "user")
    return f"{system}\n\n{user}".strip() if system else user


def _load_jsonl(relpath: str) -> list[dict]:
    """Load a BFCL JSONL file, fetching + caching from upstream on first use."""
    local_dir = os.getenv("BFCL_DATA_DIR")
    if local_dir:  # offline: read from a gorilla checkout
        return _read_jsonl(Path(local_dir) / relpath)

    cached = _CACHE_DIR / relpath
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(_fetch(relpath))
    return _read_jsonl(cached)


def _fetch(relpath: str) -> bytes:
    errors = []
    for base in _RAW_BASES:
        url = f"{base}/{relpath}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except Exception as exc:  # try the next known layout
            errors.append(f"{url}: {exc}")
    raise RuntimeError(
        "could not fetch BFCL data (set BFCL_DATA_DIR to a local gorilla "
        "checkout, or BFCL_RAW_BASE):\n  " + "\n  ".join(errors)
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
