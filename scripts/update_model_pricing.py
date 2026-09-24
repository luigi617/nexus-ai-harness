#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO / "src" / "plugins" / "models"
SOURCE_URL = "https://models.dev/api.json"
LINE_LIMIT = 88  # ruff line-length; rewritten dict lines must fit


@dataclass
class ProviderSpec:
    """Maps one of this repo's model backends to a models.dev provider feed.

    Attributes:
        key: Our provider name, also the backend module stem.
        class_name: The backend class to rewrite in that module.
        source_key: The provider's key in the models.dev feed.
        id_prefix: If set, keep only model ids starting with this prefix.
    """

    key: str
    class_name: str
    source_key: str
    id_prefix: str = ""


SPECS = [
    ProviderSpec("openai", "OpenAIModel", "openai"),
    ProviderSpec("anthropic", "AnthropicModel", "anthropic"),
    ProviderSpec("gemini", "GeminiModel", "google", id_prefix="gemini"),
    ProviderSpec("groq", "GroqModel", "groq"),
    ProviderSpec("deepseek", "DeepSeekModel", "deepseek"),
    ProviderSpec("xai", "XAIModel", "xai"),
    ProviderSpec("minimax", "MiniMaxModel", "minimax"),
    ProviderSpec("qwen", "QwenModel", "alibaba", id_prefix="qwen"),
    ProviderSpec("glm", "GLMModel", "zhipuai", id_prefix="glm"),
    # Bedrock lists many families; scope to the region-profiled Anthropic ids.
    ProviderSpec(
        "bedrock", "BedrockModel", "amazon-bedrock", id_prefix="us.anthropic."
    ),
]


@dataclass
class ProviderChanges:
    """What changed for one provider, used to build the PR summary."""

    provider: str
    added: list[str] = field(default_factory=list)
    updated: list[tuple[str, tuple, tuple]] = field(default_factory=list)
    kept_unlisted: list[str] = field(default_factory=list)
    skipped_reason: str = ""

    @property
    def touched(self) -> bool:
        """Whether the provider gained or repriced any models."""
        return bool(self.added or self.updated)


# --- source parsing -------------------------------------------------------


def load_source(url: str, file: str | None) -> dict:
    """Load the models.dev feed from ``file`` when given, else fetch ``url``."""
    if file:
        return json.loads(Path(file).read_text())
    req = urllib.request.Request(url, headers={"User-Agent": "nexus-price-bot"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _is_chat(model: dict) -> bool:
    """Whether a models.dev entry is a text-generating chat model.

    Requires text-only output (excludes image/audio/video and realtime) and a
    generative capability flag (excludes embeddings, which are text-out only).
    """
    if (model.get("modalities", {}) or {}).get("output") != ["text"]:
        return False
    return bool(
        model.get("tool_call")
        or model.get("structured_output")
        or model.get("reasoning")
    )


def collect(spec: ProviderSpec, source: dict) -> tuple[dict, dict]:
    """Return ``{model_id: (in_per_m, out_per_m)}`` and ``{model_id: blurb}``."""
    prices: dict[str, tuple[float, float]] = {}
    blurbs: dict[str, str] = {}
    models = (source.get(spec.source_key) or {}).get("models", {})
    for mid, model in models.items():
        if spec.id_prefix and not mid.startswith(spec.id_prefix):
            continue
        if not _is_chat(model):
            continue
        cost = model.get("cost") or {}
        cin, cout = cost.get("input"), cost.get("output")
        if cin is None or cout is None:
            continue
        prices[mid] = (float(cin), float(cout))
        desc = (model.get("description") or "").strip()
        if desc:
            blurbs[mid] = desc
    return prices, blurbs


# --- reading current dicts (via AST, no imports / side effects) -----------


def _current_dicts(path: Path, class_name: str) -> tuple[dict, dict]:
    """Read the existing ``pricing`` and ``descriptions`` dicts from a backend."""
    tree = ast.parse(path.read_text())
    cls = _class_node(tree, class_name)
    if cls is None:
        raise ValueError(f"class {class_name} not found in {path}")
    pricing = _literal_of(cls, "pricing") or {}
    descriptions = _literal_of(cls, "descriptions") or {}
    return dict(pricing), dict(descriptions)


def _literal_of(cls: ast.ClassDef, name: str):
    node = _target_assign(cls, name)
    if node is None:
        return None
    return ast.literal_eval(node.value)


# --- AST rewrite ----------------------------------------------------------


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef | None:
    return next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name), None
    )


def _target_assign(cls: ast.ClassDef, name: str):
    for node in cls.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return node
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node
    return None


def _insert_after_line(cls: ast.ClassDef) -> int:
    """1-based source line to insert a new class var after."""
    last_var = None
    for node in cls.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            last_var = node
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            break
    anchor = last_var or cls.body[0]  # fall back to the docstring / first stmt
    end = anchor.end_lineno
    assert end is not None  # nodes from ast.parse always carry positions
    return end


def _render(name: str, annotation: str, items: list[str]) -> list[str]:
    if not items:
        return [f"    {name}: {annotation} = {{}}"]
    return [f"    {name}: {annotation} = {{", *items, "    }"]


def _render_pricing(d: dict) -> list[str]:
    items = [
        f"        {json.dumps(k)}: ({float(i)}, {float(o)}),"
        for k, (i, o) in sorted(d.items())
    ]
    return _render("pricing", "ClassVar[dict[str, tuple[float, float]]]", items)


def _fit_desc(model_id: str, desc: str) -> str:
    """Trim a description so its rendered line stays within the line limit."""
    budget = LINE_LIMIT - len(f"        {json.dumps(model_id)}: ") - len(",")
    trimmed = desc
    while trimmed and len(json.dumps(trimmed)) > budget:
        trimmed = trimmed[:-1].rstrip()
    return trimmed


def _render_descriptions(d: dict) -> list[str]:
    items = [
        f"        {json.dumps(k)}: {json.dumps(_fit_desc(k, v))},"
        for k, v in sorted(d.items())
    ]
    return _render("descriptions", "ClassVar[dict[str, str]]", items)


def _ensure_classvar_import(lines: list[str]) -> list[str]:
    # Match import statements only: the inserted dict literal also says "ClassVar".
    if any(
        ln.startswith(("from typing import", "import typing")) and "ClassVar" in ln
        for ln in lines
    ):
        return lines
    for i, ln in enumerate(lines):
        if ln.startswith("from typing import "):
            lines[i] = ln.rstrip() + ", ClassVar"
            return lines
    # No typing import: add one as its own stdlib block after ``__future__``.
    fut = next(
        (i for i, ln in enumerate(lines) if ln.startswith("from __future__")), -1
    )
    at = fut + 1
    if at < len(lines) and lines[at].strip() == "":
        at += 1  # place after the blank line that follows __future__
    block = ["from typing import ClassVar"]
    if at < len(lines) and lines[at].strip() != "":
        block.append("")
    lines[at:at] = block
    return lines


def _rewrite_dict(path: Path, class_name: str, name: str, rendered: list[str]) -> None:
    """Replace ``name``'s dict literal in the class, or insert it if absent."""
    lines = path.read_text().split("\n")
    tree = ast.parse("\n".join(lines))
    cls = _class_node(tree, class_name)
    if cls is None:
        raise ValueError(f"class {class_name} not found in {path}")
    target = _target_assign(cls, name)
    if target is not None:
        lo, hi = target.lineno - 1, target.end_lineno
        lines = lines[:lo] + rendered + lines[hi:]
    else:
        at = _insert_after_line(cls)
        lines = lines[:at] + rendered + lines[at:]
    lines = _ensure_classvar_import(lines)
    path.write_text("\n".join(lines))


# --- orchestration --------------------------------------------------------


def process(spec: ProviderSpec, source: dict, *, apply: bool) -> ProviderChanges:
    """Compute updates for a provider; write them to the backend when ``apply``."""
    changes = ProviderChanges(provider=spec.key)
    path = MODELS_DIR / f"{spec.key}.py"
    if not path.exists():
        changes.skipped_reason = "no backend file"
        return changes

    prices, blurbs = collect(spec, source)
    if not prices:
        changes.skipped_reason = f"no chat models under models.dev '{spec.source_key}'"
        return changes

    cur_pricing, cur_desc = _current_dicts(path, spec.class_name)

    new_pricing = dict(cur_pricing)
    for mid, price in prices.items():
        if mid not in cur_pricing:
            changes.added.append(mid)
        elif cur_pricing[mid] != price:
            changes.updated.append((mid, tuple(cur_pricing[mid]), price))
        new_pricing[mid] = price
    changes.kept_unlisted = [m for m in cur_pricing if m not in prices]

    new_desc = dict(cur_desc)
    for mid in prices:  # preserve hand-written descriptions; only fill gaps
        if mid not in new_desc and mid in blurbs:
            new_desc[mid] = blurbs[mid]

    if apply and changes.touched:
        _rewrite_dict(path, spec.class_name, "pricing", _render_pricing(new_pricing))
        _rewrite_dict(
            path, spec.class_name, "descriptions", _render_descriptions(new_desc)
        )
    return changes


def render_report(results: list[ProviderChanges]) -> str:
    """Render a Markdown summary of the changes for the PR body."""
    lines = ["## Model pricing update (source: models.dev)", ""]
    any_change = False
    for r in results:
        if r.skipped_reason:
            lines.append(f"- **{r.provider}**: skipped — {r.skipped_reason}")
            continue
        if not r.touched:
            lines.append(f"- **{r.provider}**: up to date")
            continue
        any_change = True
        lines.append(
            f"- **{r.provider}**: +{len(r.added)} added, {len(r.updated)} repriced"
        )
        for mid in sorted(r.added):
            lines.append(f"    - add `{mid}`")
        for mid, old, new in sorted(r.updated):
            lines.append(f"    - `{mid}` {old} → {new}")
        if r.kept_unlisted:
            kept = ", ".join(f"`{m}`" for m in sorted(r.kept_unlisted))
            lines.append(f"    - kept (not in source): {kept}")
    if not any_change:
        lines.append("")
        lines.append("_No pricing changes._")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Parse args, refresh pricing, print the report, and return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report only, no writes")
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if updates are needed (no writes)"
    )
    parser.add_argument("--only", nargs="*", help="limit to these providers")
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--source-file", default=None, help="read the feed from a file")
    parser.add_argument("--summary", default=None, help="write the report to this path")
    args = parser.parse_args(argv)

    source = load_source(args.source_url, args.source_file)
    specs = [s for s in SPECS if not args.only or s.key in args.only]
    apply = not (args.dry_run or args.check)

    results = [process(s, source, apply=apply) for s in specs]
    report = render_report(results)
    print(report)
    if args.summary:
        Path(args.summary).write_text(report + "\n")

    if args.check and any(r.touched for r in results):
        print("\nchanges needed (run without --check to apply)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
