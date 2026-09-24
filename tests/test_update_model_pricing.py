from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "update_model_pricing", REPO / "scripts" / "update_model_pricing.py"
)
assert _spec is not None and _spec.loader is not None
ump = importlib.util.module_from_spec(_spec)
sys.modules["update_model_pricing"] = ump  # @dataclass needs the module resolvable
_spec.loader.exec_module(ump)


def _model(output=("text",), tool_call=True, cost=(1.0, 2.0), desc="d", **extra):
    m = {"modalities": {"input": ["text"], "output": list(output)}, "description": desc}
    if tool_call is not None:
        m["tool_call"] = tool_call
    if cost is not None:
        m["cost"] = {"input": cost[0], "output": cost[1]}
    m.update(extra)
    return m


# --- chat filtering -------------------------------------------------------


def test_is_chat_accepts_generative_text_models():
    assert ump._is_chat(_model(tool_call=True))
    assert ump._is_chat(_model(tool_call=False, structured_output=True))
    assert ump._is_chat(_model(tool_call=False, reasoning=True))


def test_is_chat_rejects_embeddings_and_non_text_output():
    # embedding: text out but no generative capability
    assert not ump._is_chat(_model(tool_call=False))
    # realtime / image / audio out
    assert not ump._is_chat(_model(output=("text", "audio")))
    assert not ump._is_chat(_model(output=("image",)))


# --- collect --------------------------------------------------------------


def test_collect_filters_by_prefix_and_chat_and_maps_cost():
    source = {
        "alibaba": {
            "models": {
                "qwen-flash": _model(cost=(0.05, 0.4), desc="fast qwen"),
                "qwen-embed": _model(tool_call=False, cost=(0.01, 0.0)),  # not chat
                "kimi-k2": _model(cost=(1.0, 2.0)),  # wrong prefix
            }
        }
    }
    spec = ump.ProviderSpec("qwen", "QwenModel", "alibaba", id_prefix="qwen")
    prices, blurbs = ump.collect(spec, source)
    assert prices == {"qwen-flash": (0.05, 0.4)}
    assert blurbs == {"qwen-flash": "fast qwen"}


def test_collect_skips_models_without_full_cost():
    source = {"openai": {"models": {"m": _model(cost=None)}}}
    spec = ump.ProviderSpec("openai", "OpenAIModel", "openai")
    prices, _ = ump.collect(spec, source)
    assert prices == {}


# --- description line fitting ---------------------------------------------


def test_fit_desc_keeps_line_within_limit():
    long_id = "us.anthropic.claude-some-very-long-model-identifier-v1:0"
    desc = "a very long marketing description " * 5
    fitted = ump._fit_desc(long_id, desc)
    line = f'        "{long_id}": "{fitted}",'
    assert len(line) <= ump.LINE_LIMIT


# --- AST rewrite ----------------------------------------------------------

_REPLACE_SRC = """\
from __future__ import annotations

from typing import ClassVar


class M:
    provider = "x"
    pricing: ClassVar[dict[str, tuple[float, float]]] = {
        "old": (1.0, 2.0),
    }
"""

_INSERT_SRC = """\
from __future__ import annotations

from plugins.models.openai_compatible import OpenAICompatibleModel


class M(OpenAICompatibleModel):
    provider = "x"
    api_key_env = "X_API_KEY"
"""


def test_rewrite_replaces_existing_dict(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(_REPLACE_SRC)
    rendered = ump._render_pricing({"a": (3.0, 4.0), "b": (5.0, 6.0)})
    ump._rewrite_dict(p, "M", "pricing", rendered)
    pricing, _ = ump._current_dicts(p, "M")
    assert pricing == {"a": (3.0, 4.0), "b": (5.0, 6.0)}  # old replaced
    ast.parse(p.read_text())  # still valid Python


def test_rewrite_inserts_dict_and_classvar_import(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(_INSERT_SRC)
    ump._rewrite_dict(p, "M", "pricing", ump._render_pricing({"a": (1.0, 2.0)}))
    text = p.read_text()
    ast.parse(text)  # valid Python
    assert "from typing import ClassVar" in text  # import injected
    pricing, _ = ump._current_dicts(p, "M")
    assert pricing == {"a": (1.0, 2.0)}


def test_rewrite_is_idempotent(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(_INSERT_SRC)
    rendered = ump._render_pricing({"a": (1.0, 2.0)})
    ump._rewrite_dict(p, "M", "pricing", rendered)
    once = p.read_text()
    # re-render from what we just wrote and rewrite again -> no drift
    pricing, _ = ump._current_dicts(p, "M")
    ump._rewrite_dict(p, "M", "pricing", ump._render_pricing(pricing))
    assert p.read_text() == once


def test_render_pricing_lines_within_limit():
    d = {"us.anthropic.claude-3-5-haiku-20241022-v1:0": (0.8, 4.0)}
    for line in ump._render_pricing(d):
        assert len(line) <= ump.LINE_LIMIT
