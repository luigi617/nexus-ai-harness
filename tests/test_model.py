from __future__ import annotations

import importlib
import importlib.util
import json
import re
import sys
import urllib.request
from pathlib import Path

import pytest

from benchmarks.core.models import build_model

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "update_model_pricing", REPO / "scripts" / "update_model_pricing.py"
)
assert _spec is not None and _spec.loader is not None
ump = importlib.util.module_from_spec(_spec)
sys.modules["update_model_pricing"] = ump
_spec.loader.exec_module(ump)


def _spec_classes():
    for spec in ump.SPECS:
        module = importlib.import_module(f"plugins.models.{spec.key}")
        yield spec, getattr(module, spec.class_name)


SPEC_CLASSES = list(_spec_classes())
_IDS = [s.key for s, _ in SPEC_CLASSES]
# Model ids across providers: alphanum plus . _ - : / (e.g. bedrock profiles).
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-/]*$")
_MAX_PER_M = 10_000.0  # anything higher signals a unit slip (e.g. per-token vs per-1M)


@pytest.mark.parametrize("spec,cls", SPEC_CLASSES, ids=_IDS)
def test_pricing_entries_wellformed(spec, cls):
    assert cls.pricing, f"{spec.key} has no pricing entries"
    for mid, price in cls.pricing.items():
        assert _NAME_RE.match(mid), f"{spec.key}: malformed id {mid!r}"
        assert isinstance(price, tuple) and len(price) == 2, f"{spec.key}:{mid}"
        inp, out = price
        assert isinstance(inp, (int, float)) and isinstance(out, (int, float))
        for v in (inp, out):
            assert 0 <= v <= _MAX_PER_M, f"{spec.key}:{mid} price {v}"


@pytest.mark.parametrize("spec,cls", SPEC_CLASSES, ids=_IDS)
def test_every_priced_model_constructs(spec, cls):
    # build_model just wires the id to a backend; it makes no API call.
    for mid in cls.pricing:
        model = build_model(f"{spec.key}:{mid}")
        assert model.name == mid
        assert model.provider == spec.key


def _fetch_source_or_skip():
    try:
        req = urllib.request.Request(
            ump.SOURCE_URL, headers={"User-Agent": "nexus-price-test"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as exc:  # offline -> can't verify presence, not a failure
        pytest.skip(f"models.dev unreachable: {exc}")


@pytest.mark.parametrize("spec,cls", SPEC_CLASSES, ids=_IDS)
def test_every_listed_model_is_present_and_usable(spec, cls):
    """Every listed model exists in models.dev as a usable chat model."""
    source = _fetch_source_or_skip()
    models = (source.get(spec.source_key) or {}).get("models", {})
    for mid in cls.pricing:
        assert mid in models, f"{spec.key}:{mid} is not listed by models.dev"
        assert ump._is_chat(models[mid]), f"{spec.key}:{mid} is not a usable chat model"
