from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from tests.e2e import Spec, discover_specs, evaluate, run_case

CASES_DIR = Path(__file__).parent / "e2e" / "cases"

# Live cases cost money and are nondeterministic, so they need an explicit opt-in.
LIVE_ENABLED = os.getenv("NEXUS_E2E_LIVE", "").lower() in {"1", "true", "yes"}

_SPECS = discover_specs(CASES_DIR) if CASES_DIR.exists() else []


@pytest.mark.parametrize("spec", _SPECS, ids=lambda s: s.name)
def test_e2e_case(spec: Spec) -> None:
    if spec.skip:
        pytest.skip(f"{spec.name}: skip=true in spec")
    if spec.is_live and not LIVE_ENABLED:
        pytest.skip(f"{spec.name}: live case; set NEXUS_E2E_LIVE=1 to run")

    result = asyncio.run(run_case(spec))
    failures = evaluate(spec.expect, result)
    if failures:
        detail = "\n".join(f"  - {f}" for f in failures)
        pytest.fail(
            f"e2e case {spec.name!r} failed:\n{detail}\n"
            f"(output={result.output!r}, stop_reason={result.stop_reason!r})"
        )
