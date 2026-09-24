from __future__ import annotations

import re
from typing import Any

from tests.e2e.runner import CaseResult
from tests.e2e.spec import Expect


def evaluate(expect: Expect, result: CaseResult) -> list[str]:
    """Run every active check and return a list of human-readable failures.

    An empty list means the case passed. Checks are independent and all run, so
    one report lists every problem at once.
    """
    failures: list[str] = []

    # --- behavioral -------------------------------------------------------
    if expect.no_error and result.error is not None:
        failures.append(f"run errored: {result.error}")

    if expect.stop_reason is not None and result.stop_reason != expect.stop_reason:
        failures.append(
            f"stop_reason: expected {expect.stop_reason!r}, got {result.stop_reason!r}"
        )

    if (
        expect.stop_reason_contains is not None
        and expect.stop_reason_contains not in result.stop_reason
    ):
        failures.append(
            f"stop_reason {result.stop_reason!r} does not contain "
            f"{expect.stop_reason_contains!r}"
        )

    if expect.max_cost_usd is not None and result.cost > expect.max_cost_usd:
        failures.append(
            f"cost ${result.cost:.6f} exceeds max ${expect.max_cost_usd:.6f}"
        )

    if expect.max_iterations is not None and result.model_calls > expect.max_iterations:
        failures.append(
            f"iterations {result.model_calls} exceeds max {expect.max_iterations}"
        )

    if expect.tool_names is not None:
        got_names = sorted(c.get("name", "") for c in result.tool_calls)
        want_names = sorted(expect.tool_names)
        if got_names != want_names:
            failures.append(f"tool_names: expected {want_names}, got {got_names}")

    failures.extend(_check_tool_calls(expect, result))

    for want in expect.tool_denied:
        if not any(_matches(want, c) for c in result.denied):
            failures.append(
                f"expected denied tool call {want} not found among {result.denied}"
            )

    # --- text matching ----------------------------------------------------
    out = result.transcript  # all turns, so a fact from an earlier turn counts
    for needle in expect.output_contains:
        if needle not in out:
            failures.append(f"output missing required substring {needle!r}")

    if expect.output_contains_any and not any(
        n in out for n in expect.output_contains_any
    ):
        failures.append(f"output contains none of {expect.output_contains_any}")

    for needle in expect.output_not_contains:
        if needle in out:
            failures.append(f"output contains forbidden substring {needle!r}")

    if expect.output_regex is not None and not re.search(expect.output_regex, out):
        failures.append(f"output does not match regex {expect.output_regex!r}")

    return failures


def _check_tool_calls(expect: Expect, result: CaseResult) -> list[str]:
    """Match each expected tool call against the recorded calls.

    Unordered (default): every expectation must match some call. Ordered: the
    expectations must appear as an in-order subsequence of the calls.
    """
    if not expect.tool_calls:
        return []
    calls = result.tool_calls

    if expect.tool_calls_ordered:
        i = 0
        for want in expect.tool_calls:
            while i < len(calls) and not _matches(want, calls[i]):
                i += 1
            if i >= len(calls):
                return [f"tool call {want} not found in order within {calls}"]
            i += 1
        return []

    return [
        f"expected tool call {want} not found among {calls}"
        for want in expect.tool_calls
        if not any(_matches(want, c) for c in calls)
    ]


def _matches(want: dict[str, Any], call: dict[str, Any]) -> bool:
    """Whether ``call`` matches the ``name`` and every ``arguments_contains`` pair."""
    if "name" in want and call.get("name") != want["name"]:
        return False
    contains = want.get("arguments_contains", {})
    args = call.get("arguments", {})
    return all(args.get(k) == v for k, v in contains.items())
