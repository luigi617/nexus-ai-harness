from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from swebench.harness.run_evaluation import main as run_swebench_evaluation

from benchmarks.core.benchmark import Benchmark, Episode
from benchmarks.core.registry import register
from benchmarks.core.task import Score, Task
from harness import NexusAIHarness
from harness.result import RunResult
from harness.session import Session
from plugins.guards import MaxIterations, Timeout
from plugins.hooks import ElapsedTime, IterationCounter
from plugins.loops import AgenticLoop
from plugins.permissions import AutoApprove
from plugins.sandbox import WorkspaceSandbox
from plugins.tools import ListDir, ReadFile, Shell, WriteFile
from protocols.lifecycle import Lifecycle
from protocols.model import Model
from protocols.plugin import Plugin

_MAX_STEPS = int(os.getenv("SWE_BENCH_MAX_STEPS", "50"))
_TIMEOUT_S = float(os.getenv("SWE_BENCH_TIMEOUT_S", "1800"))
# The agent's own shell runs on the host, not in the grading container; allow it
# network by default so it can explore/build, override with SWE_BENCH_ALLOW_NETWORK=0.
_ALLOW_NETWORK = os.getenv("SWE_BENCH_ALLOW_NETWORK", "1") != "0"
_MODEL_NAME = "nexus-ai-harness"


@dataclass
class SWEState:
    """Where the agent's checkout lives, so the grader can diff and evaluate it."""

    checkout: str = ""


class _WorkspaceCleanup(Plugin, Lifecycle):
    """Deletes the per-task workspace when the harness stops (teardown contract)."""

    def __init__(self, workspace: str) -> None:
        self._workspace = workspace

    async def stop(self) -> None:
        """Remove the temp workspace; best-effort so teardown never raises."""
        shutil.rmtree(self._workspace, ignore_errors=True)


@register
class SWEBench(Benchmark):
    """SWE-bench: the agent edits a real repo to fix an issue; graded by its tests.

    The production ``AgenticLoop`` drives the fix against a sandboxed git
    checkout; grading runs the repository's FAIL_TO_PASS / PASS_TO_PASS tests
    through the official ``swebench`` Docker harness, so a pass means the project's
    own tests pass on the model's patch.
    """

    name = "swe-bench"
    description = (
        "SWE-bench: fix a real GitHub issue; graded by the repo's "
        "FAIL_TO_PASS/PASS_TO_PASS tests"
    )

    def load_tasks(self, *, limit: int | None = None) -> list[Task]:
        """Load SWE-bench rows from the configured dataset into tasks."""
        dataset = load_dataset(
            os.getenv("SWE_BENCH_DATASET", "princeton-nlp/SWE-bench_Verified"),
            split="test",
        )
        rows = (
            dataset
            if limit is None
            else dataset.select(range(min(limit, len(dataset))))
        )
        return [
            Task(
                task_id=row["instance_id"],
                prompt=row["problem_statement"],
                metadata={
                    "repo": row["repo"],
                    "base_commit": row["base_commit"],
                    "test_patch": row["test_patch"],
                    "gold_patch": row.get("patch"),  # reference only, never applied
                    "fail_to_pass": _extract_test_ids(row.get("FAIL_TO_PASS")),
                    "pass_to_pass": _extract_test_ids(row.get("PASS_TO_PASS")),
                    "environment_setup_commit": row.get("environment_setup_commit"),
                    "version": row.get("version"),
                },
            )
            for row in rows
        ]

    def build_harness(self, task: Task, model: Model, session: Session) -> Episode:
        """Check out the repo at its base commit and compose an agentic harness."""
        meta = task.metadata
        workspace = tempfile.mkdtemp(prefix="swe-bench-")
        checkout = os.path.join(workspace, "repo")
        # Blocking clone is fine: build_harness runs off the event loop.
        _clone_repo(meta["repo"], meta["base_commit"], checkout)
        session.state(SWEState).checkout = checkout

        harness = (
            NexusAIHarness()
            .use(AgenticLoop())  # the real production loop under test
            .use(model)
            .use(AutoApprove())
            .use(IterationCounter())
            .use(ElapsedTime())
            .use(MaxIterations(_MAX_STEPS))
            .use(Timeout(_TIMEOUT_S))
            .use(WorkspaceSandbox(checkout, allow_network=_ALLOW_NETWORK))
            .use(ReadFile())
            .use(WriteFile())
            .use(ListDir())
            .use(Shell())
            .use(_WorkspaceCleanup(workspace))  # stop() rmtrees after grading
        )
        return Episode(harness=harness, initial_input=_render_prompt(task))

    async def grade(self, task: Task, result: RunResult) -> Score:
        """Diff the checkout and run the repo's tests via the swebench harness.

        Grading happens before teardown, so the checkout is still present. The
        Docker evaluation is offloaded to a worker thread: the runner calls
        ``grade`` on the event loop, and the swebench harness blocks for minutes
        building images and running containers.
        """
        checkout = result.session.state(SWEState).checkout
        if not checkout:
            raise RuntimeError(
                "no checkout recorded in session state; build_harness must run first"
            )
        model_patch = _git_diff(checkout)
        report = await asyncio.to_thread(_run_swebench_evaluation, task, model_patch)
        return _score_from_report(task.task_id, report)


# --- testable pure helpers -------------------------------------------------


def _extract_test_ids(raw: object) -> list[str]:
    """Parse a FAIL_TO_PASS / PASS_TO_PASS field into a list of test ids.

    The dataset encodes these as a JSON list string, but an already-decoded list
    is accepted too so a caller that pre-parsed the row still works.

    Args:
        raw: A JSON list string, a list, or an empty/``None`` value.

    Returns:
        The test ids as strings (empty when the field is blank).
    """
    if raw is None or raw == "":
        return []
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    return [str(item) for item in parsed]


def _git_diff(cwd: str) -> str:
    """Return the agent's changes at ``cwd`` as a unified diff (the model patch).

    Stages everything first (``git add -A``) so newly created files appear, then
    diffs the index against ``HEAD`` (the base commit the checkout started at).

    Args:
        cwd: Path to the git checkout the agent edited.

    Returns:
        The unified diff of the working tree against the base commit.
    """
    subprocess.run(
        ["git", "-C", cwd, "add", "-A"],
        check=True,
        capture_output=True,
        text=True,
    )
    proc = subprocess.run(
        ["git", "-C", cwd, "diff", "--cached"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def _score_from_report(instance_id: str, report: dict) -> Score:
    """Decide pass/fail for ``instance_id`` from a swebench evaluation report.

    Accepts either swebench's run-level summary (with ``resolved_ids`` /
    ``unresolved_ids`` lists) or a per-instance mapping
    (``{instance_id: {"resolved": bool, ...}}``). "Resolved" means every
    FAIL_TO_PASS and PASS_TO_PASS test passed, which counts as a pass.

    Args:
        instance_id: The SWE-bench instance being graded.
        report: The report produced by the swebench harness.

    Returns:
        A passed score iff ``instance_id`` is resolved in the report.
    """
    resolved_ids = report.get("resolved_ids")
    if resolved_ids is not None:
        resolved = instance_id in set(resolved_ids)
        return Score(
            passed=resolved,
            detail={"instance_id": instance_id, "resolved": resolved},
        )
    instance = report.get(instance_id)
    if isinstance(instance, dict):
        resolved = bool(instance.get("resolved", False))
        return Score(
            passed=resolved,
            detail={
                "instance_id": instance_id,
                "resolved": resolved,
                "tests_status": instance.get("tests_status", {}),
            },
        )
    # Absent from the report -> swebench errored/skipped it; not a pass.
    return Score(
        passed=False,
        detail={
            "instance_id": instance_id,
            "resolved": False,
            "reason": "instance not present in report",
        },
    )


# --- environment + grading side effects ------------------------------------


def _render_prompt(task: Task) -> str:
    """Build the agent's instructions: fix the issue by editing the repo files."""
    return (
        "You are fixing a bug in a checked-out GitHub repository. The repository "
        "root is your workspace root; use the tools to read, edit, and run files.\n\n"
        "Resolve the issue below by editing the repository's source files. Do NOT "
        "edit or add test files: the fix is graded by the project's own test "
        "suite, which is applied separately.\n\n"
        f"Repository: {task.metadata['repo']}\n\n"
        f"Issue:\n{task.prompt}"
    )


def _clone_repo(repo: str, base_commit: str, dest: str) -> None:
    """Clone ``repo`` from GitHub into ``dest`` and check out ``base_commit``.

    Args:
        repo: The ``owner/name`` slug of the GitHub repository.
        base_commit: The commit the agent should start editing from.
        dest: The directory to clone into.
    """
    url = f"https://github.com/{repo}.git"
    subprocess.run(
        ["git", "clone", url, dest], check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "-C", dest, "checkout", base_commit],
        check=True,
        capture_output=True,
        text=True,
    )


def _run_swebench_evaluation(task: Task, model_patch: str) -> dict:
    """Run the official swebench harness on the model's patch and return its report.

    Requires a running Docker daemon (the swebench harness runs each test inside
    a container); it fails with swebench's own error if one is not reachable.

    Args:
        task: The graded task (its id selects the dataset instance).
        model_patch: The unified diff the agent produced.

    Returns:
        The parsed swebench report for the instance.
    """
    dataset = os.getenv("SWE_BENCH_DATASET", "princeton-nlp/SWE-bench_Verified")
    run_id = f"nexus-{task.task_id}"
    work = tempfile.mkdtemp(prefix="swe-bench-eval-")
    try:
        prediction = {
            "instance_id": task.task_id,
            "model_name_or_path": _MODEL_NAME,
            "model_patch": model_patch,
        }
        preds_path = os.path.join(work, "predictions.json")
        Path(preds_path).write_text(json.dumps([prediction]))
        run_swebench_evaluation(
            dataset_name=dataset,
            split="test",
            instance_ids=[task.task_id],
            predictions_path=preds_path,
            max_workers=1,
            force_rebuild=False,
            cache_level="env",
            clean=False,
            open_file_limit=4096,
            run_id=run_id,
            timeout=1800,
            report_dir=work,
        )
        return _load_report(run_id, [work, os.getcwd()])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _load_report(run_id: str, search_dirs: list[str]) -> dict:
    """Find and parse the swebench report written for ``run_id``.

    Args:
        run_id: The run id passed to the swebench harness.
        search_dirs: Directories to look in (the harness's report location has
            varied across versions).

    Returns:
        The parsed report dict.

    Raises:
        RuntimeError: If no report file is found in any search directory.
    """
    name = f"{_MODEL_NAME}.{run_id}.json"
    for directory in search_dirs:
        candidate = Path(directory) / name
        if candidate.exists():
            return json.loads(candidate.read_text())
    raise RuntimeError(
        f"swebench evaluation produced no report {name!r} in {search_dirs}"
    )
