from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import MISSING, Field, fields

from benchmarks.core.config import load_config_file
from benchmarks.core.models import build_model
from benchmarks.core.registry import get_benchmark, registered_benchmarks
from benchmarks.core.runner import Runner


def _cmd_list(_: argparse.Namespace) -> int:
    for name, cls in sorted(registered_benchmarks().items()):
        print(f"{name:<12} {cls.description}")
        knobs = ", ".join(f"{f.name}={_default(f)}" for f in fields(cls.config_type))
        if knobs:  # show the --set knobs and their defaults for discoverability
            print(f"{'':<12} config: {knobs}")
    return 0


def _default(field: Field) -> object:
    """The field's default value, resolving a default_factory when present."""
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:
        return field.default_factory()
    return None


def _parse_overrides(pairs: list[str]) -> dict[str, str]:
    """Parse ``key=value`` CLI overrides into a dict, failing on malformed input."""
    overrides: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        overrides[key.strip()] = value
    return overrides


async def _run(args: argparse.Namespace) -> int:
    try:
        # File first, then --set on top: an explicit CLI override wins over the
        # file. FileNotFoundError etc. surface as a clean SystemExit too.
        file_cfg = load_config_file(args.config, args.benchmark) if args.config else {}
        overrides = {**file_cfg, **_parse_overrides(args.set)}
        benchmark = get_benchmark(args.benchmark, overrides)
    except OSError as exc:  # str(exc) is the readable message; args[0] is the errno
        raise SystemExit(str(exc)) from exc
    except (KeyError, ValueError) as exc:  # args[0] avoids KeyError's quote-wrapping
        raise SystemExit(exc.args[0] if exc.args else str(exc)) from exc
    model = build_model(args.model, max_tokens=args.max_tokens)
    runner = Runner(
        benchmark,
        model,
        k=args.k,
        concurrency=args.concurrency,
        task_timeout_s=args.task_timeout,
        output_path=args.output,
        resume=args.resume,
    )
    report = await runner.run(limit=args.limit)
    print(report.render())
    if args.json:
        print(json.dumps(report.summary()))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    return asyncio.run(_run(args))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list registered benchmarks").set_defaults(
        func=_cmd_list
    )

    run = sub.add_parser("run", help="run a benchmark")
    run.add_argument("benchmark", help="benchmark name (see `list`)")
    run.add_argument(
        "--model",
        default="bedrock:us.anthropic.claude-opus-4-8",
        help="provider:model-id",
    )
    run.add_argument("--limit", type=int, default=None, help="max tasks")
    run.add_argument("--k", type=int, default=1, help="attempts per task (pass^k)")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--task-timeout", type=float, default=600.0)
    run.add_argument("--max-tokens", type=int, default=1024)
    run.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="YAML file of per-benchmark config (this benchmark's section is used)",
    )
    run.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config field (repeatable); wins over --config; see `list`",
    )
    run.add_argument("--output", default=None, help="JSONL results path")
    run.add_argument(
        "--resume", action="store_true", help="skip attempts already in --output"
    )
    run.add_argument("--json", action="store_true", help="also print summary as JSON")
    run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
