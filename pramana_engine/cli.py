from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .examples import SCENARIOS
from .io import infer_from_file, infer_many_from_file


def _run_scenario(name: str) -> int:
    builder = SCENARIOS.get(name)
    if builder is None:
        print(f"Unknown scenario: {name}")
        return 1

    engine, request = builder()
    result = engine.infer(request)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _run_all() -> int:
    rows = []
    for name in SCENARIOS:
        engine, request = SCENARIOS[name]()
        result = engine.infer(request)
        rows.append({"scenario": name, "status": result.status.value, "accepted": result.accepted})

    print(json.dumps(rows, indent=2))
    return 0


def _list_scenarios() -> int:
    for name in SCENARIOS:
        print(name)
    return 0


def _run_infer_file(file_path: str) -> int:
    try:
        result = infer_from_file(file_path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc), "file": file_path}, indent=2))
        return 1

    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _run_infer_batch(file_path: str, out_path: str | None) -> int:
    try:
        report = infer_many_from_file(file_path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc), "file": file_path}, indent=2))
        return 1

    if out_path:
        output = Path(out_path)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"ok": True, "report": str(output), "status_counts": report["status_counts"]}, indent=2))
        return 0

    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pramana-engine",
        description="Pramana-constrained inference engine with inspectable reasoning traces.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available inference scenarios")

    infer_parser = subparsers.add_parser("infer", help="Run one inference scenario")
    infer_parser.add_argument("scenario", help="Scenario name")

    infer_file_parser = subparsers.add_parser("infer-file", help="Run inference from a JSON payload file")
    infer_file_parser.add_argument("file_path", help="Absolute or relative path to input JSON")

    infer_batch_parser = subparsers.add_parser("infer-batch", help="Run inference over all records in a JSON file")
    infer_batch_parser.add_argument("file_path", help="Absolute or relative path to input JSON")
    infer_batch_parser.add_argument("--out", dest="out_path", help="Optional report output JSON path")

    subparsers.add_parser("run-all", help="Run all built-in scenarios")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "list":
        return _list_scenarios()
    if args.command == "infer":
        return _run_scenario(args.scenario)
    if args.command == "infer-file":
        return _run_infer_file(args.file_path)
    if args.command == "infer-batch":
        return _run_infer_batch(args.file_path, args.out_path)
    if args.command == "run-all":
        return _run_all()

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
