"""Run a fast project smoke check.

Default mode validates imports and key API paths quickly.
Use --full to run the complete test suite.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_step(label: str, cmd: list[str]) -> int:
    print(f"\n[{label}] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode == 0:
        print(f"[OK] {label}")
    else:
        print(f"[FAIL] {label} (exit={result.returncode})")
    return result.returncode


def main() -> int:
    full_mode = "--full" in sys.argv[1:]

    steps: list[tuple[str, list[str]]] = [
        ("Import verification", [sys.executable, "verify_production_imports.py"]),
        (
            "Core pytest smoke",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_engine.py",
                "tests/test_web_api.py",
                "tests/test_qa_solver_rag.py",
            ],
        ),
    ]

    if full_mode:
        steps.append(("Full pytest suite", [sys.executable, "-m", "pytest", "-q"]))

    for label, cmd in steps:
        code = run_step(label, cmd)
        if code != 0:
            return code

    print("\nSmoke check completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())