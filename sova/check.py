"""Run all checks: formatting, linting, and tests."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    venv = Path(sys.executable).parent
    checks = [
        ("fmt", [str(venv / "ruff"), "format", "--check", "."]),
        ("lint", [str(venv / "ruff"), "check", "."]),
        ("types", [str(venv / "ty"), "check"]),
        ("test", [sys.executable, "-m", "pytest", "tests/", "-q"]),
    ]

    failed = []
    for name, cmd in checks:
        header = f"  {name}"
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FAIL {header}")
            # Show output on failure
            if result.stdout.strip():
                for line in result.stdout.strip().splitlines():
                    print(f"  {line}")
            if result.stderr.strip():
                for line in result.stderr.strip().splitlines():
                    print(f"  {line}")
            failed.append(name)
        else:
            # For pytest, grab the summary line
            summary = ""
            if name == "test":
                lines = result.stdout.strip().splitlines()
                summary = lines[-1] if lines else ""
            print(f"  ok {header}" + (f"  {summary}" if summary else ""))

    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("all checks passed")
