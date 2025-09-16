#!/usr/bin/env python3
"""Run a Python module inside the repository local .venv.

Usage:
  python scripts/run_in_venv.py <module> [args...]

This script locates a .venv directory at the repository root and executes
the given module with that venv's python (via -m). It's used by pre-commit
hooks to ensure tools run inside the project's virtualenv.
"""

import subprocess
import sys
from pathlib import Path


def find_venv_python(root: Path) -> Path | None:
    candidates = [
        root / ".venv" / "bin" / "python",
        root / ".venv" / "bin" / "python3",
        root / ".venv" / "Scripts" / "python.exe",
        root / "venv" / "bin" / "python",
        root / "venv" / "Scripts" / "python.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: run_in_venv.py <module> [args...]")
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    venv_python = find_venv_python(repo_root)
    if venv_python is None:
        print("Could not find .venv Python under repository root; ensure .venv exists and is populated.")
        return 2

    module = argv[1]
    mod_args = argv[2:]

    cmd = [str(venv_python), "-m", module] + mod_args

    # Run the tool and stream output
    proc = subprocess.run(cmd)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
