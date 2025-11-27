#!/usr/bin/env python3
"""
Utility runner for scripts in training/.

Allows printing the console output of one or more training/evaluation
scripts without typing each python command manually.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import os
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"


def list_training_scripts() -> List[Path]:
    """Return sorted list of python scripts inside training/."""
    return sorted(TRAINING_DIR.glob("*.py"))


def run_script(script_path: Path) -> int:
    """Execute a script and stream its stdout/stderr to the terminal."""
    print(f"\n=== Running {script_path.relative_to(PROJECT_ROOT)} ===")
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    new_pythonpath = str(PROJECT_ROOT)
    if existing:
        new_pythonpath = f"{new_pythonpath}{os.pathsep}{existing}"
    env["PYTHONPATH"] = new_pythonpath

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        env=env,
    )
    print(f"=== Finished {script_path.name} (exit={result.returncode}) ===\n")
    return result.returncode


def run_training_scripts(script_names: Iterable[str] | None = None) -> None:
    """
    Run selected scripts from training/ and print their output.

    Args:
        script_names: Optional iterable of filenames to execute. If omitted,
                      every *.py file in training/ runs in alphabetical order.
    """
    scripts = list_training_scripts()

    if script_names:
        lookup = {path.name: path for path in scripts}
        try:
            scripts = [lookup[name] for name in script_names]
        except KeyError as missing:
            raise SystemExit(f"Script not found: {missing}") from None

    if not scripts:
        raise SystemExit("No training scripts found.")

    for script in scripts:
        code = run_script(script)
        if code != 0:
            print(f"Stopping because {script.name} exited with {code}")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run training scripts and print their output."
    )
    parser.add_argument(
        "scripts",
        nargs="*",
        help="Specific script filenames to run (default: all in training/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training_scripts(args.scripts or None)


if __name__ == "__main__":
    main()

