#!/usr/bin/env python3
"""Run the repository's local formatting, linting, typing, and test gates."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Command:
    """One quality command and its user-facing label."""

    label: str
    arguments: tuple[str, ...]


def _display_command(arguments: tuple[str, ...]) -> str:
    return subprocess.list2cmdline(arguments) if os.name == "nt" else shlex.join(arguments)


def _available_path(*parts: str) -> str | None:
    path = REPOSITORY_ROOT.joinpath(*parts)
    return str(path.relative_to(REPOSITORY_ROOT)) if path.exists() else None


def _commands(action: str, *, skip_tests: bool) -> list[Command]:
    if action == "fix":
        commands = [
            Command("Ruff lint fixes", ("ruff", "check", "--fix", ".")),
            Command("Ruff formatting", ("ruff", "format", ".")),
        ]
    else:
        commands = [
            Command("Ruff format check", ("ruff", "format", "--check", ".")),
            Command("Ruff lint", ("ruff", "check", ".")),
        ]

    type_targets = tuple(
        target
        for target in (
            _available_path("src", "number_plate_recognition"),
            _available_path("app"),
            _available_path("tests"),
        )
        if target is not None
    )
    if type_targets:
        commands.append(Command("mypy", ("mypy", *type_targets)))
    commands.append(
        Command(
            "model manifest",
            (sys.executable, "scripts/doctor.py", "--manifest-only"),
        )
    )
    if not skip_tests:
        commands.append(Command("pytest", ("pytest",)))
    return commands


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("check", "fix"),
        nargs="?",
        default="check",
        help="check files without edits, or apply safe Ruff fixes and formatting",
    )
    parser.add_argument("--skip-tests", action="store_true", help="omit the pytest gate")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="run remaining gates after a failure",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run selected gates and return a process exit code."""
    arguments = build_parser().parse_args(argv)
    commands = _commands(arguments.action, skip_tests=arguments.skip_tests)

    missing_tools = sorted(
        {
            command.arguments[0]
            for command in commands
            if command.arguments[0] != sys.executable and shutil.which(command.arguments[0]) is None
        }
    )
    if missing_tools:
        print(
            "Missing quality tools: "
            + ", ".join(missing_tools)
            + ". Run this command through `uv run` after bootstrapping.",
            file=sys.stderr,
        )
        return 2

    failures = 0
    for command in commands:
        print(f"\n==> {command.label}: {_display_command(command.arguments)}", flush=True)
        # Every argv vector is an internal constant; no shell or user command is accepted.
        completed = subprocess.run(  # noqa: S603
            command.arguments,
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        if completed.returncode == 0:
            continue
        failures += 1
        print(
            f"Gate failed with exit code {completed.returncode}: {command.label}",
            file=sys.stderr,
        )
        if not arguments.keep_going:
            break

    if failures:
        print(f"\nQuality checks failed ({failures} gate(s)).", file=sys.stderr)
        return 1
    print("\nAll selected quality checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
