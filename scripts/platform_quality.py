#!/usr/bin/env python3
"""Run deterministic quality gates for the ML app, control API, and web console."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTROL_API_ROOT = REPOSITORY_ROOT / "services" / "control_api"
CONSOLE_ROOT = REPOSITORY_ROOT / "web" / "console"

Scope = Literal["all", "root", "service", "frontend", "scripts"]


@dataclass(frozen=True, slots=True)
class Command:
    """One fixed command executed without a shell."""

    label: str
    arguments: tuple[str, ...]
    working_directory: Path = REPOSITORY_ROOT


def _display(command: Command) -> str:
    rendered = (
        subprocess.list2cmdline(command.arguments)
        if os.name == "nt"
        else shlex.join(command.arguments)
    )
    relative_directory = command.working_directory.relative_to(REPOSITORY_ROOT)
    location = "." if str(relative_directory) == "." else str(relative_directory)
    return f"({location}) {rendered}"


def _basetemp(prefix: str) -> str:
    runtime = REPOSITORY_ROOT / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    return str(runtime / f"pytest-{prefix}-{uuid.uuid4().hex}")


def _root_commands(*, sync: bool) -> list[Command]:
    targets = (
        "src/number_plate_recognition",
        "app",
        "scripts",
        "services/inference_worker",
        "tests/unit",
        "tests/conftest.py",
        "tests/integration",
        "tests/smoke",
        "tests/model",
        "tests/platform_inference",
    )
    commands = [Command("Root lock", ("uv", "lock", "--check"))]
    if sync:
        commands.append(
            Command("Root environment sync", ("uv", "sync", "--locked", "--group", "dev"))
        )
    commands.extend(
        [
            Command(
                "Root Ruff format",
                ("uv", "run", "--frozen", "ruff", "format", "--check", *targets),
            ),
            Command("Root Ruff lint", ("uv", "run", "--frozen", "ruff", "check", *targets)),
            Command("Root mypy", ("uv", "run", "--frozen", "mypy", *targets)),
            Command(
                "Model manifest",
                ("uv", "run", "--frozen", "python", "scripts/doctor.py", "--manifest-only"),
            ),
            Command(
                "Platform doctor",
                ("uv", "run", "--frozen", "python", "scripts/platform_doctor.py"),
            ),
            Command(
                "Root pytest",
                (
                    "uv",
                    "run",
                    "--frozen",
                    "pytest",
                    "tests/unit",
                    "tests/integration",
                    "tests/smoke",
                    "tests/model",
                    "tests/platform_inference",
                    "--basetemp",
                    _basetemp("root"),
                ),
            ),
        ]
    )
    return commands


def _service_commands(*, sync: bool) -> list[Command]:
    commands = [
        Command(
            "Control API lock",
            ("uv", "lock", "--check"),
            CONTROL_API_ROOT,
        )
    ]
    if sync:
        commands.append(
            Command(
                "Control API environment sync",
                ("uv", "sync", "--locked", "--group", "dev"),
                CONTROL_API_ROOT,
            )
        )
    service_targets = ("control_api", "../../tests/platform_backend")
    commands.extend(
        [
            Command(
                "Control API Ruff format",
                ("uv", "run", "--frozen", "ruff", "format", "--check", *service_targets),
                CONTROL_API_ROOT,
            ),
            Command(
                "Control API Ruff lint",
                ("uv", "run", "--frozen", "ruff", "check", *service_targets),
                CONTROL_API_ROOT,
            ),
            Command(
                "Control API mypy",
                ("uv", "run", "--frozen", "mypy", *service_targets),
                CONTROL_API_ROOT,
            ),
            Command(
                "Control API pytest",
                (
                    "uv",
                    "run",
                    "--frozen",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--basetemp",
                    _basetemp("control-api"),
                ),
                CONTROL_API_ROOT,
            ),
        ]
    )
    return commands


def _frontend_commands() -> list[Command]:
    version_check = (
        "const major=Number(process.versions.node.split('.')[0]);"
        "if(major<18){console.error('Node.js 18 or newer is required');process.exit(1)};"
        "console.log('Node.js '+process.versions.node)"
    )
    commands = [
        Command("Node.js version", ("node", "--eval", version_check), CONSOLE_ROOT),
    ]
    commands.extend(
        Command(f"Web syntax: {module}", ("node", "--check", module), CONSOLE_ROOT)
        for module in (
            "app.mjs",
            "api.mjs",
            "config.mjs",
            "core.mjs",
            "demo-data.mjs",
            "i18n.mjs",
        )
    )
    commands.append(
        Command(
            "Web console tests",
            (
                "node",
                "--test",
                "tests/core.test.mjs",
                "tests/api.test.mjs",
                "tests/static.test.mjs",
            ),
            CONSOLE_ROOT,
        )
    )
    return commands


def _powershell_parser_command(executable: str) -> Command:
    scripts = (
        "scripts/bootstrap.ps1",
        "scripts/bootstrap_platform.ps1",
        "scripts/run_app.ps1",
        "scripts/run_platform.ps1",
    )
    quoted = ",".join(f"'{script}'" for script in scripts)
    parser = (
        "$allErrors=@();"
        f"foreach($path in @({quoted})){{"
        "$tokens=$null;$errors=$null;"
        "[System.Management.Automation.Language.Parser]::ParseFile("
        "(Resolve-Path -LiteralPath $path),[ref]$tokens,[ref]$errors)|Out-Null;"
        "$allErrors+=$errors};"
        "if($allErrors.Count -gt 0){$allErrors|Format-List;exit 1}"
    )
    return Command("PowerShell syntax", (executable, "-NoProfile", "-Command", parser))


def _script_commands(*, require_shells: bool) -> list[Command]:
    commands: list[Command] = []
    bash = _find_bash()
    if bash:
        commands.append(
            Command(
                "Shell syntax",
                (
                    bash,
                    "-n",
                    "scripts/bootstrap.sh",
                    "scripts/bootstrap_platform.sh",
                    "scripts/run_app.sh",
                    "scripts/run_platform.sh",
                    "docker/entrypoint.sh",
                ),
            )
        )
    elif require_shells:
        raise RuntimeError("bash is required for shell syntax validation")
    else:
        print("Skipping shell syntax validation: bash is unavailable.")

    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell:
        commands.append(_powershell_parser_command(powershell))
    elif require_shells:
        raise RuntimeError("PowerShell is required for script syntax validation")
    else:
        print("Skipping PowerShell syntax validation: pwsh is unavailable.")
    return commands


def _find_bash() -> str | None:
    """Prefer Git Bash on Windows instead of the optional WSL launcher."""

    if os.name != "nt":
        return shutil.which("bash")
    git = shutil.which("git")
    if git:
        git_root = Path(git).resolve().parent.parent
        for relative_path in (("bin", "bash.exe"), ("usr", "bin", "bash.exe")):
            candidate = git_root.joinpath(*relative_path)
            if candidate.is_file():
                return str(candidate)
    return None


def _commands(scope: Scope, *, sync: bool, require_shells: bool) -> list[Command]:
    commands: list[Command] = []
    if scope in {"all", "root"}:
        commands.extend(_root_commands(sync=sync))
    if scope in {"all", "service"}:
        commands.extend(_service_commands(sync=sync))
    if scope in {"all", "frontend"}:
        commands.extend(_frontend_commands())
    if scope in {"all", "scripts"}:
        commands.extend(_script_commands(require_shells=require_shells))
    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("check",), default="check")
    parser.add_argument(
        "--scope",
        choices=("all", "root", "service", "frontend", "scripts"),
        default="all",
        help="run every gate or one project boundary",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="synchronize both selected uv environments from their locks before checking",
    )
    parser.add_argument(
        "--require-script-shells",
        action="store_true",
        help="fail instead of skipping when bash or PowerShell is unavailable",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="run remaining gates after a failure",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        commands = _commands(
            arguments.scope,
            sync=arguments.sync,
            require_shells=arguments.require_script_shells,
        )
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2

    missing_tools = sorted(
        {command.arguments[0] for command in commands if shutil.which(command.arguments[0]) is None}
    )
    if missing_tools:
        print(f"Missing required tools: {', '.join(missing_tools)}", file=sys.stderr)
        return 2

    failures = 0
    for command in commands:
        print(f"\n==> {command.label}: {_display(command)}", flush=True)
        command_environment = os.environ.copy()
        if command.working_directory == CONTROL_API_ROOT:
            # A runner started through the root uv environment must not make nested uv
            # interpret that environment as the control API's project environment.
            command_environment.pop("VIRTUAL_ENV", None)
        try:
            completed = subprocess.run(  # noqa: S603
                command.arguments,
                cwd=command.working_directory,
                env=command_environment,
                check=False,
            )
        except OSError as error:
            failures += 1
            print(f"Gate could not start: {command.label} ({error})", file=sys.stderr)
            if not arguments.keep_going:
                break
            continue
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
        print(f"\nPlatform quality failed ({failures} gate(s)).", file=sys.stderr)
        return 1
    print("\nAll selected platform quality gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
