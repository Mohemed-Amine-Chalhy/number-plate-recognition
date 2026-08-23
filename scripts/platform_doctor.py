#!/usr/bin/env python3
"""Validate the local campus-platform checkout and an optional running API."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Check:
    """One deterministic diagnostic result."""

    name: str
    passed: bool
    detail: str


def _file_check(name: str, relative_path: str) -> Check:
    path = REPOSITORY_ROOT / relative_path
    return Check(name, path.is_file(), relative_path)


def _python_check() -> Check:
    version = sys.version_info
    passed = version.major == 3 and version.minor == 12
    return Check(
        "Python runtime",
        passed,
        f"{version.major}.{version.minor}.{version.micro} (requires 3.12)",
    )


def _sqlite_check() -> Check:
    try:
        version = tuple(int(part) for part in sqlite3.sqlite_version.split("."))
    except ValueError:
        return Check("SQLite runtime", False, sqlite3.sqlite_version)
    return Check("SQLite runtime", version >= (3, 35), sqlite3.sqlite_version)


def _runtime_check() -> Check:
    runtime_directory = REPOSITORY_ROOT / ".runtime" / "platform"
    try:
        runtime_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=runtime_directory, prefix="doctor-", delete=True):
            pass
    except OSError as exc:
        return Check("Runtime directory", False, exc.__class__.__name__)
    return Check("Runtime directory", True, str(runtime_directory.relative_to(REPOSITORY_ROOT)))


def _api_check(base_url: str) -> Check:
    url = f"{base_url.rstrip('/')}/health/ready"
    request = urllib.request.Request(  # noqa: S310 - operator-provided HTTP endpoint is expected
        url,
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:  # noqa: S310
            payload = json.loads(response.read(32_768))
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return Check("Running API", False, f"{url} ({exc.__class__.__name__})")
    passed = response.status == 200 and isinstance(payload, dict)
    return Check("Running API", passed, f"{url} returned HTTP {response.status}")


def run_checks(api_url: str | None = None) -> list[Check]:
    """Run filesystem/runtime checks without importing either application."""

    checks = [
        _python_check(),
        _sqlite_check(),
        _runtime_check(),
        _file_check("Control API project", "services/control_api/pyproject.toml"),
        _file_check("Control API lock", "services/control_api/uv.lock"),
        _file_check("Web console", "web/console/index.html"),
        _file_check("Web application", "web/console/app.mjs"),
        _file_check("Tenant branding", "web/console/config.mjs"),
        _file_check("Inference worker", "services/inference_worker/worker.py"),
        _file_check("Platform documentation", "docs/platform/README.md"),
        _file_check(
            "Case-study video",
            "docs/platform/video/campus-access-case-study-2m-v1.mp4",
        ),
        _file_check("Model manifest", "models/manifest.json"),
    ]
    if api_url is not None:
        checks.append(_api_check(api_url))
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url", help="also test a running API, for example http://localhost:8000"
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    checks = run_checks(arguments.api_url)
    if arguments.json:
        print(json.dumps({"checks": [asdict(check) for check in checks]}, indent=2))
    else:
        for check in checks:
            marker = "PASS" if check.passed else "FAIL"
            print(f"[{marker}] {check.name}: {check.detail}")
    failed = sum(not check.passed for check in checks)
    if failed:
        print(f"Platform doctor found {failed} failed check(s).", file=sys.stderr)
        return 1
    if not arguments.json:
        print("Campus platform environment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
