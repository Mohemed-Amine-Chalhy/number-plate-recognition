#!/usr/bin/env python3
"""Verify model artifacts and securely fetch declared missing files."""

from __future__ import annotations

import argparse
import os
import ssl
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from doctor import (
    REPOSITORY_ROOT,
    ManifestError,
    ModelSpec,
    app_path,
    load_manifest,
    model_path,
    repository_path,
    verify_model_file,
)

CHUNK_SIZE = 1024 * 1024


class DownloadError(RuntimeError):
    """Raised when a model cannot be downloaded and verified."""


def _validate_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise DownloadError("download_url must be an absolute HTTPS URL")
    if parsed.username or parsed.password:
        raise DownloadError("download_url must not contain embedded credentials")


def _download_to_temporary_file(
    model: ModelSpec,
    target: Path,
    *,
    timeout_seconds: float,
) -> Path:
    if model.download_url is None:
        raise DownloadError(
            f"{model.role} has no download_url; provision {target.name} through an approved channel"
        )
    _validate_url(model.download_url)
    target.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(  # noqa: S310 - _validate_url requires absolute HTTPS.
        model.download_url,
        headers={"User-Agent": "number-plate-recognition-model-fetcher/1"},
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".download",
            dir=target.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            context = ssl.create_default_context()
            # S310 is safe here because the URL scheme is checked before and after redirects.
            with urllib.request.urlopen(  # noqa: S310 - URL and redirect are HTTPS-checked.
                request,
                timeout=timeout_seconds,
                context=context,
            ) as response:
                final_url = response.geturl()
                _validate_url(final_url)
                downloaded = 0
                while chunk := response.read(CHUNK_SIZE):
                    downloaded += len(chunk)
                    if downloaded > model.size_bytes:
                        raise DownloadError(
                            f"download for {model.role} exceeds declared size {model.size_bytes}"
                        )
                    temporary_file.write(chunk)
    except (OSError, ssl.SSLError, urllib.error.URLError) as exc:
        raise DownloadError(f"download failed for {model.role}: {exc}") from exc
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    if temporary_path is None:
        raise DownloadError(f"download failed for {model.role}: no temporary file was created")
    verification_error = verify_model_file(temporary_path, model)
    if verification_error:
        temporary_path.unlink(missing_ok=True)
        raise DownloadError(f"download verification failed: {verification_error}")
    return temporary_path


def _select_models(models: list[ModelSpec], roles: list[str]) -> list[ModelSpec]:
    if not roles:
        return models
    by_role = {model.role: model for model in models}
    unknown = sorted(set(roles) - by_role.keys())
    if unknown:
        raise ManifestError("unknown model role(s): " + ", ".join(unknown))
    return [by_role[role] for role in dict.fromkeys(roles)]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="manifest path (default: NPR_MODEL_MANIFEST or models/manifest.json)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="artifact directory (default: NPR_MODEL_DIR or models)",
    )
    parser.add_argument(
        "--role",
        action="append",
        default=[],
        help="operate on one declared role; may be supplied more than once",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="never download or replace files; only verify local artifacts",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing invalid artifact after a verified download",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="per-request network timeout in seconds (default: 60)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Verify or fetch requested model artifacts."""
    arguments = build_parser().parse_args(argv)
    if arguments.timeout <= 0:
        print("error: --timeout must be greater than zero", file=sys.stderr)
        return 2

    configured_app_root = repository_path(os.getenv("NPR_APP_ROOT", str(REPOSITORY_ROOT)))
    model_dir_setting = arguments.model_dir or Path(os.getenv("NPR_MODEL_DIR", "models"))
    model_dir = app_path(model_dir_setting, configured_app_root)
    manifest_setting = arguments.manifest or Path(
        os.getenv("NPR_MODEL_MANIFEST", str(model_dir / "manifest.json"))
    )
    manifest_path = app_path(manifest_setting, configured_app_root)

    try:
        selected = _select_models(load_manifest(manifest_path), arguments.role)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    failures = 0
    for model in selected:
        try:
            target = model_path(model_dir, model)
        except ManifestError as exc:
            print(f"[FAIL] {model.role}: {exc}")
            failures += 1
            continue

        error = verify_model_file(target, model)
        if error is None:
            print(f"[PASS] {model.role}: verified {target.name}")
            continue
        if arguments.verify_only:
            print(f"[FAIL] {model.role}: {error}")
            failures += 1
            continue
        if target.exists() and not arguments.force:
            print(f"[FAIL] {model.role}: {error}; use --force to replace it")
            failures += 1
            continue

        try:
            temporary_path = _download_to_temporary_file(
                model,
                target,
                timeout_seconds=arguments.timeout,
            )
            os.replace(temporary_path, target)
        except (DownloadError, OSError) as exc:
            print(f"[FAIL] {model.role}: {exc}")
            failures += 1
        else:
            print(f"[PASS] {model.role}: installed and verified {target.name}")

    print(f"Summary: {len(selected) - failures} verified, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
