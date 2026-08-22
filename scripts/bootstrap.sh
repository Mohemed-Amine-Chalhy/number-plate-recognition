#!/usr/bin/env bash
set -Eeuo pipefail

script_directory="${BASH_SOURCE[0]%/*}"
if [[ "$script_directory" == "${BASH_SOURCE[0]}" ]]; then
    script_directory="."
fi
repository_root="$(cd -- "$script_directory/.." && pwd -P)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repository_root/.uv-cache}"
runtime_only=false
all_groups=false
install_hooks=true

usage() {
    echo "Usage: $0 [--runtime-only | --all-groups] [--no-hooks]"
    echo "  --runtime-only  Install only production dependencies."
    echo "  --all-groups    Also install optional notebook dependencies."
    echo "  --no-hooks      Do not install Git pre-commit/pre-push hooks."
}

while (($#)); do
    case "$1" in
        --runtime-only)
            runtime_only=true
            ;;
        --all-groups)
            all_groups=true
            ;;
        --no-hooks)
            install_hooks=false
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ "$runtime_only" == true && "$all_groups" == true ]]; then
    echo "--runtime-only and --all-groups cannot be used together." >&2
    exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it from https://docs.astral.sh/uv/ and run this script again." >&2
    exit 1
fi

cd -- "$repository_root"

echo "Checking for Python 3.12..."
if ! uv python find 3.12 >/dev/null 2>&1; then
    echo "Python 3.12 was not found; installing a managed interpreter with uv..."
    uv python install 3.12
fi

echo "Creating or updating the locked environment..."
if [[ "$runtime_only" == true ]]; then
    uv sync --locked --no-dev
elif [[ "$all_groups" == true ]]; then
    uv sync --locked --all-groups
else
    uv sync --locked
fi

if [[ "$runtime_only" == false && "$install_hooks" == true && -f .pre-commit-config.yaml ]]; then
    echo "Installing pre-commit and pre-push hooks..."
    uv run --frozen pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push
fi

echo "Running environment diagnostics..."
uv run --frozen python scripts/doctor.py
echo "Bootstrap completed successfully."
