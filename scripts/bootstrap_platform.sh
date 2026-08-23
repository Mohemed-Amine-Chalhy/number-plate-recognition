#!/usr/bin/env bash
set -Eeuo pipefail

script_directory="${BASH_SOURCE[0]%/*}"
if [[ "$script_directory" == "${BASH_SOURCE[0]}" ]]; then
    script_directory="."
fi
repository_root="$(cd -- "$script_directory/.." && pwd -P)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repository_root/.uv-cache}"
python_path=""
install_hooks=true
run_checks=true

usage() {
    echo "Usage: $0 [--python PATH] [--no-hooks] [--skip-checks]"
}

while (($#)); do
    case "$1" in
        --python)
            shift
            if (($# == 0)); then
                echo "--python requires a path" >&2
                exit 2
            fi
            python_path="$1"
            ;;
        --no-hooks)
            install_hooks=false
            ;;
        --skip-checks)
            run_checks=false
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

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it from https://docs.astral.sh/uv/." >&2
    exit 1
fi

cd -- "$repository_root"
if [[ -z "$python_path" ]]; then
    if ! python_path="$(uv python find 3.12 2>/dev/null)"; then
        echo "Python 3.12 was not found; installing a managed interpreter with uv..."
        uv python install 3.12
        python_path="$(uv python find 3.12)"
    fi
fi

echo "Preparing the computer-vision environment with Python 3.12..."
uv sync --python "$python_path" --locked
echo "Preparing the lightweight control-plane environment..."
uv sync --project services/control_api --python "$python_path" --locked

if [[ "$install_hooks" == true && -f .pre-commit-config.yaml ]]; then
    echo "Installing pre-commit and pre-push hooks..."
    uv run --frozen pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push
fi

if [[ "$run_checks" == true ]]; then
    echo "Running platform diagnostics..."
    uv run --frozen python scripts/platform_doctor.py
    uv run --frozen python scripts/doctor.py --manifest-only
fi
echo "Campus platform bootstrap completed successfully."
