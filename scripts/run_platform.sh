#!/usr/bin/env bash
set -Eeuo pipefail

script_directory="${BASH_SOURCE[0]%/*}"
if [[ "$script_directory" == "${BASH_SOURCE[0]}" ]]; then
    script_directory="."
fi
repository_root="$(cd -- "$script_directory/.." && pwd -P)"
service_project="$repository_root/services/control_api"
environment_file="$repository_root/.env"
bind_address=""
listen_port=""
sync_environment=true
load_environment=true

usage() {
    printf '%s\n' \
        "Usage: scripts/run_platform.sh [--host ADDRESS] [--port PORT] [--no-sync] [--no-env-file]" \
        "" \
        "Starts the campus control API and same-origin console. The safe local bind is" \
        "127.0.0.1:8000 unless overridden by .env, environment variables, or flags."
}

while (($#)); do
    case "$1" in
        --host)
            shift
            if (($# == 0)); then
                echo "--host requires an address" >&2
                exit 2
            fi
            bind_address="$1"
            ;;
        --port)
            shift
            if (($# == 0)) || [[ ! "$1" =~ ^[0-9]+$ ]] || ((10#$1 < 1 || 10#$1 > 65535)); then
                echo "--port requires an integer between 1 and 65535" >&2
                exit 2
            fi
            listen_port="$1"
            ;;
        --no-sync)
            sync_environment=false
            ;;
        --no-env-file)
            load_environment=false
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
    echo "uv is required. Run scripts/bootstrap_platform.sh after installing uv." >&2
    exit 1
fi
if [[ ! -f "$service_project/uv.lock" ]]; then
    echo "Control API lockfile not found: $service_project" >&2
    exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-$repository_root/.uv-cache}"
if [[ -n "$bind_address" ]]; then
    export CONTROL_API_HOST="$bind_address"
fi
if [[ -n "$listen_port" ]]; then
    export CONTROL_API_PORT="$listen_port"
fi
mkdir -p -- "$repository_root/.runtime/platform"

cd -- "$repository_root"
if [[ "$sync_environment" == true ]]; then
    echo "Synchronizing the locked control API environment..."
    uv sync --project "$service_project" --locked
fi

uv_arguments=(run --project "$service_project" --frozen)
if [[ "$load_environment" == true && -f "$environment_file" ]]; then
    uv_arguments+=(--env-file "$environment_file")
fi

display_address="${CONTROL_API_HOST:-127.0.0.1}"
display_port="${CONTROL_API_PORT:-8000}"
echo "Starting Campus Access at http://${display_address}:${display_port}/"
exec uv "${uv_arguments[@]}" python -m control_api
