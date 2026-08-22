#!/usr/bin/env bash
set -Eeuo pipefail

script_directory="${BASH_SOURCE[0]%/*}"
if [[ "$script_directory" == "${BASH_SOURCE[0]}" ]]; then
    script_directory="."
fi
repository_root="$(cd -- "$script_directory/.." && pwd -P)"
application="$repository_root/app/streamlit_app.py"
environment_file="$repository_root/.env"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repository_root/.uv-cache}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$repository_root/.runtime/ultralytics}"
export YOLO_AUTOINSTALL="false"
export YOLO_OFFLINE="true"
mkdir -p -- "$YOLO_CONFIG_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Run scripts/bootstrap.sh after installing uv." >&2
    exit 1
fi
if [[ ! -f "$application" ]]; then
    echo "Streamlit entry point not found: $application" >&2
    exit 1
fi

cd -- "$repository_root"
uv_arguments=(run --frozen)
if [[ -f "$environment_file" ]]; then
    uv_arguments+=(--env-file "$environment_file")
fi
exec uv "${uv_arguments[@]}" streamlit run "$application" --server.headless true "$@"
