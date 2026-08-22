# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.8.15 AS uv

FROM python:3.12-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    UV_PYTHON_DOWNLOADS=never

COPY --from=uv /uv /uvx /bin/

WORKDIR /opt/app

# Install dependencies before application sources to preserve the expensive
# ML dependency layer when only project code changes.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim-bookworm AS runtime

ENV HOME=/home/app \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MPLCONFIGDIR=/tmp/matplotlib \
    NPR_APP_ROOT=/opt/app \
    NPR_IMAGE_DIR=/opt/app/images \
    NPR_MODEL_DIR=/opt/app/models \
    NPR_MODEL_MANIFEST=/opt/app/models/manifest.json \
    PATH=/opt/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_SERVER_HEADLESS=true \
    XDG_CACHE_HOME=/tmp/.cache \
    YOLO_AUTOINSTALL=false \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_OFFLINE=true

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /opt/app

COPY --from=builder --chown=app:app /opt/app/.venv ./.venv
COPY --chown=app:app app ./app
COPY --chown=app:app src ./src
COPY --chown=app:app models ./models
COPY --chown=app:app images ./images
COPY --chown=app:app .streamlit ./.streamlit
COPY --chown=app:app scripts/doctor.py ./scripts/doctor.py
COPY --chown=app:app docker/entrypoint.sh ./docker/entrypoint.sh

RUN chmod 0555 /opt/app/docker/entrypoint.sh

USER 10001:10001

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2).read()"]

ENTRYPOINT ["/opt/app/docker/entrypoint.sh"]
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
