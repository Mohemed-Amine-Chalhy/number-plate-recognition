"""Run the control-plane API with Uvicorn."""

from __future__ import annotations

import uvicorn

from control_api.config import Settings


def main() -> None:
    """Start the development server."""

    settings = Settings.from_environment()
    uvicorn.run(
        "control_api.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
