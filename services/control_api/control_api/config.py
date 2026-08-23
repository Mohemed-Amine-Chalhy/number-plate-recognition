"""Environment-backed service settings with repository-safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise ValueError("CONTROL_API_PORT must be an integer") from error
    if not 1 <= port <= 65_535:
        raise ValueError("CONTROL_API_PORT must be between 1 and 65535")
    return port


@dataclass(frozen=True, slots=True)
class Settings:
    """All filesystem/environment choices needed by the API process."""

    database_path: Path
    seed_demo_data: bool
    console_dir: Path | None
    cors_origins: tuple[str, ...]
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_environment(cls) -> Settings:
        """Resolve defaults relative to the repository, not the caller's working directory."""

        repository_root = Path(__file__).resolve().parents[3]
        database_path = Path(
            os.getenv(
                "CONTROL_API_DB_PATH",
                str(repository_root / ".runtime" / "campus-control.sqlite3"),
            )
        ).expanduser()
        configured_console = os.getenv("CAMPUS_CONSOLE_DIR")
        console_dir = (
            Path(configured_console).expanduser().resolve()
            if configured_console
            else repository_root / "web" / "console"
        )
        seed_demo_data = _parse_bool(
            os.getenv("CONTROL_API_SEED_DEMO", "true"),
            name="CONTROL_API_SEED_DEMO",
        )
        cors_origins = tuple(
            origin.strip()
            for origin in os.getenv(
                "CONTROL_API_CORS_ORIGINS",
                "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173",
            ).split(",")
            if origin.strip()
        )
        host = os.getenv("CONTROL_API_HOST", "127.0.0.1").strip()
        if not host:
            raise ValueError("CONTROL_API_HOST must not be empty")
        port = _parse_port(os.getenv("CONTROL_API_PORT", "8000"))
        return cls(
            database_path=database_path.resolve(),
            seed_demo_data=seed_demo_data,
            console_dir=console_dir.resolve(),
            cors_origins=cors_origins,
            host=host,
            port=port,
        )
