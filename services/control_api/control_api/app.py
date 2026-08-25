"""FastAPI application factory for the self-contained campus control plane."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from control_api.agentic import AgentWorkflowService
from control_api.config import Settings
from control_api.database import Database
from control_api.errors import ControlApiError
from control_api.repository import Repository
from control_api.routers import access, agent, operations, system, topology
from control_api.schemas import ErrorDetail

LOGGER = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construct an isolated API instance; initialize persistence during lifespan."""

    resolved = settings or Settings.from_environment()
    database = Database(resolved.database_path)
    repository = Repository(database)
    agent_service = AgentWorkflowService(repository)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        database.initialize(seed=resolved.seed_demo_data)
        yield

    application = FastAPI(
        title="Campus Access Control API",
        version="0.1.0",
        summary="Multi-organization campus access control-plane demo",
        description=(
            "A typed, SQLite-backed demonstration API. Recognition observations report what "
            "the ANPR worker saw; authorization decisions independently record whether access "
            "is allowed, denied, or requires review. Seeded people, vehicles, coordinates, and "
            "evidence labels are fictional. Use `/api/v1/demo-identities` for demo bearer tokens."
        ),
        lifespan=lifespan,
        openapi_tags=[
            {"name": "system", "description": "Health, metadata, and demo sessions."},
            {"name": "campus topology", "description": "Organizations, sites, gates, cameras."},
            {"name": "access", "description": "Requests, approvals, and grants."},
            {
                "name": "operations",
                "description": "Passages, recognition, authorization, events, and incidents.",
            },
            {
                "name": "agent operations",
                "description": "Bounded plans, tools, policy checks, approvals, and audit traces.",
            },
            {"name": "dashboard", "description": "Command-center aggregate read model."},
        ],
    )
    application.state.repository = repository
    application.state.agent_service = agent_service
    application.state.settings = resolved

    if resolved.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Organization-ID"],
        )

    @application.exception_handler(ControlApiError)
    async def handle_control_error(request: Request, error: ControlApiError) -> JSONResponse:
        detail = ErrorDetail(
            type=f"urn:campus-control:{error.code}",
            title=error.title,
            status=error.status_code,
            detail=str(error),
            instance=request.url.path,
        )
        return JSONResponse(
            status_code=error.status_code,
            content=detail.model_dump(mode="json"),
            media_type="application/problem+json",
        )

    @application.exception_handler(sqlite3.Error)
    async def handle_database_error(request: Request, error: sqlite3.Error) -> JSONResponse:
        LOGGER.exception("SQLite operation failed", exc_info=error)
        detail = ErrorDetail(
            type="urn:campus-control:database-error",
            title="Persistence unavailable",
            status=500,
            detail="The persistence layer could not complete the request.",
            instance=request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content=detail.model_dump(mode="json"),
            media_type="application/problem+json",
        )

    # Specific routes are registered before the catch-all static application.
    application.include_router(system.router)
    application.include_router(topology.router)
    application.include_router(access.router)
    application.include_router(operations.router)
    application.include_router(agent.router)

    console_dir = resolved.console_dir
    if console_dir is not None and (console_dir / "index.html").is_file():
        application.mount(
            "/",
            StaticFiles(directory=console_dir, html=True),
            name="campus-console",
        )
    else:

        @application.get("/", include_in_schema=False)
        def api_landing() -> dict[str, str]:
            return {
                "service": "Campus Access Control API",
                "docs": "/docs",
                "health": "/health/ready",
            }

    return application


app = create_app()
