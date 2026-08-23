"""Domain-facing errors translated to stable API problem responses."""

from __future__ import annotations


class ControlApiError(Exception):
    """Base class for expected control-plane failures."""

    status_code = 400
    code = "control_api_error"
    title = "Control API error"


class ResourceNotFoundError(ControlApiError):
    status_code = 404
    code = "resource_not_found"
    title = "Resource not found"


class ConflictError(ControlApiError):
    status_code = 409
    code = "resource_conflict"
    title = "Resource conflict"


class InvalidStateError(ControlApiError):
    status_code = 422
    code = "invalid_state"
    title = "Invalid state transition"


class ForbiddenError(ControlApiError):
    status_code = 403
    code = "forbidden"
    title = "Forbidden"
