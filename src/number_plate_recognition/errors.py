"""Application-specific exceptions with safe, user-facing messages."""


class NumberPlateRecognitionError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(NumberPlateRecognitionError):
    """Raised when environment or application configuration is invalid."""


class ImageValidationError(NumberPlateRecognitionError):
    """Raised when an uploaded image cannot be accepted safely."""


class ModelIntegrityError(NumberPlateRecognitionError):
    """Raised when a model artifact is missing or fails integrity checks."""


class InferenceError(NumberPlateRecognitionError):
    """Raised when a detector cannot complete an inference request."""
