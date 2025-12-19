"""Custom exceptions for the application."""


class ApplicationError(Exception):
    """Base exception for all application errors."""

    code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, status_code: int = 500, details: dict | None = None):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)


class ValidationError(ApplicationError):
    """Validation error (400 Bad Request)."""

    code = "VALIDATION_ERROR"

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=400, details=details)


class NotFoundError(ApplicationError):
    """Resource not found (404 Not Found)."""

    code = "NOT_FOUND"

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=404, details=details)


class ForbiddenError(ApplicationError):
    """Access forbidden (403 Forbidden)."""

    code = "FORBIDDEN"

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=403, details=details)


class UnauthorizedError(ApplicationError):
    """Authentication required (401 Unauthorized)."""

    code = "UNAUTHORIZED"

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=401, details=details)


class InsufficientTokensError(ApplicationError):
    """Insufficient tokens for operation (402 Payment Required)."""

    code = "INSUFFICIENT_TOKENS"

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=402, details=details)


class LLMTimeoutError(ApplicationError):
    """LLM request timed out (504 Gateway Timeout)."""

    code = "LLM_TIMEOUT"

    def __init__(self, message: str = "LLM request timed out", details: dict | None = None):
        super().__init__(message, status_code=504, details=details)


class LLMError(ApplicationError):
    """LLM error (500 Internal Server Error)."""

    code = "LLM_ERROR"

    def __init__(self, message: str = "LLM error", details: dict | None = None):
        super().__init__(message, status_code=500, details=details)
