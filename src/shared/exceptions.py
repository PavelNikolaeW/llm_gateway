"""Custom exceptions for the application."""


class ApplicationError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ValidationError(ApplicationError):
    """Validation error (400 Bad Request)."""

    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class NotFoundError(ApplicationError):
    """Resource not found (404 Not Found)."""

    def __init__(self, message: str):
        super().__init__(message, status_code=404)


class ForbiddenError(ApplicationError):
    """Access forbidden (403 Forbidden)."""

    def __init__(self, message: str):
        super().__init__(message, status_code=403)


class UnauthorizedError(ApplicationError):
    """Authentication required (401 Unauthorized)."""

    def __init__(self, message: str):
        super().__init__(message, status_code=401)


class InsufficientTokensError(ApplicationError):
    """Insufficient tokens for operation (402 Payment Required)."""

    def __init__(self, message: str):
        super().__init__(message, status_code=402)
