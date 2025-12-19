"""FastAPI Application Setup - Core App Configuration.

Configures the FastAPI application with:
- JWT authentication middleware
- CORS configuration
- Global exception handlers
- Structured logging
- Health check endpoint
"""
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.api.routes import dialogs_router, messages_router
from src.config.settings import settings
from src.integrations.jwt_validator import JWTValidator, JWTClaims
from src.shared.exceptions import (
    ApplicationError,
    ForbiddenError,
    InsufficientTokensError,
    LLMError,
    LLMTimeoutError,
    NotFoundError,
    UnauthorizedError,
    ValidationError,
)

logger = logging.getLogger(__name__)

# Context variables for request context
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")
user_id_ctx: ContextVar[int | None] = ContextVar("user_id", default=None)
is_admin_ctx: ContextVar[bool] = ContextVar("is_admin", default=False)


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="LLM Gateway API",
        description="API gateway for LLM providers with token management",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Configure middleware
    _configure_cors(app)
    _configure_middleware(app)

    # Configure exception handlers
    _configure_exception_handlers(app)

    # Register routes
    _register_routes(app)

    return app


def _configure_cors(app: FastAPI) -> None:
    """Configure CORS middleware.

    Allows origins from CORS_ORIGINS environment variable.
    """
    # Parse origins from comma-separated string
    origins = [origin.strip() for origin in settings.cors_origins.split(",")]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.info(f"CORS configured for origins: {origins}")


def _configure_middleware(app: FastAPI) -> None:
    """Configure custom middleware."""
    # Add request context middleware (adds request_id, logging)
    app.add_middleware(RequestContextMiddleware)

    # Add JWT auth middleware
    app.add_middleware(JWTAuthMiddleware)


def _configure_exception_handlers(app: FastAPI) -> None:
    """Configure global exception handlers.

    Maps domain exceptions to HTTP status codes:
    - ValidationError -> 400
    - UnauthorizedError -> 401
    - InsufficientTokensError -> 402
    - ForbiddenError -> 403
    - NotFoundError -> 404
    - LLMTimeoutError -> 504
    - LLMError -> 500
    - ApplicationError -> status_code from exception
    - Exception -> 500
    """

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
        return _create_error_response(exc, 400)

    @app.exception_handler(UnauthorizedError)
    async def unauthorized_error_handler(request: Request, exc: UnauthorizedError) -> JSONResponse:
        return _create_error_response(exc, 401)

    @app.exception_handler(InsufficientTokensError)
    async def insufficient_tokens_error_handler(
        request: Request, exc: InsufficientTokensError
    ) -> JSONResponse:
        return _create_error_response(exc, 402)

    @app.exception_handler(ForbiddenError)
    async def forbidden_error_handler(request: Request, exc: ForbiddenError) -> JSONResponse:
        return _create_error_response(exc, 403)

    @app.exception_handler(NotFoundError)
    async def not_found_error_handler(request: Request, exc: NotFoundError) -> JSONResponse:
        return _create_error_response(exc, 404)

    @app.exception_handler(LLMTimeoutError)
    async def llm_timeout_error_handler(request: Request, exc: LLMTimeoutError) -> JSONResponse:
        return _create_error_response(exc, 504)

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
        return _create_error_response(exc, 500)

    @app.exception_handler(ApplicationError)
    async def application_error_handler(
        request: Request, exc: ApplicationError
    ) -> JSONResponse:
        return _create_error_response(exc, exc.status_code)

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"Unhandled error: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": request_id_ctx.get(),
            },
        )


def _create_error_response(exc: ApplicationError, status_code: int) -> JSONResponse:
    """Create standardized error response.

    Args:
        exc: Application exception
        status_code: HTTP status code

    Returns:
        JSON response with error details
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.message,
            "request_id": request_id_ctx.get(),
        },
    )


def _register_routes(app: FastAPI) -> None:
    """Register API routes."""

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint.

        Returns:
            Health status
        """
        return {"status": "ok"}

    # Register API routers
    app.include_router(dialogs_router, prefix="/api/v1")
    app.include_router(messages_router, prefix="/api/v1")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to add request context (request_id, logging).

    Adds:
    - Unique request_id to each request
    - Structured logging with request details
    - Request timing
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate request ID
        req_id = str(uuid.uuid4())
        request_id_ctx.set(req_id)

        # Add request ID to response headers
        start_time = time.time()

        # Log request
        logger.info(
            f"Request started",
            extra={
                "request_id": req_id,
                "method": request.method,
                "path": request.url.path,
                "user_id": user_id_ctx.get(),
            },
        )

        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Add headers
            response.headers["X-Request-ID"] = req_id

            # Log response
            logger.info(
                f"Request completed",
                extra={
                    "request_id": req_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "user_id": user_id_ctx.get(),
                },
            )

            return response

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"Request failed: {e}",
                extra={
                    "request_id": req_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                    "user_id": user_id_ctx.get(),
                },
            )
            raise


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for JWT authentication.

    Extracts token from Authorization header, validates it,
    and injects user_id and is_admin into request context.

    Excludes:
    - /health
    - /docs
    - /redoc
    - /openapi.json
    """

    # Paths that don't require authentication
    PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}
    # Path prefixes that don't require authentication
    PUBLIC_PREFIXES = ("/health", "/docs", "/redoc")

    def __init__(self, app: FastAPI):
        super().__init__(app)
        self._validator = JWTValidator()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip auth for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Skip auth for paths with public prefixes
        if request.url.path.startswith(self.PUBLIC_PREFIXES):
            return await call_next(request)

        # Get Authorization header
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Authorization header required",
                    "request_id": request_id_ctx.get(),
                },
            )

        try:
            # Validate token and extract claims
            claims = self._validator.validate(auth_header)

            # Set context variables
            user_id_ctx.set(claims.user_id)
            is_admin_ctx.set(claims.is_admin)

            # Store claims in request state for access in route handlers
            request.state.user_id = claims.user_id
            request.state.is_admin = claims.is_admin
            request.state.jwt_claims = claims

            return await call_next(request)

        except UnauthorizedError as e:
            return JSONResponse(
                status_code=401,
                content={
                    "error": e.message,
                    "request_id": request_id_ctx.get(),
                },
            )


# Convenience functions for accessing request context
def get_request_id() -> str:
    """Get current request ID."""
    return request_id_ctx.get()


def get_current_user_id() -> int | None:
    """Get current user ID from request context."""
    return user_id_ctx.get()


def get_is_admin() -> bool:
    """Get admin status from request context."""
    return is_admin_ctx.get()


# Create the app instance
app = create_app()
