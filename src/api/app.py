"""FastAPI Application Setup - Core App Configuration.

Configures the FastAPI application with:
- JWT authentication middleware
- CORS configuration
- Global exception handlers
- Structured logging
- Health check endpoint
- Model registry initialization
"""
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.api.rate_limiter import rate_limiter
from src.api.routes import admin_router, audit_router, dialogs_router, export_router, messages_router, models_router, tokens_router
from src.config.logging import configure_logging, get_logger
from src.config.settings import settings
from src.data.database import get_session_maker
from src.domain.model_registry import model_registry
from src.integrations.jwt_validator import JWTValidator
from src.shared.metrics import record_http_request
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

logger = get_logger(__name__)

# Context variables for request context
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")
user_id_ctx: ContextVar[int | None] = ContextVar("user_id", default=None)
is_admin_ctx: ContextVar[bool] = ContextVar("is_admin", default=False)


# OpenAPI Tags metadata
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager.

    Handles startup and shutdown:
    - Startup: Load model registry from database
    - Shutdown: Cleanup resources
    """
    # Startup
    logger.info("Loading model registry from database...")
    session_maker = get_session_maker()
    async with session_maker() as session:
        await model_registry.load_models(session)
    logger.info(f"Model registry loaded: {len(model_registry.get_all_models())} models")

    yield

    # Shutdown
    logger.info("Application shutting down")


OPENAPI_TAGS = [
    {
        "name": "dialogs",
        "description": "Dialog management - create, list, and retrieve conversation dialogs.",
    },
    {
        "name": "messages",
        "description": "Message operations - send messages and stream LLM responses via SSE.",
    },
    {
        "name": "tokens",
        "description": "Token balance and usage - check remaining tokens and usage stats.",
    },
    {
        "name": "export",
        "description": "Export/Import - backup and restore dialogs in JSON format.",
    },
    {
        "name": "admin",
        "description": "Admin operations - user management, token allocation, and statistics.",
    },
]


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application
    """
    # Configure logging first
    configure_logging(
        level=settings.log_level,
        json_format=not settings.debug,  # JSON in prod, console in dev
        use_colors=True,
    )

    app = FastAPI(
        title="LLM Gateway API",
        lifespan=lifespan,
        description="""
## Overview

LLM Gateway is an API gateway for Large Language Model providers with built-in token management.

### Features

- **Multi-provider support**: OpenAI, Anthropic (Claude)
- **Token management**: Balance tracking, limits, transactions
- **Streaming**: Server-Sent Events (SSE) for real-time responses
- **Dialog management**: Persistent conversation history
- **Admin controls**: User management, token allocation, usage stats

### Authentication

All endpoints (except /health, /metrics, /docs) require JWT authentication.

Include the token in the Authorization header:
```
Authorization: Bearer <your-jwt-token>
```

### Error Responses

All errors follow the format:
```json
{
  "code": "ERROR_CODE",
  "message": "Human readable message",
  "request_id": "uuid",
  "details": {}
}
```
""",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=OPENAPI_TAGS,
        contact={
            "name": "LLM Gateway Team",
        },
        license_info={
            "name": "MIT",
        },
        responses={
            401: {"description": "Unauthorized - Invalid or missing JWT token"},
            403: {"description": "Forbidden - Insufficient permissions"},
            500: {"description": "Internal Server Error"},
        },
    )

    # Configure middleware
    _configure_cors(app)
    _configure_middleware(app)

    # Configure exception handlers
    _configure_exception_handlers(app)

    # Register routes
    _register_routes(app)

    logger.info("Application started", extra={"debug_mode": settings.debug})

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
    # Add request context middleware (adds request_id, logging, metrics)
    app.add_middleware(RequestContextMiddleware)

    # Add rate limiting middleware
    app.add_middleware(RateLimitMiddleware)

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
        return _create_error_response(exc, 400, request)

    @app.exception_handler(UnauthorizedError)
    async def unauthorized_error_handler(request: Request, exc: UnauthorizedError) -> JSONResponse:
        return _create_error_response(exc, 401, request)

    @app.exception_handler(InsufficientTokensError)
    async def insufficient_tokens_error_handler(
        request: Request, exc: InsufficientTokensError
    ) -> JSONResponse:
        return _create_error_response(exc, 402, request)

    @app.exception_handler(ForbiddenError)
    async def forbidden_error_handler(request: Request, exc: ForbiddenError) -> JSONResponse:
        return _create_error_response(exc, 403, request)

    @app.exception_handler(NotFoundError)
    async def not_found_error_handler(request: Request, exc: NotFoundError) -> JSONResponse:
        return _create_error_response(exc, 404, request)

    @app.exception_handler(LLMTimeoutError)
    async def llm_timeout_error_handler(request: Request, exc: LLMTimeoutError) -> JSONResponse:
        return _create_error_response(exc, 504, request)

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
        return _create_error_response(exc, 500, request)

    @app.exception_handler(ApplicationError)
    async def application_error_handler(
        request: Request, exc: ApplicationError
    ) -> JSONResponse:
        return _create_error_response(exc, exc.status_code, request)

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = request_id_ctx.get()

        # Log with full context and stack trace
        logger.exception(
            f"Unhandled error: {exc}",
            extra={
                "error_code": "INTERNAL_ERROR",
                "status_code": 500,
                "request_id": request_id,
                "user_id": user_id_ctx.get(),
                "path": request.url.path,
                "method": request.method,
            },
        )

        # Build response - include stack trace only in debug mode
        content: dict = {
            "code": "INTERNAL_ERROR",
            "message": "Internal server error",
            "request_id": request_id,
        }

        if settings.debug:
            import traceback
            content["details"] = {
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }

        return JSONResponse(status_code=500, content=content)


def _create_error_response(
    exc: ApplicationError,
    status_code: int,
    request: Request | None = None,
) -> JSONResponse:
    """Create standardized error response.

    Args:
        exc: Application exception
        status_code: HTTP status code
        request: Optional request for logging context

    Returns:
        JSON response with error details in format:
        {code: string, message: string, details?: object, request_id: string}
    """
    request_id = request_id_ctx.get()

    # Log error with structured context
    log_context = {
        "error_code": exc.code,
        "status_code": status_code,
        "request_id": request_id,
        "user_id": user_id_ctx.get(),
    }
    if request:
        log_context["path"] = request.url.path
        log_context["method"] = request.method

    # Log at appropriate level
    if status_code >= 500:
        logger.error(f"Server error: {exc.message}", extra=log_context)
    else:
        logger.warning(f"Client error: {exc.message}", extra=log_context)

    # Build response content
    content: dict = {
        "code": exc.code,
        "message": exc.message,
        "request_id": request_id,
    }

    # Include details if present (but never in production for 500 errors)
    if exc.details and (status_code < 500 or settings.debug):
        content["details"] = exc.details

    return JSONResponse(status_code=status_code, content=content)


def _register_routes(app: FastAPI) -> None:
    """Register API routes."""
    from src.api.health import check_system_health, is_ready, is_alive

    @app.get("/health")
    async def health_check() -> dict:
        """Comprehensive health check endpoint.

        Returns:
            Detailed health status of all components
        """
        health = await check_system_health()
        return health.to_dict()

    @app.get("/health/ready")
    async def readiness_check() -> dict[str, str]:
        """Kubernetes readiness probe.

        Returns 200 if ready to receive traffic, 503 otherwise.
        """
        if await is_ready():
            return {"status": "ready"}
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready"},
        )

    @app.get("/health/live")
    async def liveness_check() -> dict[str, str]:
        """Kubernetes liveness probe.

        Returns 200 if the process is alive.
        """
        if await is_alive():
            return {"status": "alive"}
        return JSONResponse(
            status_code=503,
            content={"status": "dead"},
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        """Prometheus metrics endpoint.

        Returns:
            Prometheus metrics in text format
        """
        return PlainTextResponse(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    # Register API routers
    app.include_router(admin_router, prefix="/api/v1")
    app.include_router(audit_router, prefix="/api/v1")
    app.include_router(dialogs_router, prefix="/api/v1")
    app.include_router(export_router, prefix="/api/v1")
    app.include_router(messages_router, prefix="/api/v1")
    app.include_router(models_router, prefix="/api/v1")
    app.include_router(tokens_router, prefix="/api/v1")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to add request context (request_id, logging, metrics).

    Adds:
    - Unique request_id to each request
    - Structured logging with request details
    - Request timing
    - Prometheus metrics
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
            "Request started",
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
            duration_seconds = time.time() - start_time
            duration_ms = int(duration_seconds * 1000)

            # Add headers
            response.headers["X-Request-ID"] = req_id

            # Record metrics
            record_http_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=duration_seconds,
            )

            # Log response
            logger.info(
                "Request completed",
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
            duration_seconds = time.time() - start_time
            duration_ms = int(duration_seconds * 1000)

            # Record error metrics
            record_http_request(
                method=request.method,
                path=request.url.path,
                status_code=500,
                duration=duration_seconds,
            )

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


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests.

    Enforces per-user rate limits using Redis sliding window.
    Falls back to allowing requests if Redis unavailable.

    Excludes:
    - /health
    - /metrics
    - /docs
    - /redoc
    - /openapi.json
    """

    # Paths excluded from rate limiting
    EXCLUDED_PATHS = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}
    EXCLUDED_PREFIXES = ("/health", "/metrics", "/docs", "/redoc")

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip rate limiting for excluded paths
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        if request.url.path.startswith(self.EXCLUDED_PREFIXES):
            return await call_next(request)

        # Get identifier for rate limiting (user_id or IP)
        identifier = self._get_identifier(request)

        # Check rate limit
        result = await rate_limiter.check(identifier)

        if not result.allowed:
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "request_id": request_id_ctx.get(),
                    "identifier": identifier,
                    "path": request.url.path,
                },
            )
            return JSONResponse(
                status_code=429,
                content={
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests. Please slow down.",
                    "request_id": request_id_ctx.get(),
                    "details": {
                        "limit": result.limit,
                        "window_seconds": result.window,
                        "retry_after": result.reset_at,
                    },
                },
                headers={
                    "Retry-After": str(result.window),
                    "X-RateLimit-Limit": str(result.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(result.reset_at),
                },
            )

        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(result.reset_at)

        return response

    def _get_identifier(self, request: Request) -> str:
        """Get identifier for rate limiting.

        Uses user_id if authenticated, otherwise IP address.
        """
        # Try to get user_id from context (set by JWT middleware)
        user_id = user_id_ctx.get()
        if user_id is not None:
            return f"user:{user_id}"

        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        # Check for X-Forwarded-For header
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        return f"ip:{client_ip}"


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for JWT authentication.

    Extracts token from Authorization header, validates it,
    and injects user_id and is_admin into request context.

    Excludes:
    - /health
    - /metrics
    - /docs
    - /redoc
    - /openapi.json
    """

    # Paths that don't require authentication
    PUBLIC_PATHS = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}
    # Path prefixes that don't require authentication
    PUBLIC_PREFIXES = ("/health", "/metrics", "/docs", "/redoc")

    def __init__(self, app: FastAPI):
        super().__init__(app)
        self._validator = JWTValidator()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip auth for CORS preflight requests
        if request.method == "OPTIONS":
            return await call_next(request)

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
                    "code": "UNAUTHORIZED",
                    "message": "Authorization header required",
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
                    "code": e.code,
                    "message": e.message,
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
