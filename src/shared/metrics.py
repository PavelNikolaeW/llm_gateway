"""Prometheus metrics for monitoring.

Provides:
- HTTP request metrics (latency, count, errors)
- Token usage metrics
- LLM request metrics
- Business metrics (dialogs, messages)
"""
from prometheus_client import Counter, Histogram, Gauge, Info

# Application info
APP_INFO = Info("llm_gateway", "LLM Gateway application info")
APP_INFO.info({"version": "1.0.0", "name": "llm_gateway"})

# HTTP Request Metrics
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "HTTP requests currently in progress",
    ["method", "path"],
)

# Token Metrics
TOKENS_USED_TOTAL = Counter(
    "tokens_used_total",
    "Total tokens used",
    ["user_id", "model"],
)

TOKENS_PROMPT_TOTAL = Counter(
    "tokens_prompt_total",
    "Total prompt tokens",
    ["model"],
)

TOKENS_COMPLETION_TOTAL = Counter(
    "tokens_completion_total",
    "Total completion tokens",
    ["model"],
)

TOKEN_BALANCE = Gauge(
    "token_balance",
    "Current token balance",
    ["user_id"],
)

# LLM Request Metrics
LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM API requests",
    ["provider", "model", "status"],
)

LLM_REQUEST_DURATION_SECONDS = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration in seconds",
    ["provider", "model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

LLM_STREAMING_CHUNKS_TOTAL = Counter(
    "llm_streaming_chunks_total",
    "Total streaming chunks received",
    ["provider", "model"],
)

# Business Metrics
DIALOGS_CREATED_TOTAL = Counter(
    "dialogs_created_total",
    "Total dialogs created",
)

MESSAGES_SENT_TOTAL = Counter(
    "messages_sent_total",
    "Total messages sent",
    ["role"],  # 'user' or 'assistant'
)

ACTIVE_USERS = Gauge(
    "active_users",
    "Number of active users in the last 24 hours",
)

# Error Metrics
ERRORS_TOTAL = Counter(
    "errors_total",
    "Total errors",
    ["error_type", "path"],
)


def record_http_request(method: str, path: str, status_code: int, duration: float) -> None:
    """Record HTTP request metrics.

    Args:
        method: HTTP method (GET, POST, etc.)
        path: Request path (normalized)
        status_code: HTTP status code
        duration: Request duration in seconds
    """
    # Normalize path to avoid high cardinality
    normalized_path = _normalize_path(path)

    HTTP_REQUESTS_TOTAL.labels(
        method=method,
        path=normalized_path,
        status_code=str(status_code),
    ).inc()

    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=method,
        path=normalized_path,
    ).observe(duration)


def record_llm_request(
    provider: str,
    model: str,
    status: str,
    duration: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """Record LLM request metrics.

    Args:
        provider: LLM provider (openai, anthropic)
        model: Model name
        status: Request status (success, error, timeout)
        duration: Request duration in seconds
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
    """
    LLM_REQUESTS_TOTAL.labels(
        provider=provider,
        model=model,
        status=status,
    ).inc()

    LLM_REQUEST_DURATION_SECONDS.labels(
        provider=provider,
        model=model,
    ).observe(duration)

    if prompt_tokens > 0:
        TOKENS_PROMPT_TOTAL.labels(model=model).inc(prompt_tokens)

    if completion_tokens > 0:
        TOKENS_COMPLETION_TOTAL.labels(model=model).inc(completion_tokens)


def record_token_usage(user_id: int, model: str, tokens: int) -> None:
    """Record token usage.

    Args:
        user_id: User ID
        model: Model name
        tokens: Number of tokens used
    """
    TOKENS_USED_TOTAL.labels(user_id=str(user_id), model=model).inc(tokens)


def record_token_balance(user_id: int, balance: int) -> None:
    """Record current token balance.

    Args:
        user_id: User ID
        balance: Current balance
    """
    TOKEN_BALANCE.labels(user_id=str(user_id)).set(balance)


def record_dialog_created() -> None:
    """Record dialog creation."""
    DIALOGS_CREATED_TOTAL.inc()


def record_message_sent(role: str) -> None:
    """Record message sent.

    Args:
        role: Message role ('user' or 'assistant')
    """
    MESSAGES_SENT_TOTAL.labels(role=role).inc()


def record_error(error_type: str, path: str) -> None:
    """Record error occurrence.

    Args:
        error_type: Error type/code
        path: Request path
    """
    normalized_path = _normalize_path(path)
    ERRORS_TOTAL.labels(error_type=error_type, path=normalized_path).inc()


def _normalize_path(path: str) -> str:
    """Normalize path to reduce cardinality.

    Replaces UUIDs and numeric IDs with placeholders.

    Args:
        path: Original path

    Returns:
        Normalized path
    """
    import re

    # Replace UUIDs
    path = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "{id}",
        path,
        flags=re.IGNORECASE,
    )

    # Replace numeric IDs
    path = re.sub(r"/\d+(/|$)", r"/{id}\1", path)

    return path
