# LLM Gateway

API service for managing LLM chat requests with dialog management, token usage control, and admin operations.

## Architecture

This project follows a clean architecture approach with the following layers:

- **API Layer**: FastAPI endpoints, request validation, JWT authentication
- **Domain Layer**: Business logic for dialogs, messages, tokens, models, agents
- **Data Layer**: Database access with SQLAlchemy async ORM
- **Integrations**: External services (OpenAI, Anthropic, JWT validation)
- **Shared**: Common utilities, DTOs, exceptions

## Tech Stack

- Python 3.11+
- FastAPI for REST API
- PostgreSQL 15+ for persistence
- Redis for caching (optional)
- SQLAlchemy 2.0 async ORM
- Alembic for migrations
- OpenAI & Anthropic SDKs

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- (Optional) Redis

### Installation

1. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -e ".[dev]"
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

4. Run database migrations:
```bash
alembic upgrade head
```

5. Start the server:
```bash
uvicorn src.api.main:app --reload
```

## Development

### Database Migrations

Create new migration:
```bash
alembic revision --autogenerate -m "description"
```

Apply migrations:
```bash
alembic upgrade head
```

Rollback:
```bash
alembic downgrade -1
```

### Available Models

The system is seeded with the following LLM models:

**OpenAI:**
- `gpt-4-turbo` - 128k context, $0.01/$0.03 per 1k tokens
- `gpt-3.5-turbo` - 16k context, $0.0005/$0.0015 per 1k tokens

**Anthropic:**
- `claude-3-opus` - 200k context, $0.015/$0.075 per 1k tokens
- `claude-3-sonnet` - 200k context, $0.003/$0.015 per 1k tokens

View all models:
```bash
psql $DATABASE_URL -c "SELECT name, provider, context_window, enabled FROM models;"
```

**Note:** Admin API for updating model metadata will be implemented in a future epic.

### Testing

Run all tests (from project root):
```bash
pytest tests/
```

Run individual test files (recommended for async tests):
```bash
pytest tests/integration/test_cache.py -v
pytest tests/integration/test_data_access.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

**Note:** Due to pytest-asyncio limitations, running all tests together may cause event loop conflicts. Running tests individually is recommended.

### Code Quality

Format code:
```bash
black src tests
```

Lint:
```bash
ruff check src tests
```

Type check:
```bash
mypy src
```

## Admin Panel

The service includes a web-based admin panel at `/admin` built with SQLAdmin.

### Features

- **Models Management**: CRUD operations for LLM models with hot reload
- **Token Balances**: View and edit user token balances and limits
- **Token Transactions**: Audit log of all token operations (read-only)
- **Dialogs & Messages**: View and manage conversations
- **Audit Logs**: System-wide audit trail (read-only)
- **System Config**: Store API keys and settings (secrets are masked)

### Access

1. Navigate to `http://localhost:8001/admin`
2. Login using your OmniMap credentials (username/password)
3. Only users with `is_staff=True` can access the admin panel

### Configuration

Set the authentication endpoint in environment variables:

```bash
# Default: http://localhost:8000/api/v1/login/
BACKEND_AUTH_URL=http://api.omnimap.cloud.ru/api/v1/login/
```

### Hot Reload Models

After editing models in the admin panel, reload the model registry:

```bash
curl -X POST http://localhost:8001/api/v1/admin/models/reload \
  -H "Authorization: Bearer $JWT_TOKEN"
```

Or use the admin panel to make changes and call the reload endpoint.

## API Documentation

When the server is running, API documentation is available at:

- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`
- OpenAPI JSON: `http://localhost:8001/openapi.json`

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SQL_USER` | PostgreSQL username | required |
| `SQL_PASSWD` | PostgreSQL password | required |
| `SQL_NAME` | PostgreSQL database name | required |
| `SQL_HOST` | PostgreSQL host | localhost |
| `SQL_PORT` | PostgreSQL port | 5432 |
| `DJANGO_SECRET_KEY` | JWT signing key (shared with omnimap-back) | required |
| `BACKEND_AUTH_URL` | omnimap-back login endpoint | http://localhost:8000/api/v1/login/ |
| `OPENAI_API_KEY` | OpenAI API key | optional |
| `ANTHROPIC_API_KEY` | Anthropic API key | optional |
| `GIGACHAT_AUTH_KEY` | GigaChat auth key (base64) | optional |
| `REDIS_URL` | Redis connection URL | optional |
| `DEBUG` | Enable debug mode | false |

## Project Status

This project is under active development. Epics are implemented sequentially according to the architecture defined in `snapshot.json`.

Current progress tracked in `/packs` directory.
