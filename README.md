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

## Project Status

This project is under active development. Epics are implemented sequentially according to the architecture defined in `snapshot.json`.

Current progress tracked in `/packs` directory.
