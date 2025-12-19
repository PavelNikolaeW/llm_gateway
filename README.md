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

### Testing

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

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
