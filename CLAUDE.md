# Claude Code Rules

## Before Committing

**ALWAYS run all tests before making a commit:**

```bash
pytest tests/ -v --tb=short
```

- All tests must pass (0 failed, 0 errors)
- If tests fail, fix the issues before committing
- Do not commit broken code

## Project Structure

- `src/` - source code
- `tests/unit/` - unit tests with mocked dependencies
- `tests/integration/` - integration tests with real database
- `packs/` - epic definitions (PACK-001.json, PACK-002.json, etc.)

## Workflow

1. Read the pack requirements
2. Implement the feature
3. Write unit tests
4. Write integration tests
5. Run ALL tests: `pytest tests/`
6. Fix any failures
7. Commit only when all tests pass
