# Claude Code Rules

## Git Workflow

**IMPORTANT: Разработка ведётся в отдельных ветках!**

1. **Создай ветку** для задачи:
   ```bash
   git checkout -b feature/название-задачи
   # или
   git checkout -b fix/описание-бага
   ```

2. **Работай в своей ветке** — никогда не коммить напрямую в `main`

3. **После завершения задачи** создай Pull Request:
   ```bash
   git push -u origin feature/название-задачи
   gh pr create --title "Описание" --body "Детали изменений"
   ```

4. **Дождись ревью** от Claude Code Action перед мержем

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

1. Create a feature branch
2. Read the pack requirements
3. Implement the feature
4. Write unit tests
5. Write integration tests
6. Run ALL tests: `pytest tests/`
7. Fix any failures
8. Commit only when all tests pass
9. Push and create a Pull Request
