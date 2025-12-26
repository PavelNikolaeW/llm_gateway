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

## Admin Panel

Админ-панель доступна по адресу `/admin` и использует SQLAdmin.

### Структура модуля `src/admin/`

```
src/admin/
├── __init__.py      # Экспортирует setup_admin()
├── auth.py          # JWT аутентификация через omnimap-back
├── setup.py         # Конфигурация SQLAdmin
├── views.py         # ModelView для всех сущностей
└── templates/
    └── login.html   # Кастомная форма входа
```

### Аутентификация

- Форма логина запрашивает username/password
- Credentials проверяются через `omnimap-back /api/v1/login/`
- Требуется `is_staff=True` для доступа
- JWT токен сохраняется в сессии

### Настройки

```bash
# URL для аутентификации (по умолчанию localhost)
BACKEND_AUTH_URL=http://localhost:8000/api/v1/login/
```

### Hot Reload моделей

После изменения моделей в админке нужно перезагрузить реестр:

```bash
POST /api/v1/admin/models/reload
Authorization: Bearer $JWT_TOKEN
```

### Добавление новых сущностей

1. Создай модель в `src/data/models.py`
2. Создай миграцию: `alembic revision --autogenerate -m "Add new model"`
3. Добавь ModelView в `src/admin/views.py`
4. Зарегистрируй в `src/admin/setup.py`

## Cross-Service Changes (ВАЖНО!)

**НИКОГДА не изменяй код других сервисов напрямую!**

Если изменения в llm-gateway требуют изменений в других сервисах:

1. **НЕ редактируй** файлы в `omnimap-back`, `omnimap-front` или `omnimap-sync`
2. **Создай файл задач** `FRONTEND_TASKS.md` или `BACKEND_TASKS.md` в корне этого репозитория
3. **В PR укажи**, что требуются изменения в других сервисах
4. Агент, работающий над соответствующим сервисом, выполнит задачи

## Incoming Tasks

Проверь файл `GATEWAY_TASKS.md` (если существует) — там могут быть задачи от фронтенда или других сервисов.
