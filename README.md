# LLM Gateway

Шлюз для маршрутизации запросов от веб-чата к разным LLM-провайдерам.

- Авторизация по JWT через существующий Django REST API (`AUTH_VERIFY_URL`).
- Опциональная подгрузка контекста из Django (`DATA_BASE_URL`) и/или из Postgres.
- Абстракция провайдеров LLM (OpenAI-совместимые, Ollama, заглушка).
- Нестрим и стрим-эндпоинты (`/chat`, `/chat/stream`).
- Логирование запросов/латентности в Postgres.
- Простой rate limiting per-user.

