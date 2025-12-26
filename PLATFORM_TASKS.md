# Задачи для omnimap-platform

Этот файл содержит задачи для обновления docker-compose и Kubernetes конфигурации после добавления админ-панели в llm-gateway.

## infrastructure/docker-compose.yml

### 1. Добавить переменные окружения для llm-gateway

```yaml
services:
  llm-gateway:
    environment:
      # ... существующие переменные ...

      # Admin Panel - URL для аутентификации через omnimap-back
      - BACKEND_AUTH_URL=http://omnimap-back:8000/api/v1/login/
```

**Важно:** Используй внутреннее DNS имя сервиса `omnimap-back`, а не `localhost`.

### 2. Проверить сетевые настройки

Убедись, что `llm-gateway` и `omnimap-back` находятся в одной Docker сети и могут общаться друг с другом.

## deploy/kubernetes/

### 3. Обновить ConfigMap для llm-gateway

```yaml
# deploy/kubernetes/base/llm-gateway/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-gateway-config
data:
  # ... существующие настройки ...
  BACKEND_AUTH_URL: "http://omnimap-back-service:8000/api/v1/login/"
```

### 4. Обновить overlays для разных окружений

**Dev overlay:**
```yaml
# deploy/kubernetes/overlays/dev/llm-gateway-patch.yaml
- op: add
  path: /data/BACKEND_AUTH_URL
  value: "http://omnimap-back-service:8000/api/v1/login/"
```

**Prod overlay:**
```yaml
# deploy/kubernetes/overlays/prod/llm-gateway-patch.yaml
- op: add
  path: /data/BACKEND_AUTH_URL
  value: "http://api.omnimap.cloud.ru/api/v1/login/"
```

### 5. Добавить Ingress для admin панели (опционально)

Если нужен прямой доступ к админке через домен:

```yaml
# deploy/kubernetes/base/llm-gateway/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llm-gateway-admin-ingress
spec:
  rules:
  - host: llm-admin.omnimap.cloud.ru
    http:
      paths:
      - path: /admin
        pathType: Prefix
        backend:
          service:
            name: llm-gateway-service
            port:
              number: 8001
```

## Проверка после деплоя

После применения изменений проверь:

1. **Доступность админки:**
   ```bash
   curl -I http://llm-gateway:8001/admin/login
   # Должен вернуть 200 OK с HTML
   ```

2. **Связь с omnimap-back:**
   ```bash
   # Из контейнера llm-gateway
   curl http://omnimap-back:8000/api/v1/login/ -X POST \
     -H "Content-Type: application/json" \
     -d '{"username":"admin","password":"test"}'
   ```

3. **Логин в админку:**
   - Открой `http://localhost:8001/admin`
   - Введи credentials администратора
   - Должна открыться панель управления

## Зависимости между сервисами

```
┌─────────────────┐      ┌─────────────────┐
│   llm-gateway   │──────│  omnimap-back   │
│    (FastAPI)    │ HTTP │    (Django)     │
│   port: 8001    │      │   port: 8000    │
└─────────────────┘      └─────────────────┘
         │                       │
         │                       │
         ▼                       ▼
    ┌─────────┐           ┌─────────────┐
    │ Postgres│           │   Postgres  │
    │ llm_db  │           │  omnimap_db │
    └─────────┘           └─────────────┘
```

**Порядок запуска:**
1. PostgreSQL (оба)
2. omnimap-back (должен быть ready для аутентификации)
3. llm-gateway

## Статус

- [ ] Обновлён docker-compose.yml
- [ ] Обновлены Kubernetes манифесты
- [ ] Проверена связь между сервисами
- [ ] Протестирован логин в админку
