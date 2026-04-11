# fl-client

Docker-образ федеративного клиента. На Day 4 умеет только heartbeat —
раз в `FL_HEARTBEAT_INTERVAL` секунд (по умолчанию 30) POST'ит
`/client/heartbeat` на бэкенд с opaque-токеном в `Authorization` заголовке.
Бэкенд обновляет `last_seen_at` соответствующего клиентского токена, и в
админском UI проекта видно, что клиент жив.

Flower SuperNode, обучение, загрузчики данных и пр. добавятся на Day 5–6.

## Сборка

```bash
cd client
docker build -t fl-client:dev .
```

Образ без зависимостей кроме stdlib Python 3.12, собирается секунды.

## Запуск против локального бэкенда

```bash
docker run --rm --network host \
  -e FL_TOKEN=flwc_... \
  -e FL_SERVER_URL=http://localhost:8000 \
  fl-client:dev
```

`--network host` даёт контейнеру доступ к `localhost:8000` WSL2-хоста, где
крутится uvicorn. На чистом Docker Desktop без WSL2 можно использовать
`host.docker.internal` вместо `localhost` и убрать `--network host`.

## Переменные окружения

| Переменная | Обязательно | По умолчанию | Описание |
|---|---|---|---|
| `FL_TOKEN` | да | — | Opaque-токен клиента (`flwc_...`), создаётся в UI проекта |
| `FL_SERVER_URL` | да | — | Базовый URL бэкенда, например `http://localhost:8000` |
| `FL_HEARTBEAT_INTERVAL` | нет | `30` | Период heartbeat в секундах |

## Коды выхода

- `0` — получен SIGTERM/SIGINT, graceful shutdown
- `1` — сервер вернул 401 (токен недействителен или удалён)
- `2` — не задана обязательная env переменная
