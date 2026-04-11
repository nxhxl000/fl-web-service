# client

Docker-образ федеративного клиента `fl-client`. Обёртка над Flower SuperNode
и кодом клиента из flwr-репо: регистрация узла через API веб-сервиса, запуск
обучения, отправка heartbeat'ов.

Цели: multi-stage build на `python:3.12-slim`, torch-CPU wheel, размер образа
≤ 800 МБ. Никакого GPU-образа — все реальные узлы работают на CPU.

`Dockerfile` и entrypoint появятся на Дне 4 дорожной карты.
