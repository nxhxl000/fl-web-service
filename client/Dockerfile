FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY fl_client ./fl_client

ENTRYPOINT ["python", "-m", "fl_client"]
