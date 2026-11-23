FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY agents/ ./agents/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e . && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "agent_builder.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
