# Serving image: the API and the champion bundle, without training dependencies
FROM python:3.13-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --prefix=/install ".[serving]"


FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /install /usr/local
COPY configs/ ./configs/
COPY artifacts/champion/ ./artifacts/champion/

RUN useradd -m app && mkdir -p logs && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "churn.serving.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
