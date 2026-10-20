# Pipeline image: training, batch scoring, monitoring and the MLflow UI
FROM python:3.13-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --prefix=/install ".[serving,pipeline]"


FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg

COPY --from=builder /install /usr/local
COPY configs/ ./configs/

ENTRYPOINT ["churn"]
CMD ["simulate"]
