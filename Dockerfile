FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY rageval ./rageval
COPY api ./api
RUN pip install --no-cache-dir ".[all]"

COPY configs ./configs
COPY data ./data
COPY examples ./examples

RUN useradd --create-home appuser && mkdir -p /app/runs && chown appuser /app/runs
USER appuser

ENV RAGEVAL_CONFIG=configs/default.yaml
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
