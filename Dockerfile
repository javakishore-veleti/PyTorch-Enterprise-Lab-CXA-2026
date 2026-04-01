FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/

RUN pip install --no-cache-dir -e ".[serve]" 2>/dev/null || pip install --no-cache-dir fastapi uvicorn pydantic torch --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "quantedge_services.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
