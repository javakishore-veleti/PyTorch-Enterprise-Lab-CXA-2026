#!/usr/bin/env bash
# QuantEdge — Bring up all infrastructure + services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "==> QuantEdge: Starting all infrastructure"

echo "--- [1/3] PostgreSQL (local dev)"
docker compose -f "${REPO_ROOT}/DevOps/Local/postgres/docker-compose.yml" up -d

echo "--- [2/3] MLflow tracking server"
docker compose -f "${REPO_ROOT}/DevOps/MLOps/mlflow/docker-compose.yml" up -d

echo "--- [3/3] QuantEdge API"
bash "${SCRIPT_DIR}/app-api.sh" start

echo ""
echo "✅ All services up"
echo "   PostgreSQL : localhost:5432"
echo "   MLflow UI  : http://localhost:5000"
echo "   API        : http://localhost:8000  (docs: /docs)"
