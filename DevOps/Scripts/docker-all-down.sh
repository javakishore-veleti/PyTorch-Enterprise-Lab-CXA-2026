#!/usr/bin/env bash
# QuantEdge — Tear down all infrastructure + services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "==> QuantEdge: Stopping all services"

echo "--- [1/3] QuantEdge API"
bash "${SCRIPT_DIR}/app-api.sh" stop

echo "--- [2/3] MLflow tracking server"
docker compose -f "${REPO_ROOT}/DevOps/MLOps/mlflow/docker-compose.yml" down

echo "--- [3/3] PostgreSQL (local dev)"
docker compose -f "${REPO_ROOT}/DevOps/Local/postgres/docker-compose.yml" down

echo "✅ All services stopped"
