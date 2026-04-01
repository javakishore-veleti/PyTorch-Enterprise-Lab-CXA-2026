#!/usr/bin/env bash
# QuantEdge — Status of all infrastructure + services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=============================="
echo " QuantEdge — Service Status   "
echo "=============================="

echo ""
echo "[PostgreSQL]"
docker compose -f "${REPO_ROOT}/DevOps/Local/postgres/docker-compose.yml" ps 2>/dev/null || echo "⛔ Not running or Docker unavailable"

echo ""
echo "[MLflow]"
docker compose -f "${REPO_ROOT}/DevOps/MLOps/mlflow/docker-compose.yml" ps 2>/dev/null || echo "⛔ Not running or Docker unavailable"

echo ""
echo "[QuantEdge API]"
bash "${SCRIPT_DIR}/app-api.sh" status

echo ""
echo "[QuantEdge Customer Portal]"
bash "${SCRIPT_DIR}/app-portal.sh" customer status

echo ""
echo "[QuantEdge Admin Portal]"
bash "${SCRIPT_DIR}/app-portal.sh" admin status

echo ""
