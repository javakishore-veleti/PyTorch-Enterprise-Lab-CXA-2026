#!/usr/bin/env bash
# QuantEdge — FastAPI app control (start | stop | status)

set -euo pipefail

ACTION="${1:-status}"
PID_FILE="/tmp/quantedge-api.pid"
LOG_FILE="/tmp/quantedge-api.log"
VENV_PATH="${HOME}/runtime_data/python_venvs/PyTorchMastery/PyTorch-Enterprise-Lab-CXA-2026"

case "${ACTION}" in
  start)
    if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      echo "⚠️  API already running (PID $(cat "${PID_FILE}"))"
      exit 0
    fi
    echo "==> Starting QuantEdge API..."
    conda run --prefix "${VENV_PATH}" \
      uvicorn quantedge_services.api.app:app \
        --host 0.0.0.0 --port 8000 --reload \
        >> "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    sleep 2
    echo "✅ API started (PID $(cat "${PID_FILE}")) — http://localhost:8000"
    echo "   Docs: http://localhost:8000/docs"
    ;;
  stop)
    if [ -f "${PID_FILE}" ]; then
      echo "==> Stopping QuantEdge API (PID $(cat "${PID_FILE}"))..."
      kill "$(cat "${PID_FILE}")" 2>/dev/null || true
      rm -f "${PID_FILE}"
      echo "✅ API stopped"
    else
      echo "⚠️  API is not running"
    fi
    ;;
  status)
    if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      echo "✅ API is running (PID $(cat "${PID_FILE}")) — http://localhost:8000"
    else
      echo "⛔ API is not running"
    fi
    ;;
  *)
    echo "Usage: $0 [start|stop|status]"
    exit 1
    ;;
esac
