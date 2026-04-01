#!/usr/bin/env bash
# QuantEdge — Angular portal control (customer | admin) (start | stop | status)

set -euo pipefail

PORTAL="${1:-customer}"
ACTION="${2:-status}"

case "${PORTAL}" in
  customer)
    PORTAL_DIR="portals/quantedge-portal-customer"
    PORT=4200
    ;;
  admin)
    PORTAL_DIR="portals/quantedge-portal-admin"
    PORT=4201
    ;;
  *)
    echo "Usage: $0 [customer|admin] [start|stop|status]"
    exit 1
    ;;
esac

PID_FILE="/tmp/quantedge-portal-${PORTAL}.pid"
LOG_FILE="/tmp/quantedge-portal-${PORTAL}.log"

case "${ACTION}" in
  start)
    if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      echo "⚠️  ${PORTAL} portal already running (PID $(cat "${PID_FILE}"))"
      exit 0
    fi
    echo "==> Starting QuantEdge ${PORTAL} portal on port ${PORT}..."
    cd "${PORTAL_DIR}"
    npx ng serve --port "${PORT}" >> "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "✅ ${PORTAL} portal starting — http://localhost:${PORT}"
    ;;
  stop)
    if [ -f "${PID_FILE}" ]; then
      echo "==> Stopping ${PORTAL} portal (PID $(cat "${PID_FILE}"))..."
      kill "$(cat "${PID_FILE}")" 2>/dev/null || true
      rm -f "${PID_FILE}"
      echo "✅ ${PORTAL} portal stopped"
    else
      echo "⚠️  ${PORTAL} portal is not running"
    fi
    ;;
  status)
    if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      echo "✅ ${PORTAL} portal running (PID $(cat "${PID_FILE}")) — http://localhost:${PORT}"
    else
      echo "⛔ ${PORTAL} portal is not running"
    fi
    ;;
esac
