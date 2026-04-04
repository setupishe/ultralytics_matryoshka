#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:-vanilla}"
shift || true

MONITOR_INTERVAL="${MONITOR_INTERVAL:-2}"
MONITOR_LOG="${MONITOR_LOG:-${ROOT_DIR}/monitor_${VARIANT}_$(date +%Y%m%d_%H%M%S).log}"

cleanup() {
  if [[ -n "${MONITOR_PID:-}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

echo "Dummy commands while you are away:"
echo "  nvidia-smi"
echo "  ps -eo pid,ppid,ni,%cpu,%mem,etime,cmd --sort=-%cpu | sed -n '1,16p'"
echo "  tail -f \"${MONITOR_LOG}\""
echo ""

"${ROOT_DIR}/monitor_train.sh" "${MONITOR_INTERVAL}" "${MONITOR_LOG}" &
MONITOR_PID=$!

echo "Started monitor PID ${MONITOR_PID}"
echo "Monitor log: ${MONITOR_LOG}"
echo ""

"${ROOT_DIR}/scripts/run_coco_strict_train.sh" "${VARIANT}" "$@"
